"use client"

import React, { useEffect, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { LucideIcon, LayoutDashboard } from "lucide-react"
import { cn } from "@/lib/utils"
import { ShimmerButton } from "@/components/ui/shimmer-button"
import { useAuthStatus } from "@/context/AuthContext"
import { signOutUser } from "@/hooks/useAuth"

interface NavItem {
  name: string
  url: string
  icon: LucideIcon
}

interface NavBarProps {
  items: NavItem[]
  className?: string
  defaultActive?: string
}

export function AnimeNavBar({ items, className, defaultActive = "Home" }: NavBarProps) {
  const pathname = usePathname()
  const { user, loading } = useAuthStatus()
  const [mounted, setMounted] = useState(false)
  const [hoveredTab, setHoveredTab] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<string>(defaultActive)
  const [isMobile, setIsMobile] = useState(false)

  // Add Dashboard to nav items for authenticated users
  const navItems = user ? [
    ...items,
    { name: "Dashboard", url: "/dashboard", icon: LayoutDashboard }
  ] : items

  const handleSignOut = async () => {
    await signOutUser()
  }

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768)
    }

    handleResize()
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  if (!mounted) return null

  return (
    <div className="fixed top-3 left-0 right-0 z-[9999] px-4">
      <div className="flex justify-center">
        <motion.div 
          className="flex items-center gap-1 md:gap-3 bg-black/50 border border-white/10 backdrop-blur-lg py-2 px-2 rounded-full shadow-lg relative max-w-full overflow-hidden"
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{
            type: "spring",
            stiffness: 260,
            damping: 20,
          }}
        >
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = activeTab === item.name
            const isHovered = hoveredTab === item.name

            return (
              <Link
                key={item.name}
                href={item.url}
                onClick={(e) => {
                  setActiveTab(item.name)
                  
                  // Handle smooth scrolling for anchor links
                  if (item.url.startsWith('#')) {
                    e.preventDefault()
                    const element = document.querySelector(item.url)
                    if (element) {
                      element.scrollIntoView({ 
                        behavior: 'smooth',
                        block: 'start'
                      })
                    }
                  }
                }}
                onMouseEnter={() => setHoveredTab(item.name)}
                onMouseLeave={() => setHoveredTab(null)}
                className={cn(
                  "relative cursor-pointer text-sm font-semibold px-3 md:px-6 py-2 md:py-3 rounded-full transition-all duration-300 flex items-center justify-center min-w-0",
                  "text-white/70 hover:text-white",
                  isActive && "text-white"
                )}
              >
                {isActive && (
                  <motion.div
                    className="absolute inset-0 rounded-full -z-10 overflow-hidden"
                    initial={{ opacity: 0 }}
                    animate={{ 
                      opacity: [0.3, 0.5, 0.3],
                      scale: [1, 1.03, 1]
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                  >
                    <div className="absolute inset-0 bg-primary/25 rounded-full blur-md" />
                    <div className="absolute inset-[-4px] bg-primary/20 rounded-full blur-xl" />
                    <div className="absolute inset-[-8px] bg-primary/15 rounded-full blur-2xl" />
                    <div className="absolute inset-[-12px] bg-primary/5 rounded-full blur-3xl" />
                    
                    <div 
                      className="absolute inset-0 bg-gradient-to-r from-primary/0 via-primary/20 to-primary/0"
                      style={{
                        animation: "shine 3s ease-in-out infinite"
                      }}
                    />
                  </motion.div>
                )}

                <motion.span
                  className="hidden md:inline relative z-10"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.2 }}
                >
                  {item.name}
                </motion.span>
                <motion.span 
                  className="md:hidden relative z-10 flex items-center justify-center"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <Icon size={16} strokeWidth={2.5} />
                </motion.span>
          
                <AnimatePresence>
                  {isHovered && !isActive && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      className="absolute inset-0 bg-white/10 rounded-full -z-10"
                    />
                  )}
                </AnimatePresence>
              </Link>
            )
          })}
          
          {/* Conditional Login/Signup or Logout button */}
          {!loading && (
            <div className="flex items-center gap-1 md:gap-2 ml-1 md:ml-2 pl-1 md:pl-2 border-l border-white/20">
              {user ? (
                <button
                  onClick={handleSignOut}
                  className="relative px-2 md:px-4 py-2 text-xs md:text-sm font-medium text-white bg-white/10 hover:bg-white/20 border border-white/20 rounded-full transition-colors overflow-hidden group whitespace-nowrap"
                >
                  <span className="relative z-10">Logout</span>
                  <div className="absolute inset-0 -top-full group-hover:top-full transition-all duration-1000 bg-gradient-to-r from-transparent via-white/20 to-transparent transform skew-x-12"></div>
                </button>
              ) : (
                <Link
                  href="/login"
                  className="relative px-2 md:px-4 py-2 text-xs md:text-sm font-medium text-white bg-white/10 hover:bg-white/20 border border-white/20 rounded-full transition-colors overflow-hidden group whitespace-nowrap"
                >
                  <span className="relative z-10 hidden sm:inline">Login/Signup</span>
                  <span className="relative z-10 sm:hidden">Login</span>
                  <div className="absolute inset-0 -top-full group-hover:top-full transition-all duration-1000 bg-gradient-to-r from-transparent via-white/20 to-transparent transform skew-x-12"></div>
                </Link>
              )}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  )
}