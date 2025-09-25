"use client"

import React from 'react'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import { 
  LayoutDashboard, 
  TrendingUp, 
  BarChart3, 
  Calculator,
  LogOut,
  Sun,
  Moon
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useAuthStatus } from '@/context/AuthContext'
import { signOutUser } from '@/hooks/useAuth'
import { useTheme } from '@/contexts/theme-context'
import { 
  Sidebar, 
  SidebarBody, 
  SidebarLink,
  useSidebar 
} from '@/components/ui/aceternity-sidebar'

const sidebarItems = [
  {
    label: 'Overview',
    href: '/dashboard',
    icon: <LayoutDashboard className="text-neutral-700 dark:text-neutral-200 h-5 w-5 flex-shrink-0" />
  },
  {
    label: 'Prediction',
    href: '/dashboard/prediction',
    icon: <TrendingUp className="text-neutral-700 dark:text-neutral-200 h-5 w-5 flex-shrink-0" />
  },
  {
    label: 'Financials',
    href: '/dashboard/financials',
    icon: <Calculator className="text-neutral-700 dark:text-neutral-200 h-5 w-5 flex-shrink-0" />
  },
  {
    label: 'Simulation',
    href: '/dashboard/simulation',
    icon: <BarChart3 className="text-neutral-700 dark:text-neutral-200 h-5 w-5 flex-shrink-0" />
  }
]

function SidebarContent() {
  const { user } = useAuthStatus()
  const { open } = useSidebar()
  const { theme, toggleTheme } = useTheme()
  const pathname = usePathname()

  const handleSignOut = async () => {
    await signOutUser()
  }

  return (
    <div className="flex flex-col h-full justify-between">
      {/* Top Section */}
      <div className="space-y-6">
        {/* Header Section */}
        <div className="flex flex-col items-start">
          <motion.div
            animate={{
              display: open ? "block" : "none",
              opacity: open ? 1 : 0,
            }}
            className="space-y-1"
          >
            <h2 className="text-xl font-bold text-neutral-800 dark:text-neutral-200">
              Euler
            </h2>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              AI Trading Dashboard
            </p>
          </motion.div>
          
          {!open && (
            <div className="w-8 h-8 rounded-full bg-neutral-300 dark:bg-neutral-600 flex items-center justify-center text-neutral-700 dark:text-neutral-200 text-sm font-semibold mx-auto">
              E
            </div>
          )}
        </div>

        {/* Navigation Links */}
        <div className="flex flex-col space-y-2">
          {sidebarItems.map((item) => {
            const isActive = pathname === item.href
            return (
              <SidebarLink
                key={item.href}
                link={item}
                className={cn(
                  "px-3 py-3 rounded-lg transition-all duration-200 flex items-center justify-start",
                  "hover:bg-neutral-200 dark:hover:bg-neutral-700",
                  !open && "justify-center", // Center icons when collapsed
                  isActive && "bg-blue-100 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800"
                )}
              />
            )
          })}
        </div>
      </div>

      {/* Bottom Section */}
      <div className="space-y-4">

        {/* User Profile */}
        {user && (
          <div className={cn(
            "flex items-center gap-2 group/sidebar py-2 px-3 py-3 rounded-lg transition-all duration-200 hover:bg-neutral-200 dark:hover:bg-neutral-700",
            open ? "justify-start" : "justify-center"
          )}>
            <div className="w-8 h-8 rounded-full bg-neutral-400 dark:bg-neutral-500 flex items-center justify-center text-white text-sm font-semibold flex-shrink-0">
              {user.email?.[0]?.toUpperCase() || 'U'}
            </div>
            
            <motion.div
              animate={{
                display: open ? "block" : "none",
                opacity: open ? 1 : 0,
              }}
              className="flex-1 min-w-0"
            >
              <div className="text-sm font-medium text-neutral-700 dark:text-neutral-200 truncate">
                {user.displayName || user.email?.split('@')[0] || 'User'}
              </div>
              <div className="text-xs text-neutral-500 dark:text-neutral-400 truncate">
                {user.email}
              </div>
            </motion.div>
          </div>
        )}

        {/* Sign Out Button */}
        <button
          onClick={handleSignOut}
          className={cn(
            "w-full flex items-center gap-2 group/sidebar py-2 px-3 py-3 rounded-lg transition-all duration-200 hover:bg-red-100 dark:hover:bg-red-900/30 text-neutral-700 dark:text-neutral-200 hover:text-red-600 dark:hover:text-red-400",
            open ? "justify-start" : "justify-center"
          )}
        >
          <LogOut className="h-5 w-5 flex-shrink-0" />
          <motion.span
            animate={{
              display: open ? "inline-block" : "none",
              opacity: open ? 1 : 0,
            }}
            className="text-sm group-hover/sidebar:translate-x-1 transition duration-150 whitespace-pre inline-block !p-0 !m-0"
          >
            Sign Out
          </motion.span>
        </button>
      </div>
    </div>
  )
}

export default function DashboardSidebar() {
  const [open, setOpen] = React.useState(false)

  return (
    <Sidebar open={open} setOpen={setOpen}>
      <SidebarBody className="justify-start gap-0">
        <SidebarContent />
      </SidebarBody>
    </Sidebar>
  )
}
