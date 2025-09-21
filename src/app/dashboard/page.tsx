"use client"

import React, { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStatus } from '@/context/AuthContext';
import { AnimeNavBar } from '@/components/ui/anime-navbar';
import Footer from '@/components/Footer';
import ComingSoon from '@/components/ui/coming-soon';
import Beams from '@/components/ui/Beams';
import { Home, Zap, Users } from 'lucide-react';

const navItems = [
  { name: "Home", url: "/", icon: Home },
  { name: "Features", url: "/features", icon: Zap },
  { name: "About Us", url: "/about", icon: Users },
];

export default function DashboardPage() {
  const { user, loading } = useAuthStatus();
  const router = useRouter();
  const launchDate = new Date('2025-12-31T23:59:59');

  useEffect(() => {
    // If auth state is not loading and there's no user, redirect to login
    if (!loading && !user) {
      router.push('/login');
    }
  }, [user, loading, router]);

  // Show a loading state while checking auth status
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-black">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white"></div>
      </div>
    );
  }

  // If no user, show nothing (redirect is happening)
  if (!user) {
    return null;
  }
  
  return (
    <div className="relative min-h-screen">
      {/* Beams Background */}
      <div className="absolute inset-0 w-full h-full">
        <Beams
          beamWidth={2}
          beamHeight={15}
          beamNumber={12}
          lightColor="#ffffff"
          speed={2}
          noiseIntensity={1.75}
          scale={0.2}
          rotation={0}
        />
      </div>
      
      {/* Navigation Bar */}
      <AnimeNavBar items={navItems} defaultActive="Dashboard" />
      
      {/* Coming Soon Content */}
      <ComingSoon
        launchDate={launchDate}
        title="Welcome to Euler Dashboard"
        description="We're crafting an extraordinary AI-powered trading dashboard that will revolutionize your investment strategy. Get ready for the future of automated finance!"
        showNewsletter={true}
      />
      
      {/* Footer */}
      <Footer />
    </div>
  );
}