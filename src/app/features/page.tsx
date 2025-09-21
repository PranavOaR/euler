"use client"

import React from 'react';
import { AnimeNavBar } from '@/components/ui/anime-navbar';
import Footer from '@/components/Footer';
import { FeaturesSection } from '@/components/sections/features';
import { Home, Zap, Users } from 'lucide-react';

const navItems = [
  { name: "Home", url: "/", icon: Home },
  { name: "Features", url: "/features", icon: Zap },
  { name: "About Us", url: "/about", icon: Users },
];

export default function FeaturesPage() {
  return (
    <div className="relative min-h-screen">
      {/* Navigation Bar */}
      <AnimeNavBar items={navItems} defaultActive="Features" />
      
      {/* Features Section */}
      <FeaturesSection />
      
      {/* Footer */}
      <Footer />
    </div>
  );
}