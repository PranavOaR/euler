"use client"

import React from 'react';
import { AnimeNavBar } from '@/components/ui/anime-navbar';
import Footer from '@/components/Footer';
import Silk from '@/components/Silk';
import { Home, Zap, Users } from 'lucide-react';

const navItems = [
  { name: "Home", url: "/", icon: Home },
  { name: "Features", url: "/features", icon: Zap },
  { name: "About Us", url: "/about", icon: Users },
];

export default function HomePage() {
  return (
    <div className="relative min-h-screen">
      {/* Navigation Bar */}
      <AnimeNavBar items={navItems} defaultActive="Home" />
      
      {/* Hero Section with Silk Background */}
      <div className="relative h-screen flex items-center justify-center overflow-hidden">
        {/* Silk Animated Background */}
        <div className="absolute inset-0 w-full h-full">
          <Silk
            speed={5}
            scale={1}
            color="#7B7481"
            noiseIntensity={1.5}
            rotation={0}
          />
        </div>
        
        {/* Hero Content */}
        <div className="relative z-10 text-center px-4">
          <h1 className="text-6xl md:text-8xl font-bold text-white mb-6 tracking-tight drop-shadow-2xl">
            Euler
          </h1>
          <p className="text-xl md:text-2xl text-white/90 max-w-2xl mx-auto leading-relaxed drop-shadow-lg">
            Building the Future of Automated Capital
          </p>
        </div>
        
        {/* Overlay for better text readability */}
        <div className="absolute inset-0 bg-black/20 z-0"></div>
      </div>
      
      {/* Footer */}
      <Footer />
    </div>
  );
}