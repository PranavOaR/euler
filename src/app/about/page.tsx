"use client"

import React from 'react';
import { AnimeNavBar } from '@/components/ui/anime-navbar';
import Footer from '@/components/Footer';
import Silk from '@/components/Silk';
import { SplineSceneBasic } from '@/components/ui/spline-scene-basic';
import { Home, Zap, Users } from 'lucide-react';

const navItems = [
  { name: "Home", url: "/", icon: Home },
  { name: "Features", url: "/features", icon: Zap },
  { name: "About Us", url: "/about", icon: Users },
];

export default function AboutPage() {
  return (
    <div className="relative min-h-screen">
      {/* Navigation Bar */}
      <AnimeNavBar items={navItems} defaultActive="About Us" />
      
      {/* About Us Section */}
      <section className="relative pt-32 pb-24 px-4 overflow-hidden min-h-screen">
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
        
        <div className="relative z-10 max-w-7xl mx-auto">
          {/* Main Content with 3D Interactive Component */}
          <div className="grid lg:grid-cols-2 gap-12 items-center min-h-[80vh]">
            {/* Left Content */}
            <div className="space-y-8">
              <div className="space-y-6">
                <h1 className="text-5xl md:text-7xl font-light tracking-wide bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400">
                  About Euler
                </h1>
                <div className="w-24 h-1 bg-gradient-to-r from-purple-500 to-blue-500"></div>
              </div>
              
              <div className="space-y-6">
                <p className="text-xl md:text-2xl text-white/90 leading-relaxed font-light tracking-wide">
                  Euler is where advanced AI meets automated finance. Our mission is to democratize sophisticated financial technology, giving a new generation of investors access to powerful tools once reserved for institutions.
                </p>
                
                <p className="text-lg md:text-xl text-white/80 leading-relaxed font-light">
                  By combining cutting-edge machine learning with robotic precision, Euler provides a seamless platform for intelligent financial insights and automation, redefining what&apos;s possible in the market.
                </p>
              </div>
              
              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                  <span className="text-white/90 font-light tracking-wide">Democratize sophisticated financial technology</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                  <span className="text-white/90 font-light tracking-wide">Intelligent automation and machine learning</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span className="text-white/90 font-light tracking-wide">Institutional-grade AI trading tools</span>
                </div>
              </div>
            </div>

            {/* Right Content - 3D Interactive Component */}
            <div className="lg:pl-8">
              <SplineSceneBasic />
            </div>
          </div>
        </div>
      </section>
      
      {/* Footer */}
      <Footer />
    </div>
  );
}