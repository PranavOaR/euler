"use client";

import React from 'react';
import { Moon, Sun } from 'lucide-react';
import { useTheme } from '@/contexts/theme-context';

const ThemeToggle: React.FC = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      className="relative inline-flex items-center justify-center w-8 h-8 md:w-10 md:h-10 rounded-full bg-white/10 hover:bg-white/20 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-white/50 focus:ring-offset-0 border border-white/10"
      aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
    >
      {theme === 'dark' ? (
        <Sun className="w-4 h-4 md:w-5 md:h-5 text-yellow-400 transition-all duration-300 rotate-0 scale-100" />
      ) : (
        <Moon className="w-4 h-4 md:w-5 md:h-5 text-blue-400 transition-all duration-300 rotate-0 scale-100" />
      )}
    </button>
  );
};

export default ThemeToggle;