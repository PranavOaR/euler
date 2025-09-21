"use client"

import React from 'react';

function MinimalFooter() {
  return (
    <footer className="relative border-t border-white/10 bg-gray-900/80 text-white transition-colors duration-300">
      <div className="container mx-auto px-4 py-6">
        <div className="flex flex-col items-center justify-between gap-4 text-center md:flex-row">
          <p className="text-sm text-white/70">
            © 2024 Euler. All rights reserved.
          </p>
          <nav className="flex gap-6 text-sm">
            <a href="/privacy" className="transition-colors hover:text-white text-white/70">
              Privacy Policy
            </a>
            <a href="/terms" className="transition-colors hover:text-white text-white/70">
              Terms of Service
            </a>
          </nav>
        </div>
      </div>
    </footer>
  );
}

export { MinimalFooter };