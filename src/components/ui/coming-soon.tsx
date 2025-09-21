"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Clock, Calendar, Rocket, Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface TimeLeft {
  days: number;
  hours: number;
  minutes: number;
  seconds: number;
}

interface ComingSoonProps {
  launchDate?: Date;
  title?: string;
  description?: string;
  showNewsletter?: boolean;
  className?: string;
}

const ComingSoon: React.FC<ComingSoonProps> = ({
  launchDate = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days from now
  title = "Something Amazing is Coming Soon",
  description = "We're working hard to bring you an incredible experience. Stay tuned for the big reveal!",
  showNewsletter = true,
  className = "",
}) => {
  const [timeLeft, setTimeLeft] = useState<TimeLeft>({
    days: 0,
    hours: 0,
    minutes: 0,
    seconds: 0,
  });
  const [email, setEmail] = useState("");
  const [isSubscribed, setIsSubscribed] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const calculateTimeLeft = () => {
      const difference = launchDate.getTime() - new Date().getTime();
      
      if (difference > 0) {
        setTimeLeft({
          days: Math.floor(difference / (1000 * 60 * 60 * 24)),
          hours: Math.floor((difference / (1000 * 60 * 60)) % 24),
          minutes: Math.floor((difference / 1000 / 60) % 60),
          seconds: Math.floor((difference / 1000) % 60),
        });
      } else {
        setTimeLeft({ days: 0, hours: 0, minutes: 0, seconds: 0 });
      }
    };

    calculateTimeLeft();
    const timer = setInterval(calculateTimeLeft, 1000);

    return () => clearInterval(timer);
  }, [launchDate, mounted]);

  const handleNewsletterSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (email) {
      setIsSubscribed(true);
      setEmail("");
    }
  };

  const timeUnits = [
    { label: "Days", value: timeLeft.days, icon: Calendar },
    { label: "Hours", value: timeLeft.hours, icon: Clock },
    { label: "Minutes", value: timeLeft.minutes, icon: Clock },
    { label: "Seconds", value: timeLeft.seconds, icon: Clock },
  ];

  const AnimatedBlob = ({ color, position, delay = "" }: { color: string; position: string; delay?: string }) => (
    <div className={`absolute ${position} w-72 h-72 ${color} rounded-full mix-blend-screen filter blur-xl opacity-30 animate-pulse ${delay}`} />
  );

  if (!mounted) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-pulse">Loading...</div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen bg-transparent relative overflow-hidden ${className}`}>
      {/* Removed Animated Background Elements for cleaner look */}

      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center space-y-8 max-w-4xl mx-auto"
        >
          {/* Icon Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.6 }}
            className="inline-flex items-center justify-center w-16 h-16 bg-purple-500/10 rounded-full mb-6"
          >
            <Rocket className="w-8 h-8 text-purple-400" />
          </motion.div>

          {/* Title */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.8 }}
            className="text-4xl md:text-6xl font-bold text-white tracking-tight"
          >
            {title}
          </motion.h1>

          {/* Description */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.8 }}
            className="text-xl text-white/80 max-w-2xl mx-auto"
          >
            {description}
          </motion.p>

          {/* Countdown Timer */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.8 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-2xl mx-auto"
          >
            {timeUnits.map((unit, index) => (
              <Card key={unit.label} className="bg-black/50 backdrop-blur-sm border-white/10">
                <CardContent className="p-6 text-center">
                  <motion.div
                    key={unit.value}
                    initial={{ scale: 1.2, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.3 }}
                    className="text-3xl md:text-4xl font-bold text-white mb-2"
                  >
                    {unit.value.toString().padStart(2, '0')}
                  </motion.div>
                  <div className="text-sm text-white/60 font-medium">
                    {unit.label}
                  </div>
                </CardContent>
              </Card>
            ))}
          </motion.div>

          {/* Newsletter Signup */}
          {showNewsletter && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, duration: 0.8 }}
              className="max-w-md mx-auto"
            >
              <Card className="bg-black/80 backdrop-blur-sm border-white/10">
                <CardHeader>
                  <CardTitle className="text-center flex items-center justify-center gap-2 text-white">
                    <Star className="w-5 h-5 text-purple-400" />
                    Get Notified
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <AnimatePresence mode="wait">
                    {!isSubscribed ? (
                      <motion.form
                        key="form"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onSubmit={handleNewsletterSubmit}
                        className="space-y-4"
                      >
                        <div className="space-y-2">
                          <Label htmlFor="email" className="text-sm font-medium text-white">
                            Email Address
                          </Label>
                          <Input
                            id="email"
                            type="email"
                            placeholder="you@example.com"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            className="w-full bg-white/10 border-white/20 text-white placeholder:text-white/50"
                          />
                        </div>
                        <Button 
                          type="submit" 
                          className="w-full bg-black hover:bg-gray-900 text-white font-semibold text-base border border-white/20 relative overflow-hidden group"
                        >
                          <span className="relative z-10">Notify Me at Launch</span>
                          <div className="absolute inset-0 -top-full group-hover:top-full transition-all duration-1000 bg-gradient-to-r from-transparent via-white/20 to-transparent transform skew-x-12"></div>
                        </Button>
                      </motion.form>
                    ) : (
                      <motion.div
                        key="success"
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0 }}
                        className="text-center py-4"
                      >
                        <div className="text-2xl mb-2">🎉</div>
                        <h3 className="font-semibold text-white mb-1">
                          You're on the list!
                        </h3>
                        <p className="text-sm text-white/60">
                          We'll notify you as soon as we launch.
                        </p>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* Progress Dots */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8, duration: 0.6 }}
            className="flex justify-center gap-2 pt-8"
          >
            {[0, 1, 2].map((index) => (
              <div
                key={index}
                className={`w-2 h-2 rounded-full transition-all duration-300 ${
                  index <= 1 ? 'bg-purple-400' : 'bg-purple-400/40'
                }`}
              />
            ))}
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default ComingSoon;