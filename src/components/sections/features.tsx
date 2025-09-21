import {
  ActivityIcon,
  BarChart3Icon,
  BrainIcon,
  ShieldCheckIcon,
  TrendingUpIcon,
  LineChartIcon,
} from "lucide-react";

import { BentoCard, BentoGrid } from "@/components/ui/bento-grid";
import Silk from "@/components/Silk";

const features = [
  {
    Icon: BrainIcon,
    name: "AI-Powered Market Prediction Engine",
    description: "Sophisticated machine learning model that analyzes vast datasets to predict short-term stock movements, volatility, and market sentiment from news feeds.",
    href: "#prediction",
    cta: "Learn more",
    background: (
      <div className="absolute inset-0 bg-gradient-to-br from-gray-800/30 to-gray-900/40">
        <div className="absolute top-4 right-4 w-32 h-32 bg-gray-700/20 rounded-full blur-2xl"></div>
        <div className="absolute bottom-4 left-4 w-24 h-24 bg-gray-600/20 rounded-full blur-xl"></div>
      </div>
    ),
    className: "lg:row-start-1 lg:row-end-3 lg:col-start-1 lg:col-end-2",
  },
  {
    Icon: ActivityIcon,
    name: "Automated Trading Strategy Simulation",
    description: "Design and test trading rules in a risk-free environment. Backtest strategies against historical data with paper money to validate profitability.",
    href: "#simulation",
    cta: "Try simulation",
    background: (
      <div className="absolute inset-0 bg-gradient-to-br from-gray-800/30 to-gray-900/40">
        <div className="absolute top-8 right-8 w-20 h-20 bg-gray-700/20 rounded-lg rotate-12 blur-sm"></div>
        <div className="absolute bottom-8 left-8 w-16 h-16 bg-gray-600/20 rounded-lg -rotate-12 blur-sm"></div>
      </div>
    ),
    className: "lg:row-start-1 lg:row-end-2 lg:col-start-2 lg:col-end-4",
  },
  {
    Icon: LineChartIcon,
    name: "Interactive Performance Dashboard",
    description: "Advanced visualization interface showing profit/loss, returns, and risk-to-reward ratios.",
    href: "#dashboard",
    cta: "View dashboard",
    background: (
      <div className="absolute inset-0 bg-gradient-to-br from-gray-800/30 to-gray-900/40">
        <div className="absolute top-6 left-6 w-28 h-28 bg-gray-700/20 rounded-full blur-xl"></div>
        <div className="absolute bottom-6 right-6 w-20 h-20 bg-gray-600/20 rounded-full blur-lg"></div>
      </div>
    ),
    className: "lg:row-start-2 lg:row-end-3 lg:col-start-2 lg:col-end-3 dashboard-card",
  },
  {
    Icon: ShieldCheckIcon,
    name: "Risk and Portfolio Management",
    description: "Smart AI analysis of your portfolio with alerts for high-risk positions, diversification recommendations, and profit/loss optimization strategies.",
    href: "#risk-management",
    cta: "Manage risk",
    background: (
      <div className="absolute inset-0 bg-gradient-to-br from-gray-800/30 to-gray-900/40">
        <div className="absolute top-4 left-4 w-24 h-24 bg-gray-700/20 rounded-full blur-lg"></div>
        <div className="absolute bottom-4 right-4 w-32 h-32 bg-gray-600/20 rounded-full blur-2xl"></div>
      </div>
    ),
    className: "lg:row-start-2 lg:row-end-3 lg:col-start-3 lg:col-end-4",
  },
];

export function FeaturesSection() {
  return (
    <section id="features" className="relative pt-32 pb-24 px-4 overflow-hidden min-h-screen">
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
        {/* Bento Grid */}
        <BentoGrid className="lg:grid-rows-2 max-w-6xl mx-auto">
          {features.map((feature) => (
            <BentoCard key={feature.name} {...feature} />
          ))}
        </BentoGrid>
      </div>
    </section>
  );
}