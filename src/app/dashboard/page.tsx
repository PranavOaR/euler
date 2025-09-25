"use client"

import React from 'react'
import { motion } from 'framer-motion'
import { 
  TrendingUp, 
  DollarSign, 
  BarChart3, 
  PieChart,
  ArrowUpRight,
  ArrowDownRight,
  Activity,
  Target
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

// Mock data for demonstration
const mockKPIs = [
  {
    title: "Portfolio Value",
    value: "$847,230",
    change: "+12.5%",
    isPositive: true,
    icon: DollarSign,
    description: "Total portfolio valuation"
  },
  {
    title: "Monthly Return",
    value: "8.2%",
    change: "+2.1%",
    isPositive: true,
    icon: TrendingUp,
    description: "Return for current month"
  },
  {
    title: "Active Positions",
    value: "24",
    change: "+3",
    isPositive: true,
    icon: BarChart3,
    description: "Currently open positions"
  },
  {
    title: "Win Rate",
    value: "74.5%",
    change: "-1.2%",
    isPositive: false,
    icon: Target,
    description: "Successful trade percentage"
  }
]

const mockRecentTrades = [
  { symbol: "AAPL", action: "BUY", amount: "$15,230", profit: "+$1,245", time: "2h ago" },
  { symbol: "TSLA", action: "SELL", amount: "$8,500", profit: "+$850", time: "4h ago" },
  { symbol: "MSFT", action: "BUY", amount: "$12,100", profit: "-$120", time: "6h ago" },
  { symbol: "GOOGL", action: "SELL", amount: "$9,800", profit: "+$980", time: "1d ago" },
]

const mockAlerts = [
  { type: "success", message: "AAPL target reached - Consider taking profits", time: "5m ago" },
  { type: "warning", message: "High volatility detected in tech sector", time: "15m ago" },
  { type: "info", message: "New AI prediction model ready for TSLA", time: "1h ago" },
]

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      {/* Dashboard Overview Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="text-3xl font-bold tracking-tight">Dashboard Overview</h2>
        <p className="text-muted-foreground mt-2">
          Real-time insights into your AI-powered trading performance
        </p>
      </motion.div>

      {/* Key Performance Indicators */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        {mockKPIs.map((kpi, index) => {
          const Icon = kpi.icon
          return (
            <Card key={kpi.title} className="hover:shadow-lg transition-shadow duration-200">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">{kpi.title}</CardTitle>
                <Icon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{kpi.value}</div>
                <div className="flex items-center space-x-1 text-xs">
                  {kpi.isPositive ? (
                    <ArrowUpRight className="h-3 w-3 text-green-500" />
                  ) : (
                    <ArrowDownRight className="h-3 w-3 text-red-500" />
                  )}
                  <span className={kpi.isPositive ? "text-green-500" : "text-red-500"}>
                    {kpi.change}
                  </span>
                  <span className="text-muted-foreground">from last month</span>
                </div>
                <p className="text-xs text-muted-foreground mt-1">{kpi.description}</p>
              </CardContent>
            </Card>
          )
        })}
      </motion.div>

      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Portfolio Performance Chart Placeholder */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="lg:col-span-2"
        >
          <Card className="h-[400px]">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Activity className="h-5 w-5" />
                <span>Portfolio Performance</span>
              </CardTitle>
              <CardDescription>
                Track your portfolio's performance over time
              </CardDescription>
            </CardHeader>
            <CardContent className="flex items-center justify-center h-[300px]">
              <div className="text-center space-y-4">
                <div className="w-16 h-16 mx-auto rounded-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center">
                  <BarChart3 className="h-8 w-8 text-white" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold">Interactive Chart Coming Soon</h3>
                  <p className="text-sm text-muted-foreground">
                    Advanced portfolio analytics and performance visualization
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Recent Activity */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <Card className="h-[400px]">
            <CardHeader>
              <CardTitle>Recent Trades</CardTitle>
              <CardDescription>Latest trading activity</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {mockRecentTrades.map((trade, index) => (
                <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-accent/50">
                  <div className="flex items-center space-x-3">
                    <div className={`w-2 h-2 rounded-full ${
                      trade.action === 'BUY' ? 'bg-green-500' : 'bg-red-500'
                    }`} />
                    <div>
                      <div className="font-medium text-sm">{trade.symbol}</div>
                      <div className="text-xs text-muted-foreground">{trade.action} • {trade.time}</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium">{trade.amount}</div>
                    <div className={`text-xs ${
                      trade.profit.startsWith('+') ? 'text-green-500' : 'text-red-500'
                    }`}>
                      {trade.profit}
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Alerts & Market Insights */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        {/* Market Alerts */}
        <Card>
          <CardHeader>
            <CardTitle>Market Alerts</CardTitle>
            <CardDescription>Important market notifications and updates</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {mockAlerts.map((alert, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 rounded-lg bg-accent/30">
                <div className={`w-2 h-2 rounded-full mt-2 ${
                  alert.type === 'success' ? 'bg-green-500' :
                  alert.type === 'warning' ? 'bg-yellow-500' : 'bg-blue-500'
                }`} />
                <div className="flex-1">
                  <p className="text-sm">{alert.message}</p>
                  <p className="text-xs text-muted-foreground mt-1">{alert.time}</p>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* AI Insights */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <PieChart className="h-5 w-5" />
              <span>AI Market Insights</span>
            </CardTitle>
            <CardDescription>Powered by advanced machine learning models</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-4 rounded-lg bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-200 dark:border-blue-800">
              <h4 className="font-semibold text-sm mb-2">Today's Market Sentiment</h4>
              <div className="flex items-center justify-between">
                <span className="text-sm">Bullish</span>
                <span className="text-sm font-bold text-green-500">72%</span>
              </div>
              <div className="w-full bg-accent rounded-full h-2 mt-2">
                <div className="bg-green-500 h-2 rounded-full" style={{ width: '72%' }}></div>
              </div>
            </div>
            
            <div className="p-4 rounded-lg bg-accent/50">
              <h4 className="font-semibold text-sm mb-2">Next Prediction Update</h4>
              <p className="text-sm text-muted-foreground">
                New AI predictions will be available in 2 hours
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}