"use client"

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Play, 
  Pause, 
  Settings, 
  TrendingUp, 
  TrendingDown,
  DollarSign,
  Clock,
  BarChart3,
  Target,
  AlertTriangle,
  CheckCircle,
  X,
  Plus
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

// Mock simulation data
const mockSimulationRuns = [
  {
    id: 1,
    name: "Conservative Growth",
    status: "completed",
    startDate: "2024-01-01",
    endDate: "2024-03-01",
    initialCapital: 100000,
    finalValue: 112350,
    totalReturn: 12.35,
    totalTrades: 45,
    winRate: 78.5,
    maxDrawdown: -4.2
  },
  {
    id: 2,
    name: "Aggressive Tech Focus",
    status: "running",
    startDate: "2024-02-15",
    endDate: "2024-05-15",
    initialCapital: 150000,
    finalValue: 158940,
    totalReturn: 5.96,
    totalTrades: 28,
    winRate: 71.4,
    maxDrawdown: -8.7
  }
]

const mockTrades = [
  {
    id: 1,
    timestamp: "2024-03-15 09:30:00",
    symbol: "AAPL",
    action: "BUY",
    quantity: 50,
    price: 172.50,
    value: 8625,
    reason: "AI model predicted 8% upside",
    pnl: null,
    status: "open"
  },
  {
    id: 2,
    timestamp: "2024-03-14 14:15:00",
    symbol: "TSLA",
    action: "SELL",
    quantity: 25,
    price: 245.80,
    value: 6145,
    reason: "Stop loss triggered",
    pnl: -340,
    status: "closed"
  },
  {
    id: 3,
    timestamp: "2024-03-14 10:45:00",
    symbol: "MSFT",
    action: "BUY",
    quantity: 30,
    price: 420.30,
    value: 12609,
    reason: "Earnings momentum strategy",
    pnl: null,
    status: "open"
  },
  {
    id: 4,
    timestamp: "2024-03-13 16:00:00",
    symbol: "GOOGL",
    action: "SELL",
    quantity: 15,
    price: 138.90,
    value: 2083.50,
    reason: "Target price reached",
    pnl: +240.75,
    status: "closed"
  }
]

export default function SimulationPage() {
  const [isCreatingStrategy, setIsCreatingStrategy] = useState(false)
  const [strategyName, setStrategyName] = useState("")
  const [initialCapital, setInitialCapital] = useState("100000")
  const [riskLevel, setRiskLevel] = useState("moderate")

  const handleCreateStrategy = () => {
    // Simulate strategy creation
    console.log("Creating strategy:", { strategyName, initialCapital, riskLevel })
    setIsCreatingStrategy(false)
    setStrategyName("")
    setInitialCapital("100000")
    setRiskLevel("moderate")
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="text-3xl font-bold tracking-tight">Trading Simulation</h2>
        <p className="text-muted-foreground mt-2">
          Test your AI-powered trading strategies with historical data
        </p>
      </motion.div>

      {/* Strategy Builder */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center space-x-2">
                  <Settings className="h-5 w-5" />
                  <span>Strategy Builder</span>
                </CardTitle>
                <CardDescription>
                  Create and configure your automated trading strategy
                </CardDescription>
              </div>
              <Button onClick={() => setIsCreatingStrategy(true)}>
                <Plus className="h-4 w-4 mr-2" />
                New Strategy
              </Button>
            </div>
          </CardHeader>
          
          {isCreatingStrategy && (
            <CardContent className="space-y-4 border-t">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="strategyName">Strategy Name</Label>
                  <Input
                    id="strategyName"
                    placeholder="Enter strategy name"
                    value={strategyName}
                    onChange={(e) => setStrategyName(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="initialCapital">Initial Capital ($)</Label>
                  <Input
                    id="initialCapital"
                    type="number"
                    value={initialCapital}
                    onChange={(e) => setInitialCapital(e.target.value)}
                  />
                </div>
              </div>
              
              <div className="space-y-2">
                <Label>Risk Level</Label>
                <div className="flex space-x-2">
                  {["conservative", "moderate", "aggressive"].map((level) => (
                    <Button
                      key={level}
                      variant={riskLevel === level ? "default" : "outline"}
                      size="sm"
                      onClick={() => setRiskLevel(level)}
                    >
                      {level.charAt(0).toUpperCase() + level.slice(1)}
                    </Button>
                  ))}
                </div>
              </div>
              
              <div className="flex space-x-2 pt-4">
                <Button onClick={handleCreateStrategy} disabled={!strategyName}>
                  Create Strategy
                </Button>
                <Button variant="outline" onClick={() => setIsCreatingStrategy(false)}>
                  Cancel
                </Button>
              </div>
            </CardContent>
          )}
        </Card>
      </motion.div>

      {/* Active Simulations */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <Card>
          <CardHeader>
            <CardTitle>Active Simulations</CardTitle>
            <CardDescription>Monitor your running trading strategies</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {mockSimulationRuns.map((simulation) => (
                <div key={simulation.id} className="p-4 rounded-lg border bg-accent/20">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className={`w-3 h-3 rounded-full ${
                        simulation.status === 'running' ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
                      }`} />
                      <h3 className="font-semibold">{simulation.name}</h3>
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        simulation.status === 'running' 
                          ? 'bg-green-500/20 text-green-700 dark:text-green-300' 
                          : 'bg-gray-500/20 text-gray-700 dark:text-gray-300'
                      }`}>
                        {simulation.status.toUpperCase()}
                      </span>
                    </div>
                    <div className="flex space-x-2">
                      {simulation.status === 'running' && (
                        <Button size="sm" variant="outline">
                          <Pause className="h-3 w-3" />
                        </Button>
                      )}
                      <Button size="sm" variant="outline">
                        <Settings className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 text-sm">
                    <div>
                      <div className="text-muted-foreground">Initial Capital</div>
                      <div className="font-semibold">${simulation.initialCapital.toLocaleString()}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Current Value</div>
                      <div className="font-semibold">${simulation.finalValue.toLocaleString()}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Total Return</div>
                      <div className={`font-semibold ${
                        simulation.totalReturn >= 0 ? 'text-green-500' : 'text-red-500'
                      }`}>
                        {simulation.totalReturn >= 0 ? '+' : ''}{simulation.totalReturn}%
                      </div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Win Rate</div>
                      <div className="font-semibold">{simulation.winRate}%</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Total Trades</div>
                      <div className="font-semibold">{simulation.totalTrades}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Max Drawdown</div>
                      <div className="font-semibold text-red-500">{simulation.maxDrawdown}%</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Trade Log */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>Simulated Trade Log</span>
            </CardTitle>
            <CardDescription>Recent trades executed by your strategies</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {mockTrades.map((trade) => (
                <div key={trade.id} className="p-4 rounded-lg bg-accent/30 border">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-3">
                      <div className={`p-1 rounded ${
                        trade.action === 'BUY' ? 'bg-green-500/20' : 'bg-red-500/20'
                      }`}>
                        {trade.action === 'BUY' ? (
                          <TrendingUp className="h-3 w-3 text-green-500" />
                        ) : (
                          <TrendingDown className="h-3 w-3 text-red-500" />
                        )}
                      </div>
                      <div>
                        <div className="font-semibold text-sm">{trade.symbol}</div>
                        <div className="text-xs text-muted-foreground">{trade.timestamp}</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <div className="text-right">
                        <div className="text-sm font-medium">
                          {trade.action} {trade.quantity} @ ${trade.price}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Total: ${trade.value.toLocaleString()}
                        </div>
                      </div>
                      {trade.pnl !== null && (
                        <div className={`text-sm font-semibold ${
                          trade.pnl >= 0 ? 'text-green-500' : 'text-red-500'
                        }`}>
                          {trade.pnl >= 0 ? '+' : ''}${Math.abs(trade.pnl)}
                        </div>
                      )}
                      <div className={`w-2 h-2 rounded-full ${
                        trade.status === 'open' ? 'bg-blue-500' : 'bg-gray-400'
                      }`} />
                    </div>
                  </div>
                  <div className="text-xs text-muted-foreground bg-accent/50 p-2 rounded">
                    <strong>Reason:</strong> {trade.reason}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Performance Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="grid grid-cols-1 md:grid-cols-2 gap-6"
      >
        {/* Strategy Performance */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Target className="h-5 w-5" />
              <span>Strategy Performance</span>
            </CardTitle>
            <CardDescription>Overall performance metrics</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-3 rounded-lg bg-green-500/10">
                <div className="text-lg font-bold text-green-500">76.2%</div>
                <div className="text-xs text-muted-foreground">Win Rate</div>
              </div>
              <div className="text-center p-3 rounded-lg bg-blue-500/10">
                <div className="text-lg font-bold text-blue-500">1.8</div>
                <div className="text-xs text-muted-foreground">Profit Factor</div>
              </div>
              <div className="text-center p-3 rounded-lg bg-purple-500/10">
                <div className="text-lg font-bold text-purple-500">$485</div>
                <div className="text-xs text-muted-foreground">Avg Win</div>
              </div>
              <div className="text-center p-3 rounded-lg bg-red-500/10">
                <div className="text-lg font-bold text-red-500">-$267</div>
                <div className="text-xs text-muted-foreground">Avg Loss</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Risk Metrics */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5" />
              <span>Risk Metrics</span>
            </CardTitle>
            <CardDescription>Risk assessment and drawdown analysis</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm">Max Drawdown</span>
                <span className="font-semibold text-red-500">-8.7%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Sharpe Ratio</span>
                <span className="font-semibold text-blue-500">1.42</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Volatility</span>
                <span className="font-semibold text-yellow-500">12.4%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Beta</span>
                <span className="font-semibold text-purple-500">0.85</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}

