// Mock data and API utilities for the Euler dashboard

export interface PortfolioKPI {
  title: string
  value: string
  change: string
  isPositive: boolean
  icon: string
  description: string
}

export interface Trade {
  id: number
  symbol: string
  action: 'BUY' | 'SELL'
  amount: string
  profit: string
  time: string
  status: 'open' | 'closed'
  quantity?: number
  price?: number
}

export interface MarketAlert {
  id: number
  type: 'success' | 'warning' | 'info' | 'error'
  message: string
  time: string
  priority: 'high' | 'medium' | 'low'
}

export interface PredictionData {
  symbol: string
  companyName: string
  currentPrice: number
  predictedPrice: number
  confidence: number
  direction: 'up' | 'down'
  timeframe: string
  factors: PredictionFactor[]
  technicalIndicators: TechnicalIndicators
}

export interface PredictionFactor {
  factor: string
  weight: number
  impact: 'positive' | 'negative'
}

export interface TechnicalIndicators {
  rsi: number
  macd: string
  sma50: number
  sma200: number
}

export interface SimulationRun {
  id: number
  name: string
  status: 'running' | 'completed' | 'paused'
  startDate: string
  endDate: string
  initialCapital: number
  finalValue: number
  totalReturn: number
  totalTrades: number
  winRate: number
  maxDrawdown: number
}

// Mock API functions
export const mockApi = {
  // Dashboard data
  async getPortfolioKPIs(): Promise<PortfolioKPI[]> {
    await new Promise(resolve => setTimeout(resolve, 500)) // Simulate API delay
    
    return [
      {
        title: "Portfolio Value",
        value: "$847,230",
        change: "+12.5%",
        isPositive: true,
        icon: "DollarSign",
        description: "Total portfolio valuation"
      },
      {
        title: "Monthly Return",
        value: "8.2%",
        change: "+2.1%",
        isPositive: true,
        icon: "TrendingUp",
        description: "Return for current month"
      },
      {
        title: "Active Positions",
        value: "24",
        change: "+3",
        isPositive: true,
        icon: "BarChart3",
        description: "Currently open positions"
      },
      {
        title: "Win Rate",
        value: "74.5%",
        change: "-1.2%",
        isPositive: false,
        icon: "Target",
        description: "Successful trade percentage"
      }
    ]
  },

  async getRecentTrades(): Promise<Trade[]> {
    await new Promise(resolve => setTimeout(resolve, 300))
    
    return [
      { id: 1, symbol: "AAPL", action: "BUY", amount: "$15,230", profit: "+$1,245", time: "2h ago", status: "open" },
      { id: 2, symbol: "TSLA", action: "SELL", amount: "$8,500", profit: "+$850", time: "4h ago", status: "closed" },
      { id: 3, symbol: "MSFT", action: "BUY", amount: "$12,100", profit: "-$120", time: "6h ago", status: "open" },
      { id: 4, symbol: "GOOGL", action: "SELL", amount: "$9,800", profit: "+$980", time: "1d ago", status: "closed" }
    ]
  },

  async getMarketAlerts(): Promise<MarketAlert[]> {
    await new Promise(resolve => setTimeout(resolve, 200))
    
    return [
      { 
        id: 1, 
        type: "success", 
        message: "AAPL target reached - Consider taking profits", 
        time: "5m ago",
        priority: "high"
      },
      { 
        id: 2, 
        type: "warning", 
        message: "High volatility detected in tech sector", 
        time: "15m ago",
        priority: "medium"
      },
      { 
        id: 3, 
        type: "info", 
        message: "New AI prediction model ready for TSLA", 
        time: "1h ago",
        priority: "low"
      }
    ]
  },

  // Prediction data
  async getStockPrediction(symbol: string): Promise<PredictionData | null> {
    await new Promise(resolve => setTimeout(resolve, 1500)) // Simulate AI processing time
    
    const predictions: Record<string, PredictionData> = {
      AAPL: {
        symbol: "AAPL",
        companyName: "Apple Inc.",
        currentPrice: 178.45,
        predictedPrice: 192.30,
        confidence: 87,
        direction: "up",
        timeframe: "7 days",
        factors: [
          { factor: "Strong quarterly earnings", weight: 32, impact: "positive" },
          { factor: "AI product announcements", weight: 28, impact: "positive" },
          { factor: "Market volatility", weight: 15, impact: "negative" },
          { factor: "Tech sector momentum", weight: 25, impact: "positive" }
        ],
        technicalIndicators: {
          rsi: 64.2,
          macd: "Bullish",
          sma50: 175.20,
          sma200: 168.90
        }
      },
      TSLA: {
        symbol: "TSLA",
        companyName: "Tesla Inc.",
        currentPrice: 242.84,
        predictedPrice: 218.50,
        confidence: 73,
        direction: "down",
        timeframe: "7 days",
        factors: [
          { factor: "EV market competition", weight: 35, impact: "negative" },
          { factor: "Production targets", weight: 20, impact: "positive" },
          { factor: "Regulatory changes", weight: 25, impact: "negative" },
          { factor: "Energy sector growth", weight: 20, impact: "positive" }
        ],
        technicalIndicators: {
          rsi: 42.8,
          macd: "Bearish",
          sma50: 248.60,
          sma200: 251.20
        }
      },
      MSFT: {
        symbol: "MSFT",
        companyName: "Microsoft Corporation",
        currentPrice: 420.30,
        predictedPrice: 445.80,
        confidence: 91,
        direction: "up",
        timeframe: "7 days",
        factors: [
          { factor: "Cloud growth acceleration", weight: 40, impact: "positive" },
          { factor: "AI integration success", weight: 35, impact: "positive" },
          { factor: "Enterprise demand", weight: 15, impact: "positive" },
          { factor: "Competition pressure", weight: 10, impact: "negative" }
        ],
        technicalIndicators: {
          rsi: 68.5,
          macd: "Bullish",
          sma50: 415.75,
          sma200: 398.20
        }
      }
    }
    
    return predictions[symbol.toUpperCase()] || null
  },

  // Simulation data
  async getSimulationRuns(): Promise<SimulationRun[]> {
    await new Promise(resolve => setTimeout(resolve, 400))
    
    return [
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
      },
      {
        id: 3,
        name: "Momentum Strategy",
        status: "paused",
        startDate: "2024-03-01",
        endDate: "2024-06-01",
        initialCapital: 75000,
        finalValue: 73850,
        totalReturn: -1.53,
        totalTrades: 12,
        winRate: 58.3,
        maxDrawdown: -12.1
      }
    ]
  },

  async getSimulatedTrades(): Promise<Trade[]> {
    await new Promise(resolve => setTimeout(resolve, 300))
    
    return [
      {
        id: 1,
        symbol: "AAPL",
        action: "BUY",
        quantity: 50,
        price: 172.50,
        amount: "$8,625",
        profit: "",
        time: "2h ago",
        status: "open"
      },
      {
        id: 2,
        symbol: "TSLA",
        action: "SELL",
        quantity: 25,
        price: 245.80,
        amount: "$6,145",
        profit: "-$340",
        time: "4h ago",
        status: "closed"
      },
      {
        id: 3,
        symbol: "MSFT",
        action: "BUY",
        quantity: 30,
        price: 420.30,
        amount: "$12,609",
        profit: "",
        time: "6h ago",
        status: "open"
      },
      {
        id: 4,
        symbol: "GOOGL",
        action: "SELL",
        quantity: 15,
        price: 138.90,
        amount: "$2,083.50",
        profit: "+$240.75",
        time: "1d ago",
        status: "closed"
      }
    ]
  },

  // Market data
  async getMarketSentiment(): Promise<{ sentiment: string, confidence: number, trend: 'bullish' | 'bearish' | 'neutral' }> {
    await new Promise(resolve => setTimeout(resolve, 600))
    
    return {
      sentiment: "Bullish",
      confidence: 72,
      trend: "bullish"
    }
  },

  async getPortfolioPerformance(): Promise<any> {
    await new Promise(resolve => setTimeout(resolve, 800))
    
    // Mock time series data for charts
    const dates = []
    const values = []
    const startDate = new Date('2024-01-01')
    let currentValue = 100000
    
    for (let i = 0; i < 90; i++) {
      const date = new Date(startDate)
      date.setDate(date.getDate() + i)
      dates.push(date.toISOString().split('T')[0])
      
      // Simulate portfolio growth with some volatility
      const dailyReturn = (Math.random() - 0.48) * 0.02 // Slight positive bias
      currentValue *= (1 + dailyReturn)
      values.push(Math.round(currentValue))
    }
    
    return {
      dates,
      values,
      totalReturn: ((currentValue - 100000) / 100000 * 100).toFixed(2),
      benchmark: "S&P 500",
      benchmarkReturn: "8.4%"
    }
  }
}

// Utility functions
export const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
  }).format(amount)
}

export const formatPercentage = (value: number, decimals = 1): string => {
  return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`
}

export const getRandomStockSymbols = (): string[] => {
  return ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX", "UBER", "SPOT"]
}

export const generateMockTradeId = (): string => {
  return `TXN${Date.now()}${Math.random().toString(36).substr(2, 4).toUpperCase()}`
}

