"use client"

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Search, 
  TrendingUp, 
  TrendingDown, 
  Brain, 
  Clock, 
  Target,
  DollarSign,
  BarChart3,
  AlertCircle,
  CheckCircle,
  Wifi,
  WifiOff
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { stockApi } from '@/lib/stockApi'

// Mock prediction data
const mockPredictions: Record<string, any> = {
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
  },
  GOOGL: {
    symbol: "GOOGL",
    companyName: "Alphabet Inc.",
    currentPrice: 138.90,
    predictedPrice: 152.30,
    confidence: 79,
    direction: "up",
    timeframe: "7 days",
    factors: [
      { factor: "Search revenue growth", weight: 30, impact: "positive" },
      { factor: "Cloud expansion", weight: 25, impact: "positive" },
      { factor: "AI developments", weight: 25, impact: "positive" },
      { factor: "Regulatory concerns", weight: 20, impact: "negative" }
    ],
    technicalIndicators: {
      rsi: 58.3,
      macd: "Bullish",
      sma50: 135.40,
      sma200: 132.10
    }
  },
  AMZN: {
    symbol: "AMZN",
    companyName: "Amazon.com Inc.",
    currentPrice: 155.20,
    predictedPrice: 168.50,
    confidence: 84,
    direction: "up",
    timeframe: "7 days",
    factors: [
      { factor: "AWS growth momentum", weight: 35, impact: "positive" },
      { factor: "E-commerce recovery", weight: 25, impact: "positive" },
      { factor: "Cost optimization", weight: 20, impact: "positive" },
      { factor: "Economic headwinds", weight: 20, impact: "negative" }
    ],
    technicalIndicators: {
      rsi: 62.1,
      macd: "Bullish",
      sma50: 150.80,
      sma200: 145.60
    }
  },
  NVDA: {
    symbol: "NVDA",
    companyName: "NVIDIA Corporation",
    currentPrice: 875.30,
    predictedPrice: 920.80,
    confidence: 89,
    direction: "up",
    timeframe: "7 days",
    factors: [
      { factor: "AI chip demand surge", weight: 45, impact: "positive" },
      { factor: "Data center growth", weight: 30, impact: "positive" },
      { factor: "Gaming market recovery", weight: 15, impact: "positive" },
      { factor: "Supply chain risks", weight: 10, impact: "negative" }
    ],
    technicalIndicators: {
      rsi: 71.2,
      macd: "Bullish",
      sma50: 850.40,
      sma200: 780.90
    }
  }
}

// Function to generate mock prediction for any symbol
const generateMockPrediction = (symbol: string) => {
  const basePrice = Math.random() * 500 + 50 // Random price between 50-550
  const changePercent = (Math.random() - 0.5) * 0.2 // -10% to +10%
  const predictedPrice = basePrice * (1 + changePercent)
  const direction = predictedPrice > basePrice ? "up" : "down"
  
  return {
    symbol: symbol.toUpperCase(),
    companyName: `${symbol.toUpperCase()} Corporation`,
    currentPrice: Math.round(basePrice * 100) / 100,
    predictedPrice: Math.round(predictedPrice * 100) / 100,
    confidence: Math.floor(Math.random() * 30) + 65, // 65-95%
    direction,
    timeframe: "7 days",
    factors: [
      { factor: "Market sentiment analysis", weight: 30, impact: direction === "up" ? "positive" : "negative" },
      { factor: "Technical indicators", weight: 25, impact: direction === "up" ? "positive" : "negative" },
      { factor: "Sector performance", weight: 25, impact: Math.random() > 0.5 ? "positive" : "negative" },
      { factor: "Economic conditions", weight: 20, impact: Math.random() > 0.5 ? "positive" : "negative" }
    ],
    technicalIndicators: {
      rsi: Math.round((Math.random() * 40 + 30) * 10) / 10, // 30-70
      macd: direction === "up" ? "Bullish" : "Bearish",
      sma50: Math.round((basePrice * 0.98) * 100) / 100,
      sma200: Math.round((basePrice * 0.95) * 100) / 100
    }
  }
}

const popularStocks = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX"]

export default function PredictionPage() {
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedStock, setSelectedStock] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [prediction, setPrediction] = useState<any>(null)
  const [isRealData, setIsRealData] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = async (symbol: string) => {
    if (!symbol || symbol.trim() === '') return
    
    setIsLoading(true)
    setSelectedStock(null)
    setPrediction(null)
    setError(null)
    
    try {
      // First try to get real data
      console.log(`Fetching real data for ${symbol}...`)
      const quote = await stockApi.getStockQuote(symbol.toUpperCase())
      
      if (quote) {
        // Get additional data
        const [company, technicals] = await Promise.all([
          stockApi.getCompanyOverview(symbol.toUpperCase()),
          stockApi.getTechnicalIndicators(symbol.toUpperCase())
        ])
        
        if (company && technicals) {
          // Generate prediction based on real data
          const realPrediction = stockApi.generatePrediction(quote, technicals, company)
          setPrediction(realPrediction)
          setIsRealData(true)
          console.log('Using real market data!')
        } else {
          throw new Error('Unable to fetch complete market data')
        }
      } else {
        throw new Error('Stock symbol not found')
      }
      
      setSelectedStock(symbol.toUpperCase())
    } catch (err) {
      console.log('Real API failed, falling back to mock data...')
      
      // Fallback to mock data
      const mockPrediction = mockPredictions[symbol.toUpperCase()] || generateMockPrediction(symbol.toUpperCase())
      setPrediction(mockPrediction)
      setIsRealData(false)
      setSelectedStock(symbol.toUpperCase())
    }
    
    setIsLoading(false)
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">AI Market Predictions</h2>
            <p className="text-muted-foreground mt-2">
              Get real-time AI-powered predictions for your favorite stocks
            </p>
          </div>
          
          {/* Data Source Indicator */}
          {prediction && (
            <div className={`flex items-center space-x-2 px-3 py-2 rounded-lg ${
              isRealData 
                ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300' 
                : 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300'
            }`}>
              {isRealData ? (
                <Wifi className="h-4 w-4" />
              ) : (
                <WifiOff className="h-4 w-4" />
              )}
              <span className="text-sm font-medium">
                {isRealData ? 'Live Market Data' : 'Demo Data'}
              </span>
            </div>
          )}
        </div>
      </motion.div>

      {/* Search Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Search className="h-5 w-5" />
              <span>Stock Symbol Search</span>
            </CardTitle>
            <CardDescription>
              Enter a stock symbol to get AI-powered price predictions
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex space-x-2">
              <Input
                placeholder="Enter stock symbol (e.g., AAPL, TSLA)"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value.toUpperCase())}
                onKeyPress={(e) => e.key === 'Enter' && searchTerm && handleSearch(searchTerm)}
                className="flex-1"
              />
              <Button 
                onClick={() => searchTerm && handleSearch(searchTerm)}
                disabled={!searchTerm || isLoading}
                className="px-6"
              >
                {isLoading ? (
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                ) : (
                  "Predict"
                )}
              </Button>
            </div>

            {/* Popular Stocks */}
            <div>
              <p className="text-sm text-muted-foreground mb-2">Popular stocks:</p>
              <div className="flex flex-wrap gap-2">
                {popularStocks.map((stock) => (
                  <Button
                    key={stock}
                    variant="outline"
                    size="sm"
                    onClick={() => handleSearch(stock)}
                    disabled={isLoading}
                  >
                    {stock}
                  </Button>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Loading State */}
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center justify-center py-12"
        >
          <div className="text-center space-y-4">
            <div className="w-16 h-16 mx-auto rounded-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center">
              <Brain className="h-8 w-8 text-white animate-pulse" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">AI Model Processing...</h3>
              <p className="text-sm text-muted-foreground">
                Analyzing market data and generating predictions
              </p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Prediction Results */}
      {prediction && !isLoading && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="space-y-6"
        >
          {/* Main Prediction Card */}
          <Card className="border-2 border-primary/20">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-full ${
                    prediction.direction === 'up' ? 'bg-green-500/20' : 'bg-red-500/20'
                  }`}>
                    {prediction.direction === 'up' ? (
                      <TrendingUp className="h-5 w-5 text-green-500" />
                    ) : (
                      <TrendingDown className="h-5 w-5 text-red-500" />
                    )}
                  </div>
                  <div>
                    <h3 className="text-xl font-bold">{prediction.symbol}</h3>
                    <p className="text-sm text-muted-foreground">{prediction.companyName}</p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold">${prediction.currentPrice}</div>
                  <div className="text-sm text-muted-foreground">Current Price</div>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Prediction Details */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 rounded-lg bg-accent/50">
                  <div className={`text-2xl font-bold ${
                    prediction.direction === 'up' ? 'text-green-500' : 'text-red-500'
                  }`}>
                    ${prediction.predictedPrice}
                  </div>
                  <div className="text-sm text-muted-foreground">Predicted Price</div>
                  <div className={`text-xs ${
                    prediction.direction === 'up' ? 'text-green-500' : 'text-red-500'
                  }`}>
                    {prediction.direction === 'up' ? '+' : ''}
                    {((prediction.predictedPrice - prediction.currentPrice) / prediction.currentPrice * 100).toFixed(1)}%
                  </div>
                </div>
                
                <div className="text-center p-4 rounded-lg bg-accent/50">
                  <div className="text-2xl font-bold text-blue-500">{prediction.confidence}%</div>
                  <div className="text-sm text-muted-foreground">Confidence</div>
                  <div className="text-xs text-muted-foreground">Model Accuracy</div>
                </div>
                
                <div className="text-center p-4 rounded-lg bg-accent/50">
                  <div className="text-2xl font-bold text-purple-500">{prediction.timeframe}</div>
                  <div className="text-sm text-muted-foreground">Time Frame</div>
                  <div className="text-xs text-muted-foreground">Prediction Period</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Analysis Details */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Key Factors */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Target className="h-5 w-5" />
                  <span>Key Prediction Factors</span>
                </CardTitle>
                <CardDescription>
                  Factors influencing the AI model's prediction
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {prediction.factors.map((factor, index) => (
                  <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-accent/30">
                    <div className="flex items-center space-x-3">
                      {factor.impact === 'positive' ? (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      ) : (
                        <AlertCircle className="h-4 w-4 text-red-500" />
                      )}
                      <span className="text-sm">{factor.factor}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="text-xs text-muted-foreground">{factor.weight}%</div>
                      <div className={`w-16 h-2 rounded-full bg-accent ${
                        factor.impact === 'positive' ? 'bg-green-500/20' : 'bg-red-500/20'
                      }`}>
                        <div 
                          className={`h-2 rounded-full ${
                            factor.impact === 'positive' ? 'bg-green-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${factor.weight}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Technical Indicators */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <BarChart3 className="h-5 w-5" />
                  <span>Technical Indicators</span>
                </CardTitle>
                <CardDescription>
                  Current technical analysis signals
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 rounded-lg bg-accent/30">
                    <div className="text-sm text-muted-foreground">RSI</div>
                    <div className="text-lg font-semibold">{prediction.technicalIndicators.rsi}</div>
                  </div>
                  <div className="p-3 rounded-lg bg-accent/30">
                    <div className="text-sm text-muted-foreground">MACD</div>
                    <div className={`text-lg font-semibold ${
                      prediction.technicalIndicators.macd === 'Bullish' ? 'text-green-500' : 'text-red-500'
                    }`}>
                      {prediction.technicalIndicators.macd}
                    </div>
                  </div>
                  <div className="p-3 rounded-lg bg-accent/30">
                    <div className="text-sm text-muted-foreground">SMA 50</div>
                    <div className="text-lg font-semibold">${prediction.technicalIndicators.sma50}</div>
                  </div>
                  <div className="p-3 rounded-lg bg-accent/30">
                    <div className="text-sm text-muted-foreground">SMA 200</div>
                    <div className="text-lg font-semibold">${prediction.technicalIndicators.sma200}</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Enhanced Investment Disclaimer */}
          <Card className="border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-950/20">
            <CardContent className="pt-6">
              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <AlertCircle className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                  <h3 className="text-lg font-bold text-blue-900 dark:text-blue-100">
                    Important Investment Disclosure
                  </h3>
                </div>
                
                <div className="space-y-3 text-sm text-blue-800 dark:text-blue-200">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <h4 className="font-semibold text-blue-900 dark:text-blue-100">AI Predictions</h4>
                      <p>
                        Our AI models analyze market data, technical indicators, and historical patterns. 
                        However, predictions are estimates and not guarantees of future performance.
                      </p>
                    </div>
                    
                    <div className="space-y-2">
                      <h4 className="font-semibold text-blue-900 dark:text-blue-100">Risk Warning</h4>
                      <p>
                        All investments carry risk. Stock prices can fluctuate significantly and you may 
                        lose some or all of your investment. Past performance does not predict future results.
                      </p>
                    </div>
                    
                    <div className="space-y-2">
                      <h4 className="font-semibold text-blue-900 dark:text-blue-100">Not Financial Advice</h4>
                      <p>
                        This platform provides educational and analytical tools only. We do not provide 
                        investment advice, recommendations, or endorse any specific securities.
                      </p>
                    </div>
                    
                    <div className="space-y-2">
                      <h4 className="font-semibold text-blue-900 dark:text-blue-100">Professional Guidance</h4>
                      <p>
                        Consult with qualified financial advisors, conduct your own research, and consider 
                        your risk tolerance before making investment decisions.
                      </p>
                    </div>
                  </div>
                  
                  <div className="mt-4 p-3 bg-blue-100 dark:bg-blue-900/30 rounded-lg border border-blue-200 dark:border-blue-700">
                    <p className="text-xs text-blue-700 dark:text-blue-300">
                      <strong>Data Sources:</strong> Market data is sourced from {isRealData ? 'live financial APIs' : 'simulated market data'} and may have delays. 
                      Technical indicators are calculated using standard methodologies but should be used alongside other analysis techniques.
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  )
}
