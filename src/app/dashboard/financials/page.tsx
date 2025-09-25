"use client"

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Calculator,
  TrendingUp,
  BarChart3,
  Brain,
  AlertCircle,
  CheckCircle,
  Info,
  Download,
  RefreshCw
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { financialMLModel, type FinancialInputs, type ModelPrediction } from '@/lib/mlModel'
import MLPredictionChart from '@/components/MLPredictionChart'

// Sample company data for quick testing
const sampleCompanies = [
  {
    name: "TechCorp Inc",
    data: {
      revenue: 1000000,
      totalAssets: 2500000,
      totalLiabilities: 1500000,
      currentAssets: 800000,
      currentLiabilities: 400000,
      grossProfit: 600000
    }
  },
  {
    name: "Manufacturing Ltd",
    data: {
      revenue: 2500000,
      totalAssets: 5000000,
      totalLiabilities: 3000000,
      currentAssets: 1500000,
      currentLiabilities: 800000,
      grossProfit: 1000000
    }
  },
  {
    name: "Retail Chain",
    data: {
      revenue: 5000000,
      totalAssets: 3500000,
      totalLiabilities: 2000000,
      currentAssets: 2000000,
      currentLiabilities: 1200000,
      grossProfit: 1500000
    }
  },
  {
    name: "Distressed Corp",
    data: {
      revenue: 800000,
      totalAssets: 1200000,
      totalLiabilities: 1000000,
      currentAssets: 200000,
      currentLiabilities: 300000,
      grossProfit: 80000
    }
  }
]

export default function FinancialsPage() {
  const [inputs, setInputs] = useState<FinancialInputs>({
    revenue: 0,
    totalAssets: 0,
    totalLiabilities: 0,
    currentAssets: 0,
    currentLiabilities: 0,
    grossProfit: 0
  })
  
  const [prediction, setPrediction] = useState<ModelPrediction | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [validationErrors, setValidationErrors] = useState<string[]>([])
  const [modelInfo, setModelInfo] = useState<any>(null)
  const [modelError, setModelError] = useState<string | null>(null)

  useEffect(() => {
    // Get model information when component loads and check status periodically
    const updateModelInfo = () => {
      const info = financialMLModel.getModelInfo()
      setModelInfo(info)
      
      // If model is still loading, check again in 1 second
      if (info.isLoading && !info.isLoaded) {
        setTimeout(updateModelInfo, 1000)
      }
    }
    
    updateModelInfo()
  }, [])

  const handleForceModelInit = async () => {
    try {
      setModelError(null)
      setModelInfo(prev => ({ ...prev, isLoading: true }))
      await financialMLModel.forceInitialize()
      const info = financialMLModel.getModelInfo()
      setModelInfo(info)
    } catch (error) {
      setModelError(error instanceof Error ? error.message : 'Failed to initialize model')
      console.error('Model initialization error:', error)
    }
  }

  const handleInputChange = (field: keyof FinancialInputs, value: string) => {
    const numericValue = parseFloat(value) || 0
    setInputs(prev => ({
      ...prev,
      [field]: numericValue
    }))
    
    // Clear previous errors when user starts typing
    if (validationErrors.length > 0) {
      setValidationErrors([])
    }
  }

  const loadSampleData = (companyData: FinancialInputs) => {
    setInputs(companyData)
    setValidationErrors([])
    setPrediction(null)
  }

  const handlePredict = async () => {
    setIsLoading(true)
    setPrediction(null)
    setValidationErrors([])
    setModelError(null)

    try {
      // Check if model is ready
      if (!modelInfo?.isLoaded) {
        throw new Error('ML model is not loaded. Please wait or try refreshing the page.')
      }

      // Validate inputs
      const validation = financialMLModel.validateInputs(inputs)
      if (!validation.isValid) {
        setValidationErrors(validation.errors)
        setIsLoading(false)
        return
      }

      console.log('🔄 Generating ML prediction with inputs:', inputs)

      // Generate prediction
      const result = await financialMLModel.predict(inputs)
      console.log('✅ Prediction generated:', result)
      setPrediction(result)
      
    } catch (error) {
      console.error('❌ Prediction error:', error)
      const errorMessage = error instanceof Error ? error.message : 'Prediction failed'
      
      if (errorMessage.includes('model') || errorMessage.includes('load')) {
        setModelError(errorMessage)
      } else {
        setValidationErrors([errorMessage])
      }
    } finally {
      setIsLoading(false)
    }
  }

  const clearForm = () => {
    setInputs({
      revenue: 0,
      totalAssets: 0,
      totalLiabilities: 0,
      currentAssets: 0,
      currentLiabilities: 0,
      grossProfit: 0
    })
    setPrediction(null)
    setValidationErrors([])
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount)
  }

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`
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
            <h2 className="text-3xl font-bold tracking-tight">Financial ML Predictions</h2>
            <p className="text-muted-foreground mt-2">
              Enter your company's financial parameters to get AI-powered net income predictions
            </p>
          </div>
          
          {/* Model Status */}
          {modelInfo && (
            <div className={`flex items-center space-x-2 px-3 py-2 rounded-lg ${
              modelInfo.isLoaded 
                ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300' 
                : modelInfo.isLoading
                ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300'
                : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
            }`}>
              <Brain className={`h-4 w-4 ${modelInfo.isLoading ? 'animate-pulse' : ''}`} />
              <span className="text-sm font-medium">
                {modelInfo.isLoaded 
                  ? 'ML Model Ready' 
                  : modelInfo.isLoading 
                  ? 'Loading Model...' 
                  : 'Model Failed'}
              </span>
              {!modelInfo.isLoaded && !modelInfo.isLoading && (
                <Button 
                  size="sm" 
                  variant="outline" 
                  onClick={handleForceModelInit}
                  className="ml-2 h-6 text-xs"
                >
                  Retry
                </Button>
              )}
            </div>
          )}
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input Form */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="lg:col-span-2 space-y-6"
        >
          {/* Sample Data Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Calculator className="h-5 w-5" />
                <span>Quick Start - Sample Companies</span>
              </CardTitle>
              <CardDescription>
                Load sample financial data to test the ML model quickly
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                {sampleCompanies.map((company, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    onClick={() => loadSampleData(company.data)}
                    className={`h-auto flex flex-col items-start p-4 ${
                      company.name === "Distressed Corp" ? "border-red-200 hover:border-red-300" : ""
                    }`}
                  >
                    <span className="font-semibold">{company.name}</span>
                    <span className="text-xs text-muted-foreground mt-1">
                      Revenue: {formatCurrency(company.data.revenue)}
                    </span>
                    {company.name === "Distressed Corp" && (
                      <span className="text-xs text-red-500 mt-1">High Risk</span>
                    )}
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Financial Input Form */}
          <Card>
            <CardHeader>
              <CardTitle>Financial Parameters</CardTitle>
              <CardDescription>
                Enter your company's financial data for ML prediction
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="revenue">Revenue ($)</Label>
                  <Input
                    id="revenue"
                    type="number"
                    placeholder="1,000,000"
                    value={inputs.revenue || ''}
                    onChange={(e) => handleInputChange('revenue', e.target.value)}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="grossProfit">Gross Profit ($)</Label>
                  <Input
                    id="grossProfit"
                    type="number"
                    placeholder="600,000"
                    value={inputs.grossProfit || ''}
                    onChange={(e) => handleInputChange('grossProfit', e.target.value)}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="totalAssets">Total Assets ($)</Label>
                  <Input
                    id="totalAssets"
                    type="number"
                    placeholder="2,500,000"
                    value={inputs.totalAssets || ''}
                    onChange={(e) => handleInputChange('totalAssets', e.target.value)}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="totalLiabilities">Total Liabilities ($)</Label>
                  <Input
                    id="totalLiabilities"
                    type="number"
                    placeholder="1,500,000"
                    value={inputs.totalLiabilities || ''}
                    onChange={(e) => handleInputChange('totalLiabilities', e.target.value)}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="currentAssets">Current Assets ($)</Label>
                  <Input
                    id="currentAssets"
                    type="number"
                    placeholder="800,000"
                    value={inputs.currentAssets || ''}
                    onChange={(e) => handleInputChange('currentAssets', e.target.value)}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="currentLiabilities">Current Liabilities ($)</Label>
                  <Input
                    id="currentLiabilities"
                    type="number"
                    placeholder="400,000"
                    value={inputs.currentLiabilities || ''}
                    onChange={(e) => handleInputChange('currentLiabilities', e.target.value)}
                  />
                </div>
              </div>

              {/* Model Error */}
              {modelError && (
                <div className="p-4 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-lg">
                  <div className="flex items-start space-x-2">
                    <AlertCircle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="text-sm font-semibold text-red-800 dark:text-red-200">
                        Model Error:
                      </h4>
                      <p className="mt-1 text-sm text-red-700 dark:text-red-300">
                        {modelError}
                      </p>
                      <Button 
                        size="sm" 
                        variant="outline" 
                        onClick={handleForceModelInit}
                        className="mt-2"
                      >
                        Try Again
                      </Button>
                    </div>
                  </div>
                </div>
              )}

              {/* Validation Errors */}
              {validationErrors.length > 0 && (
                <div className="p-4 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-lg">
                  <div className="flex items-start space-x-2">
                    <AlertCircle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="text-sm font-semibold text-red-800 dark:text-red-200">
                        Please fix the following errors:
                      </h4>
                      <ul className="mt-1 text-sm text-red-700 dark:text-red-300 list-disc list-inside">
                        {validationErrors.map((error, index) => (
                          <li key={index}>{error}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex space-x-3 pt-4">
                <Button 
                  onClick={handlePredict}
                  disabled={isLoading || !modelInfo?.isLoaded}
                  className="flex-1"
                >
                  {isLoading ? (
                    <>
                      <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      Processing...
                    </>
                  ) : !modelInfo?.isLoaded ? (
                    <>
                      <AlertCircle className="h-4 w-4 mr-2" />
                      {modelInfo?.isLoading ? 'Loading Model...' : 'Model Not Ready'}
                    </>
                  ) : (
                    <>
                      <Brain className="h-4 w-4 mr-2" />
                      Generate ML Prediction
                    </>
                  )}
                </Button>
                
                <Button variant="outline" onClick={clearForm}>
                  Clear Form
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Model Information Sidebar */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="space-y-6"
        >
          {/* Model Info Card */}
          {modelInfo && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Info className="h-5 w-5" />
                  <span>Model Information</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="text-sm">
                  <div className="flex justify-between mb-1">
                    <span className="text-muted-foreground">Type:</span>
                    <span className="font-medium">{modelInfo.modelType}</span>
                  </div>
                  <div className="flex justify-between mb-1">
                    <span className="text-muted-foreground">Version:</span>
                    <span className="font-medium">{modelInfo.version}</span>
                  </div>
                  <div className="flex justify-between mb-1">
                    <span className="text-muted-foreground">Training Samples:</span>
                    <span className="font-medium">{modelInfo.trainingMetrics?.samples}</span>
                  </div>
                  <div className="flex justify-between mb-1">
                    <span className="text-muted-foreground">R² Score:</span>
                    <span className="font-medium text-green-600">{modelInfo.trainingMetrics?.r2Score}</span>
                  </div>
                </div>

                <div className="pt-3 border-t">
                  <h4 className="text-sm font-semibold mb-2">Input Features:</h4>
                  <ul className="text-xs space-y-1">
                    {modelInfo.inputFeatures?.map((feature: string, index: number) => (
                      <li key={index} className="flex items-center space-x-2">
                        <div className="w-1 h-1 rounded-full bg-blue-500"></div>
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="pt-3 border-t">
                  <h4 className="text-sm font-semibold mb-2">Target Output:</h4>
                  <div className="flex items-center space-x-2">
                    <TrendingUp className="h-3 w-3 text-green-500" />
                    <span className="text-sm">{modelInfo.outputTarget}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* How It Works */}
          <Card>
            <CardHeader>
              <CardTitle>How It Works</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center flex-shrink-0">
                  <span className="text-xs font-bold text-blue-600 dark:text-blue-400">1</span>
                </div>
                <div>
                  <h4 className="font-semibold">Feature Engineering</h4>
                  <p className="text-muted-foreground">Calculate financial ratios from your inputs</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center flex-shrink-0">
                  <span className="text-xs font-bold text-blue-600 dark:text-blue-400">2</span>
                </div>
                <div>
                  <h4 className="font-semibold">Normalization</h4>
                  <p className="text-muted-foreground">Scale features using training statistics</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center flex-shrink-0">
                  <span className="text-xs font-bold text-blue-600 dark:text-blue-400">3</span>
                </div>
                <div>
                  <h4 className="font-semibold">Neural Network</h4>
                  <p className="text-muted-foreground">Process through trained model layers</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center flex-shrink-0">
                  <span className="text-xs font-bold text-blue-600 dark:text-blue-400">4</span>
                </div>
                <div>
                  <h4 className="font-semibold">Prediction</h4>
                  <p className="text-muted-foreground">Generate net income forecast</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Prediction Results */}
      {prediction && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="space-y-6"
        >
          {/* Main Prediction Result */}
          <Card className="border-2 border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-950/20">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="p-3 rounded-full bg-green-500/20">
                    <TrendingUp className="h-6 w-6 text-green-600 dark:text-green-400" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold text-green-800 dark:text-green-200">
                      ML Prediction Result
                    </h3>
                    <p className="text-green-600 dark:text-green-400">
                      Based on Neural Network Analysis
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-3xl font-bold text-green-800 dark:text-green-200">
                    {formatCurrency(prediction.predictedNetIncome)}
                  </div>
                  <div className="text-sm text-green-600 dark:text-green-400">
                    Predicted Net Income
                  </div>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="text-center p-4 rounded-lg bg-white dark:bg-green-900/20">
                  <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                    {prediction.confidence}%
                  </div>
                  <div className="text-sm text-green-700 dark:text-green-300">
                    Model Confidence
                  </div>
                </div>
                
                <div className={`text-center p-4 rounded-lg bg-white dark:bg-green-900/20`}>
                  <div className={`text-2xl font-bold ${
                    prediction.riskClassification.riskLevel === 'LOW' ? 'text-green-600' :
                    prediction.riskClassification.riskLevel === 'MEDIUM' ? 'text-yellow-600' :
                    prediction.riskClassification.riskLevel === 'HIGH' ? 'text-orange-600' : 'text-red-600'
                  }`}>
                    {prediction.riskClassification.riskLevel}
                  </div>
                  <div className="text-sm text-green-700 dark:text-green-300">
                    Risk Level
                  </div>
                </div>
                
                <div className="text-center p-4 rounded-lg bg-white dark:bg-green-900/20">
                  <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                    {prediction.riskClassification.insolvencyRisk}%
                  </div>
                  <div className="text-sm text-green-700 dark:text-green-300">
                    Insolvency Risk
                  </div>
                </div>
                
                <div className="text-center p-4 rounded-lg bg-white dark:bg-green-900/20">
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {prediction.modelMetrics.r2Score.toFixed(3)}
                  </div>
                  <div className="text-sm text-green-700 dark:text-green-300">
                    R² Score
                  </div>
                </div>
              </div>

              {/* Uncertainty Range */}
              <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800">
                <h4 className="text-sm font-semibold text-blue-800 dark:text-blue-200 mb-2">
                  Prediction Uncertainty Range (95% Confidence)
                </h4>
                <div className="flex justify-between items-center text-sm">
                  <span className="text-blue-700 dark:text-blue-300">
                    Lower Bound: {formatCurrency(prediction.uncertaintyRange.lowerBound)}
                  </span>
                  <span className="text-blue-700 dark:text-blue-300">
                    Upper Bound: {formatCurrency(prediction.uncertaintyRange.upperBound)}
                  </span>
                  <span className="text-blue-700 dark:text-blue-300">
                    Std Dev: {formatCurrency(prediction.uncertaintyRange.standardDeviation)}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Interactive Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <BarChart3 className="h-5 w-5" />
                <span>Financial Ratios Analysis</span>
              </CardTitle>
              <CardDescription>
                Interactive visualization of your financial ratios vs. target benchmarks
              </CardDescription>
            </CardHeader>
            <CardContent>
              <MLPredictionChart prediction={prediction} inputs={inputs} />
            </CardContent>
          </Card>

          {/* Feature Importance */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Brain className="h-5 w-5" />
                <span>ML Feature Importance</span>
              </CardTitle>
              <CardDescription>
                How much each financial ratio contributed to the prediction
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {prediction.featureImportance.map((item, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">{item.feature}</span>
                      <span className="text-sm text-muted-foreground">
                        {formatPercentage(item.importance)}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-gray-600 dark:bg-gray-400 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${item.importance * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  )
}
