"use client"

import React, { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { type ModelPrediction, type FinancialInputs } from '@/lib/mlModel'

interface MLPredictionChartProps {
  prediction: ModelPrediction
  inputs: FinancialInputs
}

export default function MLPredictionChart({ prediction, inputs }: MLPredictionChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * window.devicePixelRatio
    canvas.height = rect.height * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    // Clear canvas
    ctx.clearRect(0, 0, rect.width, rect.height)

    // Draw the prediction visualization
    drawPredictionChart(ctx, rect.width, rect.height, prediction, inputs)
  }, [prediction, inputs])

  const drawPredictionChart = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    prediction: ModelPrediction,
    inputs: FinancialInputs
  ) => {
    const padding = 60
    const chartWidth = width - 2 * padding
    const chartHeight = height - 2 * padding

    // Calculate financial ratios for visualization
    const shareholdersEquity = inputs.totalAssets - inputs.totalLiabilities
    const ratios = {
      netProfitMargin: inputs.grossProfit / inputs.revenue,
      debtToEquityRatio: inputs.totalLiabilities / shareholdersEquity,
      currentRatio: inputs.currentAssets / inputs.currentLiabilities,
      returnOnAssets: inputs.grossProfit / inputs.totalAssets,
      returnOnEquity: inputs.grossProfit / shareholdersEquity
    }

    // Create data points for the chart
    const dataPoints = [
      { label: 'Net Profit Margin', value: ratios.netProfitMargin, target: 0.15, importance: 0.35 },
      { label: 'Current Ratio', value: ratios.currentRatio, target: 2.0, importance: 0.05 },
      { label: 'Debt/Equity', value: ratios.debtToEquityRatio, target: 1.0, importance: 0.15 },
      { label: 'ROA', value: ratios.returnOnAssets, target: 0.08, importance: 0.20 },
      { label: 'ROE', value: ratios.returnOnEquity, target: 0.15, importance: 0.25 }
    ]

    // Draw background
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--background') || '#ffffff'
    ctx.fillRect(0, 0, width, height)

    // Draw grid lines
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--border') || '#e5e7eb'
    ctx.lineWidth = 1
    for (let i = 0; i <= 5; i++) {
      const y = padding + (i * chartHeight) / 5
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(width - padding, y)
      ctx.stroke()
    }

    // Draw bars for each ratio
    const barWidth = chartWidth / dataPoints.length - 20
    dataPoints.forEach((point, index) => {
      const x = padding + index * (chartWidth / dataPoints.length) + 10
      const normalizedValue = Math.min(point.value / (point.target * 2), 1)
      const barHeight = normalizedValue * chartHeight

      // Draw target line
      const targetY = padding + chartHeight - (0.5 * chartHeight)
      ctx.strokeStyle = '#10b981'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      ctx.moveTo(x, targetY)
      ctx.lineTo(x + barWidth, targetY)
      ctx.stroke()
      ctx.setLineDash([])

      // Draw actual value bar
      const barY = padding + chartHeight - barHeight
      const gradient = ctx.createLinearGradient(0, barY, 0, barY + barHeight)
      
      if (point.value > point.target) {
        gradient.addColorStop(0, '#22c55e')
        gradient.addColorStop(1, '#16a34a')
      } else {
        gradient.addColorStop(0, '#f59e0b')
        gradient.addColorStop(1, '#d97706')
      }

      ctx.fillStyle = gradient
      ctx.fillRect(x, barY, barWidth, barHeight)

      // Draw importance indicator
      const importanceHeight = point.importance * 20
      ctx.fillStyle = '#6366f1'
      ctx.fillRect(x + barWidth + 5, padding + chartHeight - importanceHeight, 8, importanceHeight)

      // Draw labels
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--foreground') || '#000000'
      ctx.font = '12px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(point.label, x + barWidth / 2, height - 20)
      
      // Draw values
      ctx.font = '10px sans-serif'
      ctx.fillText(point.value.toFixed(3), x + barWidth / 2, height - 5)
    })

    // Draw prediction result
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--foreground') || '#000000'
    ctx.font = 'bold 16px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(`Predicted Net Income: $${prediction.predictedNetIncome.toLocaleString()}`, width / 2, 30)
    
    ctx.font = '12px sans-serif'
    ctx.fillText(`Confidence: ${prediction.confidence}%`, width / 2, 50)

    // Draw legend
    ctx.font = '10px sans-serif'
    ctx.textAlign = 'left'
    ctx.fillStyle = '#22c55e'
    ctx.fillRect(padding, height - 45, 15, 10)
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--foreground') || '#000000'
    ctx.fillText('Target Line', padding + 20, height - 37)
    
    ctx.fillStyle = '#6366f1'
    ctx.fillRect(padding + 100, height - 45, 8, 10)
    ctx.fillText('Feature Importance', padding + 115, height - 37)
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="w-full h-96 bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 p-4"
    >
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ width: '100%', height: '100%' }}
      />
    </motion.div>
  )
}
