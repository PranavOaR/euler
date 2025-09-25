// Dashboard utility functions

export const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 2
  }).format(amount)
}

export const formatPercentage = (value: number, decimals = 1): string => {
  return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`
}

export const formatCompactNumber = (value: number): string => {
  if (value >= 1e9) {
    return (value / 1e9).toFixed(1) + 'B'
  }
  if (value >= 1e6) {
    return (value / 1e6).toFixed(1) + 'M'
  }
  if (value >= 1e3) {
    return (value / 1e3).toFixed(1) + 'K'
  }
  return value.toString()
}

export const getTimeAgo = (timestamp: string): string => {
  const now = new Date()
  const past = new Date(timestamp)
  const diffInMs = now.getTime() - past.getTime()
  
  const minutes = Math.floor(diffInMs / (1000 * 60))
  const hours = Math.floor(diffInMs / (1000 * 60 * 60))
  const days = Math.floor(diffInMs / (1000 * 60 * 60 * 24))
  
  if (days > 0) return `${days}d ago`
  if (hours > 0) return `${hours}h ago`
  if (minutes > 0) return `${minutes}m ago`
  return 'Just now'
}

export const generateGradient = (direction: 'up' | 'down'): string => {
  return direction === 'up' 
    ? 'from-green-500/10 to-emerald-500/10'
    : 'from-red-500/10 to-rose-500/10'
}

export const getStatusColor = (status: string): string => {
  switch (status.toLowerCase()) {
    case 'running':
    case 'active':
    case 'open':
      return 'text-green-500'
    case 'paused':
    case 'pending':
      return 'text-yellow-500'
    case 'completed':
    case 'closed':
      return 'text-gray-500'
    case 'error':
    case 'failed':
      return 'text-red-500'
    default:
      return 'text-gray-500'
  }
}

export const validateStockSymbol = (symbol: string): boolean => {
  // Basic validation for stock symbols (1-5 letters)
  const regex = /^[A-Z]{1,5}$/
  return regex.test(symbol.toUpperCase())
}

export const getMarketStatus = (): { isOpen: boolean, status: string } => {
  const now = new Date()
  const hours = now.getHours()
  const day = now.getDay()
  
  // Simple market hours check (9:30 AM - 4:00 PM EST, Mon-Fri)
  const isWeekday = day >= 1 && day <= 5
  const isMarketHours = hours >= 9 && hours < 16
  
  return {
    isOpen: isWeekday && isMarketHours,
    status: isWeekday && isMarketHours ? 'Market Open' : 'Market Closed'
  }
}

export const calculateROI = (initial: number, current: number): number => {
  return ((current - initial) / initial) * 100
}

export const calculateSharpeRatio = (returns: number[], riskFreeRate = 0.02): number => {
  if (returns.length === 0) return 0
  
  const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
  const stdDev = Math.sqrt(variance)
  
  return stdDev === 0 ? 0 : (avgReturn - riskFreeRate) / stdDev
}

export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeout)
    timeout = setTimeout(() => func(...args), wait)
  }
}

