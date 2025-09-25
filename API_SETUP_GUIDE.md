# 🔑 Real Stock Market Data Setup Guide

## Get Your Free Alpha Vantage API Key

To enable real-time stock market data in your Euler dashboard, follow these simple steps:

### Step 1: Get Your Free API Key
1. Visit: [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
2. Enter your email address
3. Click "GET FREE API KEY"
4. Copy the API key they provide

### Step 2: Configure Your Environment
Create a `.env.local` file in your project root and add:

```
NEXT_PUBLIC_ALPHA_VANTAGE_API_KEY=YOUR_ACTUAL_API_KEY_HERE
```

### Step 3: Restart Your Development Server
```bash
npm run dev
```

## 📊 What You Get With Real Data

### Free Tier Includes:
- ✅ **25 API calls per day**
- ✅ **5 API calls per minute**
- ✅ **Real-time stock quotes**
- ✅ **Technical indicators (RSI, SMA, MACD)**
- ✅ **Company information**
- ✅ **Historical data**

### Features Enabled:
- 🎯 **Real Current Prices** - Live market data
- 📈 **Actual Technical Analysis** - Real RSI, MACD, moving averages
- 🏢 **Company Information** - Real company names, sectors, market cap
- 🤖 **Smart Predictions** - AI analysis based on real market conditions
- 📊 **Live Market Indicators** - Real-time market sentiment

## 🔄 Fallback System

Don't worry if you don't have an API key! The system includes:

- **Automatic Fallback** - If real data fails, it uses mock data
- **Visual Indicators** - Shows whether you're viewing real or demo data
- **Seamless Experience** - Works perfectly with or without API key

## 🚀 API Alternatives

If you need more requests, consider these alternatives:

### IEX Cloud (Free Tier)
- 500,000 requests/month free
- Excellent for high-volume applications

### Finnhub (Free Tier)  
- 60 calls/minute
- Great for real-time data

### Yahoo Finance (Unofficial)
- Unlimited requests
- Used as fallback in our implementation

## 🔒 Security Notes

- Your API key is stored in environment variables
- Never commit API keys to version control
- The `.env.local` file is automatically ignored by Git

## 🎯 Testing Your Setup

1. Enter a stock symbol (like AAPL, TSLA, MSFT)
2. Look for the "Live Market Data" indicator
3. Check the browser console for API calls
4. Real data will show actual current prices and indicators

Enjoy your real-time stock market predictions! 🚀
