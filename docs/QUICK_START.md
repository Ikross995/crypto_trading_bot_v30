🚀 Quick Start Guide - Crypto Trading Bot v3.0

⚡ Get Running in 5 Minutes

Prerequisites





Python 3.8+



Binance Futures account



API keys with futures trading permissions



📦 Installation

1. Clone Repository

git clone https://github.com/yourusername/crypto-bot.git
cd crypto-bot


2. Install Dependencies

# Using pip
pip install -r requirements.txt

# Or using poetry
poetry install


3. Setup Environment Variables

Create .env file:

# Binance API (REQUIRED)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Optional: Telegram notifications
TG_BOT_TOKEN=your_telegram_bot_token
TG_CHAT_ID=your_telegram_chat_id

# Optional: CoinMarketCap (for additional price sources)
CMC_API_KEY=your_cmc_api_key




🎮 Running the Bot

Testnet Mode (Recommended for first run)

python cli.py live --timeframe 15m --testnet --use-imba --verbose


What this does:





Connects to Binance Futures Testnet (fake money!)



Uses 15-minute timeframe



Enables IMBA signals (12 indicators)



Verbose logging for debugging

Expected Output:

🏁 Starting live trading engine...
[PRELOAD] Loading 1000 historical candles...
[PORTFOLIO] Portfolio Tracker initialized
[IMBA] Signal system ready with 12 indicators

💼 PORTFOLIO SUMMARY
================================================================================
💰 BALANCE: $10,000.00 (testnet)
...


Production Mode (Real Money!)

# WARNING: This uses REAL MONEY!
python cli.py live --timeframe 15m --use-imba --verbose


Remove --testnet flag to trade with real funds.



🎯 Quick Configuration

Basic Settings

Edit core/config.py or use environment variables:

# Trading parameters
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Trading pairs
TIMEFRAME = "15m"                             # Candle timeframe
LEVERAGE = 5                                  # Futures leverage

# Risk management
RISK_PER_TRADE = 0.5    # Risk 0.5% per trade
SL_PCT = 2.0            # Stop loss 2% from entry
EMERGENCY_SL = 20.0     # Emergency stop at -20% equity

# Signal settings
USE_IMBA_SIGNALS = True           # Enable IMBA (recommended!)
BT_CONF_MIN = 1.2                 # Confidence threshold
PRELOAD_CANDLES = 1000            # Historical data for signals


Common Configurations

Conservative (Low Risk)

RISK_PER_TRADE = 0.3
SL_PCT = 1.5
LEVERAGE = 3
BT_CONF_MIN = 1.5  # Higher threshold = fewer trades


Moderate (Default)

RISK_PER_TRADE = 0.5
SL_PCT = 2.0
LEVERAGE = 5
BT_CONF_MIN = 1.2


Aggressive (High Risk)

RISK_PER_TRADE = 1.0
SL_PCT = 3.0
LEVERAGE = 10
BT_CONF_MIN = 1.0  # Lower threshold = more trades




📊 Monitoring

Portfolio Summary (Every Minute)

Bot automatically logs portfolio status:

💼 PORTFOLIO SUMMARY - 2025-10-12 12:00:00
================================================================================

💰 BALANCE:
  ├─ Total Balance: $10,000.00
  ├─ Available: $8,500.00
  ├─ Margin Used: $1,500.00
  └─ Total Value: $10,150.00

📊 OPEN POSITIONS (2):
  1. 🟢 BTCUSDT LONG 5x
     Entry: $118,000.00 → Current: $118,500.00
     P&L: 🟢 $+50.00 (+4.24%)

📈 PERFORMANCE:
  └─ APR: +60.00% 🎯


Signal Analysis (Every Symbol Check)

🟢 IMBA SIGNAL ANALYSIS: BTCUSDT 🟢
================================================================================
📈 REGIME: TREND (ADX=26.08)

🗳️  VOTING RESULTS:
  ├─ BUY votes:  1.54 🟢
  ├─ SELL votes: 0.32 ⚪
  ├─ Base confidence: 1.22
  ├─ After Fear & Greed (×1.19): 1.45
  └─ Final confidence: 1.45 ✅

🔍 SIGNAL BREAKDOWN:
  ├─ VWAP Pullback:   BUY  0.32
  ├─ CVD:             BUY  0.68
  ├─ Volume Profile:  BUY  0.54

🎯 FINAL DECISION: BUY 🟢
  └─ Trade Action: 🚀 EXECUTE




🔧 Common Commands

View Help

python cli.py --help
python cli.py live --help


Different Timeframes

# 5-minute scalping
python cli.py live --timeframe 5m --testnet --use-imba

# 1-hour swing trading
python cli.py live --timeframe 1h --testnet --use-imba

# 4-hour position trading
python cli.py live --timeframe 4h --testnet --use-imba


Multiple Symbols

# Trade multiple pairs
python cli.py live --symbols BTCUSDT,ETHUSDT,BNBUSDT --testnet --use-imba


Disable Testnet (REAL TRADING)

# Remove --testnet flag
python cli.py live --timeframe 15m --use-imba




🛡️ Safety Checklist

Before running with real money:

✅ Pre-Flight Checklist





Tested on testnet first



Set emergency stop loss (-20% default)



Configured risk per trade (0.5% or less)



Set appropriate leverage (5x or less recommended)



Enabled trailing stop loss



Configured Telegram notifications



Verified API keys have correct permissions



Checked sufficient balance for trading



Understood all risk parameters



Read risk disclaimer (you can lose money!)

⚠️ Risk Warnings





Cryptocurrency trading is risky - You can lose all your capital



Leverage amplifies losses - Use low leverage (3-5x) initially



No guarantees - Past performance ≠ future results



Start small - Test with minimal capital first



Monitor actively - Don't leave bot unattended for long periods



🔍 Troubleshooting

"Insufficient data" Error

WARNING | Insufficient data for IMBA signals: 250 candles (need >= 250)


Fix: Increase PRELOAD_CANDLES or wait for data to accumulate

"Rate limit" (429) Error

ERROR | BinanceAPIException: 429 - Too many requests


Fix: Reduce number of symbols or increase check interval

"Timestamp" (-1021) Error

ERROR | Timestamp for this request is outside of the recvWindow


Fix: Sync system clock or bot auto-retries after sync

"Invalid API key" Error

ERROR | BinanceAPIException: 401 - Invalid API key


Fix: Check API key/secret in .env file

"Position already exists" Warning

INFO | [POSITION_EXISTS] Already have BUY position for BTCUSDT, skipping


Fix: This is normal - bot prevents duplicate positions



📈 Performance Tips

1. Start Conservative





Low risk per trade (0.3-0.5%)



High confidence threshold (1.3-1.5)



Low leverage (3-5x)

2. Optimize Gradually





Monitor performance for 1-2 weeks



Adjust parameters based on results



Increase risk only if consistent profits

3. Use Portfolio Tracking





Monitor APR and Sharpe Ratio



Track max drawdown



Adjust if drawdown > 10%

4. Best Timeframes





5m: Scalping (high frequency, lower confidence needed)



15m: Day trading (balanced)



1h: Swing trading (fewer trades, higher quality)



4h: Position trading (best signal quality)



🎓 Next Steps

Learning Resources





System Architecture - Understand how it works



Configuration Guide - Detailed tuning



Trading Strategy - IMBA signals explained



Risk Management - Protect your capital

Advanced Features





DCA (Dollar Cost Averaging): Add --use-dca flag



LSTM Predictions: Enable LSTM_ENABLE=true



Custom Indicators: Add to strategy/imba_signals.py



Telegram Alerts: Setup TG_BOT_TOKEN and TG_CHAT_ID



📞 Support

Getting Help





Documentation: Read all docs in /docs folder



Issues: Check existing GitHub issues first



Discord: Join our trading community



Telegram: Enable TG notifications for bot alerts

Reporting Bugs

Include in your report:





Full command used



Error message/logs



Config settings



Python version



Exchange (testnet/mainnet)



🎉 You're Ready!

Start trading safely:

# Always start with testnet!
python cli.py live --timeframe 15m --testnet --use-imba --verbose


Good luck and trade responsibly! 🚀



Version: 3.0
Last Updated: 2025-10-12
Status: Production Ready ✅