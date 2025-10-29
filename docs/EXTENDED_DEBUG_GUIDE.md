✨ Features

🔍 All 9 Signals Visible

Unlike the compact mode (which only shows active signals), Extended Debug shows:





✅ Active signals - Contributing votes



⏸️ Waiting signals - Monitoring but conditions not met



🎯 Vote weights - How much each signal contributes



📈 Regime multipliers - How market regime affects each signal

🎨 Modern Visual Design





🟢 BUY signals with green indicators



🔴 SELL signals with red indicators



⏸️ WAIT signals with pause indicators



✅ Pass/fail visual confirmations



📊 Structured borders and sections

📈 Complete Information





Market regime detection (trend/flat/volatile)



ADX and BBW values



Vote breakdown (BUY vs SELL)



Confidence scoring



Filter pass/fail status



Final trade decision



🚀 Usage

Method 1: Launch with Multi-Symbol (Recommended)

# Multi-symbol trading (automatically uses extended debug)
cd C:\Users\User\AI_Trading_Bot\crypto_trading_bot\cripto_ai_bot
python cli.py live --timeframe 15m --testnet --use-imba --verbose


Method 2: Single Symbol with Debug

# Single symbol with extended debug
python cli.py live --symbol BTCUSDT --timeframe 15m --testnet --use-imba --verbose


Method 3: Using 15-Minute Configuration

# Copy 15m config to .env
copy .env.15m .env

# Edit .env and add your API keys:
# BINANCE_API_KEY=your_key_here
# BINANCE_API_SECRET=your_secret_here

# Launch
python cli.py live --config .env --verbose




📺 Output Example

When Extended Debug Mode is active, you'll see this beautiful output:

================================================================================
🟢 IMBA SIGNAL ANALYSIS: BTCUSDT 🟢
================================================================================
⚡ REGIME: VOLATILE
  ├─ ADX: 21.18
  ├─ BBW: 0.0023
  └─ Confidence: 0.60

🗳️  VOTING RESULTS:
  ├─ BUY votes:  1.112 🟢
  ├─ SELL votes: 0.000 ⚪
  └─ Final confidence: 1.112 ✅

🔍 SIGNAL BREAKDOWN (9 indicators):
  ├─ 1️⃣  BB Squeeze        │ ⏸️  WAIT    │ Vote: 0.00   │ Regime×: 1.00
  ├─ 2️⃣  VWAP Pullback     │ ⏸️  WAIT    │ Vote: 0.00   │ Regime×: 1.20
  ├─ 3️⃣  VWAP Mean Rev     │ 🟢 BUY      │ Vote: +0.60  │ Regime×: 1.20
  ├─ 4️⃣  Breakout Retest   │ ⏸️  WAIT    │ Vote: 0.00   │ Regime×: 0.80
  ├─ 5️⃣  ATR Momentum      │ 🟢 BUY      │ Vote: +0.70  │ Regime×: 1.30
  ├─ 6️⃣  RSI Mean Rev      │ ⏸️  WAIT    │ Vote: 0.00   │ Regime×: 1.10
  ├─ 7️⃣  Swing Failure     │ ⏸️  WAIT    │ Vote: 0.00   │ Regime×: 1.00
  ├─ 8️⃣  EMA Pinch         │ ⏸️  WAIT    │ Vote: 0.00   │ Regime×: 1.00
  └─ 9️⃣  Order Imbalance   │ ⏸️  WAIT    │ Vote: 0.00   │ Regime×: 1.00

🎯 FINAL DECISION: BUY 🟢
  ├─ Signal Strength: 1.112
  ├─ Filters Passed: ✅ YES
  └─ Trade Action: 🚀 EXECUTE
================================================================================




📖 Understanding the Output

🌍 Regime Section

Shows current market condition:





📈 TREND: Strong directional movement (ADX > 30)



📊 FLAT: Range-bound market (ADX < 20)



⚡ VOLATILE: High volatility, mixed signals

⚡ REGIME: VOLATILE
  ├─ ADX: 21.18        ← Trend strength (0-100)
  ├─ BBW: 0.0023       ← Bollinger Band Width (volatility)
  └─ Confidence: 0.60  ← How sure we are of regime


🗳️ Voting Results

Aggregated votes from all signals:

🗳️  VOTING RESULTS:
  ├─ BUY votes:  1.112 🟢   ← Total BUY weight
  ├─ SELL votes: 0.000 ⚪   ← Total SELL weight
  └─ Final confidence: 1.112 ✅  ← Net confidence (must be > 0.70)


🔍 Signal Breakdown

Each of the 9 IMBA signals:







Column



Description





Number



Signal ID (1️⃣-9️⃣)





Name



Signal name (e.g., BB Squeeze)





Status



🟢 BUY / 🔴 SELL / ⏸️ WAIT





Vote



Contribution weight





Regime×



Multiplier based on market regime

Example:

├─ 3️⃣  VWAP Mean Rev     │ 🟢 BUY      │ Vote: +0.60  │ Regime×: 1.20






Signal #3 (VWAP Mean Reversion)



Voting BUY



Contributing +0.60 to BUY votes



Regime multiplier 1.20× (boosted in volatile market)

🎯 Final Decision

🎯 FINAL DECISION: BUY 🟢
  ├─ Signal Strength: 1.112      ← Overall confidence
  ├─ Filters Passed: ✅ YES      ← Risk filters OK
  └─ Trade Action: 🚀 EXECUTE    ← Will place order




🔢 The 9 IMBA Signals Explained

1️⃣ BB Squeeze





What: Bollinger Bands squeeze breakout



Triggers: BBW in lowest 20%, then price breaks band



Best in: Volatile regime (breakout trading)

2️⃣ VWAP Pullback





What: Pullback to VWAP after trend



Triggers: Price returns to VWAP with bounce



Best in: Trend regime (continuation)

3️⃣ VWAP Mean Rev





What: Mean reversion using VWAP bands



Triggers: Price at VWAP ± 0.8% band



Best in: Volatile/flat regime

4️⃣ Breakout Retest





What: Donchian channel breakout with retest



Triggers: New high/low, then successful retest



Best in: Trend regime (breakout confirmation)

5️⃣ ATR Momentum





What: Momentum with ATR expansion



Triggers: Price move > 1.2× ATR



Best in: Volatile regime (strong moves)

6️⃣ RSI Mean Rev





What: RSI oversold/overbought reversal



Triggers: RSI < 30 (BUY) or RSI > 70 (SELL)



Best in: Flat regime (range trading)

7️⃣ Swing Failure





What: False breakout pattern (SFP)



Triggers: New high/low fails, reverses



Best in: All regimes (trap detection)

8️⃣ EMA Pinch





What: EMA convergence squeeze



Triggers: EMA gap in lowest 15%



Best in: Trend regime (breakout pending)

9️⃣ Order Imbalance





What: Order book imbalance (requires WebSocket)



Triggers: Bid/ask imbalance > 0.18



Best in: All regimes (institutional flow)



⚙️ Regime Multipliers

Each signal gets boosted or reduced based on market regime:







Signal



Trend ×



Flat ×



Volatile ×





BB Squeeze



0.8



1.0



1.3





VWAP Pullback



1.3



0.9



1.2





VWAP Mean Rev



0.9



1.2



1.2





Breakout Retest



1.3



0.7



1.0





ATR Momentum



1.2



0.8



1.3





RSI Mean Rev



0.8



1.3



1.1





Swing Failure



1.0



1.0



1.0





EMA Pinch



1.3



0.8



1.0





Order Imbalance



1.0



1.0



1.0

Example: VWAP Pullback gets 1.3× boost in TREND regime (best for continuation)



🎯 Configuration Tips

For Maximum Transparency:

# In your .env file:
USE_IMBA_SIGNALS=true     # Enable IMBA
BT_CONF_MIN=0.70          # Min confidence (lower = more signals)
SIGNAL_COOLDOWN_SECONDS=300  # 5 minutes between signals


15-Minute Timeframe (Recommended):

TIMEFRAME=15m             # Less noise than 1m
LEVERAGE=10               # Moderate aggressive
RISK_PER_TRADE_PCT=1.0    # 1% risk per trade
SL_FIXED_PCT=2.0          # 2% stop loss
TP_LEVELS=1.0,2.5,4.0     # Multiple take profits




📊 Reading the Logs

WAIT Signal Example:

⏸️ IMBA SIGNAL ANALYSIS: ETHUSDT ⏸️
...
🎯 FINAL DECISION: WAIT ⏸️
  ├─ Signal Strength: 0.450      ← Too low (< 0.70 threshold)
  ├─ Filters Passed: ❌ NO       ← Failed risk filters
  └─ Trade Action: ⏸️  WAIT      ← No trade


Why? Confidence below threshold OR filters failed.

BUY Signal Example:

🟢 IMBA SIGNAL ANALYSIS: BTCUSDT 🟢
...
🗳️  VOTING RESULTS:
  ├─ BUY votes:  1.350 🟢
  ├─ SELL votes: 0.200 ⚪
  └─ Final confidence: 1.150 ✅

🎯 FINAL DECISION: BUY 🟢
  ├─ Signal Strength: 1.150
  ├─ Filters Passed: ✅ YES
  └─ Trade Action: 🚀 EXECUTE


Result: Bot will place BUY order!



🐛 Troubleshooting

Extended Debug Not Showing?





Check verbose flag:

python cli.py live --verbose  # ← Must include --verbose




Check IMBA enabled:

python cli.py live --use-imba  # ← Must include --use-imba




Check log level in code:

# In runner/live.py or strategy/imba_integration.py
logger.setLevel(logging.INFO)  # ← Should be INFO or DEBUG


Signals Always WAIT?

Check these thresholds in your .env:

BT_CONF_MIN=0.70           # Try lowering to 0.60
MIN_ADX=25.0               # Try lowering to 20.0
PRELOAD_CANDLES=500        # Must have enough data


Too Many Signals?

Increase cooldown:

SIGNAL_COOLDOWN_SECONDS=300  # 5 minutes (was 60)
COOLDOWN_SEC=300             # Global cooldown




🚀 Next Steps





Launch with 15m config:

copy .env.15m .env
# Edit API keys
python cli.py live --config .env --verbose




Monitor the beautiful logs:





See all 9 signals



Understand vote weights



Track regime changes



Verify filter logic



Adjust parameters:





Lower BT_CONF_MIN for more signals



Raise LEVERAGE for bigger positions



Adjust TP_LEVELS for profit targets



Analyze performance:





Watch which signals trigger most



Note regime performance



Track win rates per signal



🎉 Benefits





✅ Complete transparency - See every decision



✅ Beautiful UI - Emoji-based modern design



✅ Educational - Learn what works



✅ Debugging - Find issues quickly



✅ Confidence - Know why bot trades



📚 Related Docs





IMBA Integration Guide



15-Minute Trading Guide



Configuration Reference



ENJOY YOUR BEAUTIFUL TRADING BOT! 🎨🚀💎