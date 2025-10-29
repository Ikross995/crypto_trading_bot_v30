🧠💰 AI Trading Bot с Enhanced Learning Dashboard



Advanced AI-powered trading bot with real-time learning visualization and comprehensive performance analytics



📋 Содержание





🎯 Описание



✨ Возможности



📦 Зависимости



🚀 Установка



⚙️ Конфигурация



🏃 Запуск



📊 Learning Dashboard



🔧 Troubleshooting



📚 Документация



🎯 Описание

AI Trading Bot - это продвинутый торговый бот для криптовалютных фьючерсов на Binance с возможностями машинного обучения, адаптивной оптимизации параметров и comprehensive real-time dashboard для мониторинга производительности.

🔥 Ключевые особенности:





🧠 Адаптивное обучение - бот автоматически оптимизирует параметры на основе результатов



💰 Реальные данные - интеграция с Binance API для live P&L и account data



📊 Professional Dashboard - 16 карточек с 40+ метриками производительности



🎯 DCA система - Dollar Cost Averaging с умной адаптацией к рынку



⚡ Real-time мониторинг - живое отслеживание всех параметров и позиций



✨ Возможности

🤖 AI & Machine Learning





Adaptive Learning System - непрерывная оптимизация параметров



Advanced Intelligence - Bayesian Optimization, Multi-Armed Bandit



Pattern Recognition - обнаружение паттернов и market regime detection



A/B Testing - статистическое тестирование стратегий



Real-time Adaptation - мгновенная адаптация к изменениям рынка

📈 Trading Features





Multiple Timeframes - поддержка разных таймфреймов



Smart DCA - продвинутая система усреднения позиций



Risk Management - comprehensive система управления рисками



Take Profit Levels - многоуровневая фиксация прибыли



Stop Loss Protection - адаптивные стоп-лоссы с ATR

📊 Analytics & Monitoring





Real-time Dashboard - HTML dashboard с auto-refresh



Performance Metrics - 40+ профессиональных метрик



Account Integration - live данные с Binance (balance, PnL, positions)



Risk Analytics - Kelly Criterion, Sortino Ratio, Risk/Reward



Market Intelligence - volatility, trend analysis, volume profile

🔧 Technical Features





Binance Futures - полная интеграция с Binance Futures API



Paper Trading - безопасное тестирование стратегий



Backtesting - historical данные для оптимизации



Error Handling - robust error handling и recovery



Logging - comprehensive логирование всех операций



📦 Зависимости

🐍 Python Version





Python 3.10+ (обязательно)

📚 Core Dependencies

# Trading & Exchange
python-binance>=1.0.17      # Binance API client
ccxt>=4.0.0                  # Cryptocurrency exchange trading library

# Data Analysis & ML  
pandas>=2.0.0                # Data manipulation and analysis
numpy>=1.24.0                # Numerical computing
scikit-learn>=1.3.0          # Machine learning library
scipy>=1.10.0                # Scientific computing (for advanced AI)

# Async & Networking
aiohttp>=3.8.0               # Async HTTP client/server
websockets>=11.0             # WebSocket client and server
asyncio                      # Built-in async library

# Configuration & Environment
pydantic>=2.0.0              # Data validation using Python type annotations
python-dotenv>=1.0.0         # Load environment variables from .env

# Logging & Monitoring
loguru>=0.7.0                # Structured logging

# Visualization (Optional but Recommended)
matplotlib>=3.7.0            # Plotting library
seaborn>=0.12.0              # Statistical data visualization
plotly>=5.15.0               # Interactive plotting (for dashboard)

# Development & Testing
pytest>=7.4.0               # Testing framework
pytest-asyncio>=0.21.0      # Async testing support


🌐 Web Dependencies (for Dashboard)

Dashboard использует CDN, поэтому дополнительных установок не требуется:





Plotly.js (загружается из CDN)



Modern Web Browser для просмотра dashboard



🚀 Установка

1️⃣ Клонирование репозитория

git clone <repository-url>
cd ai_trading_bot_pytorch


2️⃣ Создание виртуального окружения

# Создание виртуального окружения
python -m venv venv

# Активация (Linux/Mac)
source venv/bin/activate

# Активация (Windows)
venv\Scripts\activate


3️⃣ Установка зависимостей

Базовая установка:

pip install python-binance pandas numpy scikit-learn aiohttp pydantic python-dotenv loguru


Полная установка с visualization:

pip install python-binance pandas numpy scikit-learn scipy aiohttp pydantic python-dotenv loguru matplotlib seaborn plotly pytest pytest-asyncio ccxt websockets


Или из requirements.txt (если есть):

pip install -r requirements.txt


4️⃣ Проверка установки

python -c "
import asyncio
import pandas as pd
import numpy as np
from binance.client import Client
print('✅ All core dependencies installed successfully!')
"




⚙️ Конфигурация

🔐 API Keys Configuration

1. Создайте .env файл:

cp config/.env.example .env


2. Заполните .env файл:

# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# Trading Configuration  
TRADING_MODE=LIVE                    # PAPER/LIVE
TESTNET=false                        # true for testnet
DRY_RUN=false                        # true for simulation

# Risk Management
LEVERAGE=5                           # Leverage (1-125)
RISK_PER_TRADE_PCT=0.5              # Risk per trade (0.1-10.0%)
MAX_DAILY_LOSS_PCT=5.0              # Max daily loss (1.0-50.0%)

# Signal Configuration
BT_CONF_MIN=0.45                    # Signal confidence threshold
COOLDOWN_SEC=120                    # Cooldown between trades (seconds)

# Learning Configuration
ENABLE_ADAPTIVE_LEARNING=true       # Enable AI learning
ENABLE_ADVANCED_AI=true             # Enable advanced AI features


📊 Trading Parameters

Основные параметры можно настроить в core/config.py или через environment variables:

# Risk Management
leverage: int = 5                    # Плечо
risk_per_trade_pct: float = 0.5     # Риск на сделку (%)
max_daily_loss_pct: float = 5.0     # Максимальный дневной убыток (%)

# Signal Configuration  
bt_conf_min: float = 0.45           # Минимальная уверенность сигнала
cooldown_sec: int = 120             # Cooldown между сделками

# Take Profit & Stop Loss
sl_fixed_pct: float = 2.0           # Stop Loss (%)
tp_levels: str = "1.5,3.0,5.0"     # Take Profit levels (%)
tp_shares: str = "0.4,0.35,0.25"   # Распределение TP

# DCA Configuration
dca_ladder_str: str = "-0.6:1.0,-1.2:1.5,-2.0:2.0"  # DCA levels
adaptive_dca: bool = True           # Адаптивный DCA
max_levels: int = 3                 # Максимум DCA уровней


🧠 AI Learning Configuration

# Adaptive Learning
enable_adaptive_learning: bool = True
optimization_interval_hours: int = 24
min_trades_for_optimization: int = 20

# Advanced AI
enable_advanced_ai: bool = True
bayesian_optimization: bool = True
pattern_recognition: bool = True




🏃 Запуск

🎯 Live Trading

# Запуск live торговли
python runner/live.py

# Или через CLI с параметрами
python cli.py --mode live --symbol BTCUSDT --risk 0.5


📄 Paper Trading (Рекомендуется для начала)

# Запуск paper trading
python runner/paper.py

# Или через CLI
python cli.py --mode paper --symbol BTCUSDT


🔄 Backtesting

# Запуск backtesting
python runner/backtest.py --days 30 --symbol BTCUSDT

# Backtest с параметрами
python runner/backtest.py --days 90 --symbols BTCUSDT,ETHUSDT --risk 1.0


🧪 Testing Configuration

# Тест конфигурации
python -c "
from core.config import Config
config = Config()
print('✅ Configuration loaded successfully!')
print(f'Trading mode: {config.mode}')
print(f'Risk per trade: {config.risk_per_trade_pct}%')
"




📊 Learning Dashboard

🚀 Доступ к Dashboard

После запуска бота dashboard будет автоматически генерироваться:

# Dashboard location
data/learning_reports/learning_dashboard.html


Открытие Dashboard:





Запустите торгового бота (live или paper)



Подождите 2-3 минуты для накопления данных



Откройте в браузере: data/learning_reports/learning_dashboard.html



Dashboard обновляется каждые 60 секунд автоматически

📈 Dashboard Features

💰 Real-Time Account Data





Account Balance - текущий баланс счета



Unrealized PnL - нереализованная прибыль/убыток



Total Wallet Balance - общий баланс кошелька



Available Balance - доступный баланс



Margin Used - используемая маржа



Margin Ratio - коэффициент маржи

🎯 Position Management





Open Positions - количество открытых позиций



Total Position Value - общая стоимость позиций



Largest Position - крупнейшая позиция

📊 Performance Metrics





Win Rate - процент прибыльных сделок



Profit Factor - коэффициент прибыльности



Total PnL - общая прибыль/убыток (REAL DATA)



Avg Win/Loss - средняя прибыльная/убыточная сделка

🎲 Risk Analytics





Risk/Reward Ratio - соотношение риск/доходность



Kelly Criterion - оптимальный размер позиции



Sortino Ratio - коэффициент Сортино



Max Drawdown - максимальная просадка

🌍 Market Intelligence





Market Volatility - текущая волатильность



Market Trend - направление рынка (bullish/bearish/neutral)



24h Price Change - изменение цены за 24 часа

🧠 AI Learning Status





Adaptations Count - количество адаптаций



Learning Confidence - уверенность AI системы



Last Adaptation Trigger - последний триггер изменений



AI Recommendations - текущие рекомендации AI

📁 Generated Files

data/learning_reports/
├── learning_dashboard.html    # 🎯 Main interactive dashboard
├── learning_history.json      # 📊 Historical learning data  
├── adaptations.json           # 🔧 Parameter adaptation log
└── real_time_report_*.txt     # 📝 Timestamped status reports




🔧 Troubleshooting

❌ Common Issues

1. API Connection Errors

# Error: APIError(code=-1021): Timestamp for this request is outside of the recvWindow
Solution: Check system time synchronization


2. Import Errors

# Error: ModuleNotFoundError: No module named 'binance'
Solution: pip install python-binance


3. Dashboard Not Loading

# Issue: Dashboard shows "No learning data available"
Solution: 
1. Ensure bot is running
2. Wait 2-3 minutes for data accumulation
3. Check data/learning_reports/ directory exists


4. Permission Errors

# Error: PermissionError when creating files
Solution: Check file permissions for data/ directory
chmod 755 data/learning_reports/


🔍 Debug Mode

# Enable debug logging
export LOG_LEVEL=DEBUG
python runner/live.py

# Or in Python
import logging
logging.basicConfig(level=logging.DEBUG)


📊 Dashboard Troubleshooting

Dashboard Not Updating





Check bot is running and generating trades



Verify files in data/learning_reports/ are being updated



Refresh browser (Ctrl+F5)



Check browser console for JavaScript errors

Charts Not Loading





Ensure internet connection (Plotly loads from CDN)



Check browser supports modern JavaScript



Try different browser (Chrome, Firefox recommended)

No Real PnL Data





Verify Binance API keys are correct



Check API permissions include futures trading



Ensure bot has executed at least one trade

🆘 Getting Help

Log Locations

# Bot logs
logs/trading_bot.log

# Learning system logs  
data/learning_reports/real_time_report_*.txt

# Error logs
Check console output for immediate errors


Performance Monitoring

# Check system resources
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
"




📚 Документация

📖 Дополнительные руководства





Learning Visualization Guide - Подробное руководство по dashboard



Pull Request Documentation - Техническая документация изменений



Configuration Guide - Детальная настройка параметров

🔗 API Documentation





Binance Futures API - Official Binance API docs



Python-Binance - Python Binance client docs

📊 Strategy Documentation

Signal Generation

Bot использует комбинацию технических индикаторов:





IMBA Analysis - дисбаланс спроса/предложения



FVG (Fair Value Gap) - анализ пропусков в цене



CVD (Cumulative Volume Delta) - накопленная дельта объема



ADX/DX - сила тренда



VWAP - объемно-взвешенная средняя цена

Risk Management





Position Sizing - основан на Kelly Criterion и Risk/Reward



Stop Loss - адаптивный SL на основе ATR



Take Profit - многоуровневая фиксация прибыли



DCA System - умное усреднение позиций

Learning System





Bayesian Optimization - оптимизация параметров



Pattern Recognition - обнаружение market patterns



A/B Testing - статистическое тестирование



Adaptive Parameters - автоматическая адаптация к рынку



🚀 Quick Start Checklist

✅ Pre-Trading Checklist





✅ Dependencies installed - Все зависимости установлены



✅ API keys configured - Binance API ключи настроены



✅ Configuration reviewed - Параметры торговли проверены



✅ Paper trading tested - Paper trading протестирован



✅ Dashboard accessible - Dashboard открывается в браузере



✅ Risk limits set - Лимиты риска установлены

🎯 First Run Steps

# 1. Start with paper trading
python runner/paper.py

# 2. Open dashboard in browser
open data/learning_reports/learning_dashboard.html

# 3. Monitor for 30 minutes to verify operation

# 4. If satisfied, switch to live trading
# Edit .env: TRADING_MODE=LIVE, DRY_RUN=false
python runner/live.py


📊 Success Indicators





✅ Bot connects to Binance successfully



✅ Dashboard shows real account data



✅ Learning metrics are updating



✅ No error messages in logs



✅ PnL data reflects actual account status



🏆 Summary

AI Trading Bot представляет собой comprehensive решение для автоматической торговли криптовалютами с продвинутыми возможностями машинного обучения и real-time analytics.

🎯 Key Benefits:





💰 Real-time PnL tracking с интеграцией Binance API



🧠 Adaptive learning для непрерывного улучшения



📊 Professional dashboard с 40+ метриками



🛡️ Advanced risk management с Kelly Criterion



⚡ Production-ready с comprehensive error handling

Ready for live trading with full transparency and professional-grade analytics! 🚀
