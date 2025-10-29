#!/usr/bin/env python3
"""
Market Context Manager

Анализирует рынок перед стартом бота и восстанавливает контекст после перезапуска.
Проводит быстрый бэктест на недавних данных для понимания рыночной ситуации.
"""

import asyncio
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from loguru import logger

from core.config import Config
from core.constants import OrderSide, Timeframe
from core.types import Signal
from data.fetchers import HistoricalDataFetcher


@dataclass
class MarketContextData:
    """Данные о контексте рынка."""
    
    # Временные рамки
    analysis_start: datetime
    analysis_end: datetime
    last_update: datetime
    
    # Рыночная ситуация
    overall_trend: str  # "BULLISH", "BEARISH", "SIDEWAYS"
    volatility_level: str  # "LOW", "MEDIUM", "HIGH", "EXTREME"
    recent_signals: List[Dict[str, Any]]
    
    # Производительность стратегии на недавних данных
    backtest_win_rate: float
    backtest_total_trades: int
    backtest_pnl_pct: float
    backtest_max_drawdown: float
    
    # Рекомендации для торговли
    suggested_confidence_threshold: float
    suggested_position_size_multiplier: float
    risk_warning_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    
    # Активные позиции (если есть)
    active_positions: List[Dict[str, Any]]
    
    # Дополнительная аналитика
    market_conditions: Dict[str, Any]


class MarketContextManager:
    """
    Менеджер рыночного контекста для бота.
    
    Функции:
    - Проводит быстрый бэктест перед стартом
    - Анализирует текущие рыночные условия  
    - Восстанавливает состояние после перезапуска
    - Предоставляет рекомендации по торговле
    """
    
    def __init__(self, config: Config, data_fetcher: HistoricalDataFetcher):
        self.config = config
        self.data_fetcher = data_fetcher
        self.context_file = Path("market_context.json")
        self.current_context: Optional[MarketContextData] = None
        
    async def initialize_market_context(self, force_refresh: bool = False) -> MarketContextData:
        """
        Инициализирует рыночный контекст перед стартом торговли.
        
        Args:
            force_refresh: Принудительно обновить анализ
            
        Returns:
            MarketContextData: Данные о рыночном контексте
        """
        logger.info("🔍 [MARKET_CONTEXT] Initializing market context analysis...")
        
        # Проверяем, есть ли свежий кэш
        if not force_refresh and self._is_context_fresh():
            logger.info("📋 [MARKET_CONTEXT] Using cached market context (fresh)")
            return self._load_cached_context()
        
        # Проводим полный анализ
        logger.info("🧮 [MARKET_CONTEXT] Performing fresh market analysis...")
        
        # 1. Анализ временного окна (последние 50 дней для максимального обучения AI)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=50)
        
        # 2. Получаем исторические данные
        logger.info(f"📊 [MARKET_CONTEXT] Fetching data: {start_time} to {end_time}")
        
        symbols = getattr(self.config, 'symbols', ['BTCUSDT'])
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Фокусируемся на основном символе для анализа
        primary_symbol = symbols[0] if symbols else 'BTCUSDT'
        
        try:
            # FIXED: Use direct Binance client to get klines, then convert to DataFrame
            if hasattr(self.data_fetcher, 'client') and self.data_fetcher.client:
                logger.info(f"📊 [MARKET_CONTEXT] Fetching {primary_symbol} 1h data via direct API...")
                
                # Get raw klines from Binance
                logger.info(f"📊 [MARKET_CONTEXT] Starting to fetch 1200 candles for BACKTEST training...")
                raw_klines = self.data_fetcher.client.get_klines(
                    symbol=primary_symbol,
                    interval='1h',
                    limit=1200  # 50 days * 24 hours - MAXIMUM data for AI training and accurate backtest
                )
                logger.info(f"📊 [MARKET_CONTEXT] Successfully fetched {len(raw_klines) if raw_klines else 0} candles for AI training")
                
                if not raw_klines:
                    logger.error("❌ [MARKET_CONTEXT] No klines data received from API!")
                    return self._create_default_context()
                
                # Convert to DataFrame manually  
                import pandas as pd
                columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                          'close_time', 'quote_asset_volume', 'number_of_trades',
                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
                
                df = pd.DataFrame(raw_klines, columns=columns)
                
                # Convert timestamp and set as index
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Convert numeric columns
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                logger.info(f"✅ [MARKET_CONTEXT] Converted {len(df)} klines to DataFrame")
                
            else:
                # Fallback to original method
                logger.info(f"📊 [MARKET_CONTEXT] Using fallback method to fetch 1200 candles for BACKTEST...")
                df = self.data_fetcher.get_historical_data(
                    symbol=primary_symbol,
                    timeframe='1h',
                    start_date=start_time,
                    end_date=end_time,
                    limit=1200
                )
            
            if df is None or df.empty:
                logger.error("❌ [MARKET_CONTEXT] No historical data available!")
                return self._create_default_context()
            
            # Проверяем достаточность данных для IMBA сигналов
            if len(df) < 250:
                logger.warning(f"⚠️ [MARKET_CONTEXT] Limited data for IMBA: {len(df)} candles (recommended 250+)")
            else:
                logger.info(f"✅ [MARKET_CONTEXT] Sufficient data for IMBA: {len(df)} candles")
                
        except Exception as e:
            logger.error(f"❌ [MARKET_CONTEXT] Error fetching data: {e}")
            import traceback
            logger.error(f"❌ [MARKET_CONTEXT] Traceback: {traceback.format_exc()}")
            return self._create_default_context()
        
        # 3. Проводим анализ рынка
        market_analysis = self._analyze_market_conditions(df)
        
        # 4. Выполняем быстрый бэктест
        backtest_results = await self._run_quick_backtest(df, primary_symbol)
        
        # 5. Генерируем рекомендации
        recommendations = self._generate_trading_recommendations(market_analysis, backtest_results)
        
        # 6. Создаем контекст
        context = MarketContextData(
            analysis_start=start_time,
            analysis_end=end_time,
            last_update=datetime.now(timezone.utc),
            
            overall_trend=market_analysis['trend'],
            volatility_level=market_analysis['volatility'],
            recent_signals=market_analysis['signals'],
            
            backtest_win_rate=backtest_results['win_rate'],
            backtest_total_trades=backtest_results['total_trades'],
            backtest_pnl_pct=backtest_results['pnl_pct'],
            backtest_max_drawdown=backtest_results['max_drawdown'],
            
            suggested_confidence_threshold=recommendations['confidence_threshold'],
            suggested_position_size_multiplier=recommendations['position_multiplier'],
            risk_warning_level=recommendations['risk_level'],
            
            active_positions=[],  # TODO: Получить из exchange
            market_conditions=market_analysis
        )
        
        # 7. Сохраняем контекст
        self._save_context(context)
        self.current_context = context
        
        # 8. Логируем результаты
        self._log_market_summary(context)
        
        return context
    
    def _analyze_market_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализирует рыночные условия на основе данных."""
        try:
            # Базовая аналитика
            close_prices = df['close'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            volumes = df['volume'].values
            
            # 1. Тренд (простой анализ)
            recent_change = (close_prices[-1] - close_prices[0]) / close_prices[0] * 100
            
            if recent_change > 5:
                trend = "BULLISH"
            elif recent_change < -5:
                trend = "BEARISH"
            else:
                trend = "SIDEWAYS"
            
            # 2. Волатильность (ATR proxy)
            daily_ranges = (high_prices - low_prices) / close_prices
            avg_volatility = float(daily_ranges.mean())
            
            if avg_volatility > 0.05:
                volatility = "EXTREME"
            elif avg_volatility > 0.03:
                volatility = "HIGH"
            elif avg_volatility > 0.015:
                volatility = "MEDIUM"
            else:
                volatility = "LOW"
            
            # 3. Объемы
            avg_volume = float(volumes.mean())
            recent_volume = float(volumes[-24:].mean())  # Последние 24 часа
            volume_trend = "INCREASING" if recent_volume > avg_volume * 1.2 else "DECREASING" if recent_volume < avg_volume * 0.8 else "STABLE"
            
            # 4. Уровни поддержки/сопротивления (упрощенно)
            price_range = high_prices.max() - low_prices.min()
            current_price = close_prices[-1]
            support_level = float(low_prices[-48:].min())  # Минимум за 48 часов
            resistance_level = float(high_prices[-48:].max())  # Максимум за 48 часов
            
            return {
                'trend': trend,
                'volatility': volatility,
                'price_change_pct': float(recent_change),
                'current_price': float(current_price),
                'support_level': support_level,
                'resistance_level': resistance_level,
                'volume_trend': volume_trend,
                'avg_volatility': avg_volatility,
                'signals': []  # Заполнится в бэктесте
            }
            
        except Exception as e:
            logger.error(f"❌ [MARKET_CONTEXT] Error analyzing market: {e}")
            return {
                'trend': 'UNKNOWN',
                'volatility': 'MEDIUM',
                'price_change_pct': 0.0,
                'current_price': 0.0,
                'support_level': 0.0,
                'resistance_level': 0.0,
                'volume_trend': 'STABLE',
                'avg_volatility': 0.02,
                'signals': []
            }
    
    async def _run_quick_backtest(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Выполняет быстрый бэктест на недавних данных."""
        try:
            logger.info("🧪 [MARKET_CONTEXT] Running quick backtest...")
            
            # Импортируем компоненты стратегии
            from strategy.signals import SignalGenerator
            from strategy.risk import RiskManager
            
            signal_generator = SignalGenerator(self.config)
            risk_manager = RiskManager(self.config)
            
            # Симуляция торговли на данных
            trades = []
            signals = []
            current_position = None
            balance = 10000.0  # Стартовый баланс для тестирования
            initial_balance = balance
            
            for i in range(50, len(df)):  # Оставляем данные для индикаторов
                current_candles = df.iloc[:i+1]
                
                try:
                    # Генерируем сигнал (FIXED: Правильный вызов метода)
                    signal_data = signal_generator.generate_signal(
                        symbol=symbol,
                        market_data=current_candles,
                        config=self.config
                    )
                    
                    if signal_data and hasattr(signal_data, 'side') and signal_data.side:
                        signals.append({
                            'timestamp': current_candles.iloc[-1].name,
                            'side': signal_data.side,
                            'strength': getattr(signal_data, 'strength', 0.0),
                            'price': float(current_candles.iloc[-1]['close'])
                        })
                        
                        # Простая симуляция торговли
                        if not current_position:
                            # Открываем позицию
                            position_size = balance * 0.02  # 2% риска
                            current_position = {
                                'side': signal_data.side,
                                'entry_price': float(current_candles.iloc[-1]['close']),
                                'size': position_size,
                                'entry_time': current_candles.iloc[-1].name
                            }
                        else:
                            # Закрываем позицию при противоположном сигнале
                            if current_position['side'] != signal_data.side:
                                exit_price = float(current_candles.iloc[-1]['close'])
                                
                                if current_position['side'] == 'BUY':
                                    pnl = (exit_price - current_position['entry_price']) / current_position['entry_price']
                                else:
                                    pnl = (current_position['entry_price'] - exit_price) / current_position['entry_price']
                                
                                pnl_amount = pnl * current_position['size']
                                balance += pnl_amount
                                
                                trades.append({
                                    'entry_price': current_position['entry_price'],
                                    'exit_price': exit_price,
                                    'side': current_position['side'],
                                    'pnl_pct': pnl * 100,
                                    'pnl_amount': pnl_amount
                                })
                                
                                current_position = None
                
                except Exception as e:
                    logger.debug(f"Signal generation error at index {i}: {e}")
                    continue
            
            # Анализ результатов
            if trades:
                winning_trades = [t for t in trades if t['pnl_amount'] > 0]
                win_rate = len(winning_trades) / len(trades)
                total_pnl_pct = (balance - initial_balance) / initial_balance * 100
                
                # Максимальная просадка (упрощенная)
                running_balance = initial_balance
                peak_balance = initial_balance
                max_drawdown = 0.0
                
                for trade in trades:
                    running_balance += trade['pnl_amount']
                    if running_balance > peak_balance:
                        peak_balance = running_balance
                    
                    drawdown = (peak_balance - running_balance) / peak_balance
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                
                max_drawdown *= 100  # В процентах
            else:
                win_rate = 0.0
                total_pnl_pct = 0.0
                max_drawdown = 0.0
            
            logger.info(f"🎯 [BACKTEST] Trades: {len(trades)}, Win Rate: {win_rate:.1%}, PnL: {total_pnl_pct:.2f}%, Max DD: {max_drawdown:.2f}%")
            
            return {
                'total_trades': len(trades),
                'win_rate': win_rate,
                'pnl_pct': total_pnl_pct,
                'max_drawdown': max_drawdown,
                'signals': signals[:10],  # Последние 10 сигналов
                'trades': trades[-5:] if trades else []  # Последние 5 сделок
            }
            
        except Exception as e:
            logger.error(f"❌ [MARKET_CONTEXT] Backtest error: {e}")
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'pnl_pct': 0.0,
                'max_drawdown': 0.0,
                'signals': [],
                'trades': []
            }
    
    def _generate_trading_recommendations(self, market_analysis: Dict[str, Any], backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Генерирует рекомендации для торговли."""
        
        # Базовые значения
        base_confidence = getattr(self.config, 'bt_conf_min', 0.45)
        base_position_multiplier = 1.0
        
        # Корректировки на основе рыночных условий
        confidence_adjustment = 0.0
        position_adjustment = 1.0
        risk_level = "MEDIUM"
        
        # 1. Корректировки на основе волатильности
        volatility = market_analysis.get('volatility', 'MEDIUM')
        if volatility == "EXTREME":
            confidence_adjustment += 0.3  # Требуем более сильные сигналы
            position_adjustment *= 0.5    # Уменьшаем размер позиций
            risk_level = "CRITICAL"
        elif volatility == "HIGH":
            confidence_adjustment += 0.15
            position_adjustment *= 0.7
            risk_level = "HIGH"
        elif volatility == "LOW":
            confidence_adjustment -= 0.1  # Можем быть менее строгими
            position_adjustment *= 1.2
        
        # 2. Корректировки на основе тренда
        trend = market_analysis.get('trend', 'SIDEWAYS')
        if trend == "SIDEWAYS":
            confidence_adjustment += 0.1  # В боковике нужны более четкие сигналы
            position_adjustment *= 0.8
        
        # 3. Корректировки на основе результатов бэктеста
        win_rate = backtest_results.get('win_rate', 0.0)
        pnl_pct = backtest_results.get('pnl_pct', 0.0)
        max_drawdown = backtest_results.get('max_drawdown', 0.0)
        
        if win_rate < 0.4:  # Низкий винрейт
            confidence_adjustment += 0.2
            position_adjustment *= 0.6
            if risk_level not in ["HIGH", "CRITICAL"]:
                risk_level = "HIGH"
        elif win_rate > 0.7:  # Высокий винрейт
            confidence_adjustment -= 0.1
            position_adjustment *= 1.1
        
        if pnl_pct < -5:  # Убыточная стратегия на недавних данных
            confidence_adjustment += 0.25
            position_adjustment *= 0.5
            risk_level = "CRITICAL"
        
        if max_drawdown > 10:  # Высокая просадка
            confidence_adjustment += 0.15
            position_adjustment *= 0.7
            if risk_level == "MEDIUM":
                risk_level = "HIGH"
        
        # Финальные значения
        final_confidence = max(0.2, min(2.0, base_confidence + confidence_adjustment))
        final_position_multiplier = max(0.1, min(2.0, position_adjustment))
        
        return {
            'confidence_threshold': final_confidence,
            'position_multiplier': final_position_multiplier,
            'risk_level': risk_level,
            'reasoning': {
                'volatility': volatility,
                'trend': trend,
                'win_rate': win_rate,
                'pnl_pct': pnl_pct,
                'max_drawdown': max_drawdown,
                'confidence_adjustment': confidence_adjustment,
                'position_adjustment': position_adjustment
            }
        }
    
    def _is_context_fresh(self) -> bool:
        """Проверяет, свежий ли кэшированный контекст."""
        if not self.context_file.exists():
            return False
        
        try:
            with open(self.context_file, 'r') as f:
                data = json.load(f)
            
            last_update = datetime.fromisoformat(data['last_update'])
            age_hours = (datetime.now(timezone.utc) - last_update).total_seconds() / 3600
            
            return age_hours < 4  # Считаем свежим если моложе 4 часов
            
        except Exception:
            return False
    
    def _load_cached_context(self) -> MarketContextData:
        """Загружает кэшированный контекст."""
        with open(self.context_file, 'r') as f:
            data = json.load(f)
        
        # Конвертируем строки дат обратно в datetime
        data['analysis_start'] = datetime.fromisoformat(data['analysis_start'])
        data['analysis_end'] = datetime.fromisoformat(data['analysis_end'])
        data['last_update'] = datetime.fromisoformat(data['last_update'])
        
        return MarketContextData(**data)
    
    def _save_context(self, context: MarketContextData) -> None:
        """Сохраняет контекст в файл."""
        data = asdict(context)
        
        # Конвертируем datetime в строки для JSON
        data['analysis_start'] = context.analysis_start.isoformat()
        data['analysis_end'] = context.analysis_end.isoformat()
        data['last_update'] = context.last_update.isoformat()
        
        with open(self.context_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _create_default_context(self) -> MarketContextData:
        """Создает контекст по умолчанию при ошибках."""
        now = datetime.now(timezone.utc)
        return MarketContextData(
            analysis_start=now - timedelta(hours=24),
            analysis_end=now,
            last_update=now,
            
            overall_trend="UNKNOWN",
            volatility_level="MEDIUM",
            recent_signals=[],
            
            backtest_win_rate=0.0,
            backtest_total_trades=0,
            backtest_pnl_pct=0.0,
            backtest_max_drawdown=0.0,
            
            suggested_confidence_threshold=getattr(self.config, 'bt_conf_min', 0.45),
            suggested_position_size_multiplier=1.0,
            risk_warning_level="MEDIUM",
            
            active_positions=[],
            market_conditions={}
        )
    
    def _log_market_summary(self, context: MarketContextData) -> None:
        """Выводит краткое резюме рыночного анализа."""
        logger.info("=" * 60)
        logger.info("📊 MARKET CONTEXT ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"🎯 Overall Trend: {context.overall_trend}")
        logger.info(f"📈 Volatility Level: {context.volatility_level}")
        logger.info(f"⚠️  Risk Warning: {context.risk_warning_level}")
        logger.info("")
        logger.info("🧪 BACKTEST RESULTS (Last 50 Days):")
        logger.info(f"   📊 Total Trades: {context.backtest_total_trades}")
        logger.info(f"   🎯 Win Rate: {context.backtest_win_rate:.1%}")
        logger.info(f"   💰 PnL: {context.backtest_pnl_pct:+.2f}%")
        logger.info(f"   📉 Max Drawdown: {context.backtest_max_drawdown:.2f}%")
        logger.info("")
        logger.info("🎛️  TRADING RECOMMENDATIONS:")
        logger.info(f"   🎯 Confidence Threshold: {context.suggested_confidence_threshold:.3f}")
        logger.info(f"   📊 Position Size Multiplier: {context.suggested_position_size_multiplier:.2f}x")
        logger.info("=" * 60)
    
    def get_current_context(self) -> Optional[MarketContextData]:
        """Возвращает текущий контекст."""
        return self.current_context
    
    async def refresh_context_if_needed(self) -> MarketContextData:
        """Обновляет контекст при необходимости."""
        if self.current_context is None:
            return await self.initialize_market_context()
        
        # Проверяем возраст контекста
        age_hours = (datetime.now(timezone.utc) - self.current_context.last_update).total_seconds() / 3600
        
        if age_hours > 6:  # Обновляем каждые 6 часов
            logger.info("🔄 [MARKET_CONTEXT] Context is old, refreshing...")
            return await self.initialize_market_context(force_refresh=True)
        
        return self.current_context