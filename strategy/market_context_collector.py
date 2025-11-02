"""
📊 MARKET CONTEXT COLLECTOR
===========================

Собирает rich контекст рынка для ML системы:
- Технические индикаторы в реальном времени
- Рыночные условия и режимы
- Временные факторы
- Уровни поддержки/сопротивления
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from strategy.ml_learning_system import MarketContext, TradeOutcome
from data.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class MarketContextCollector:
    """Сборщик контекста рынка для ML системы"""
    
    def __init__(self):
        self.support_resistance_cache = {}
        self.volatility_history = {}
        
    async def collect_market_context(self, 
                                   symbol: str, 
                                   candles_data: pd.DataFrame,
                                   current_price: float,
                                   fear_greed_index: int = 50,
                                   btc_dominance: float = 50.0) -> MarketContext:
        """Собирает полный контекст рынка"""
        
        try:
            # Базовые данные
            now = datetime.now(timezone.utc)
            
            # Технические индикаторы
            tech_indicators = self._calculate_technical_indicators(candles_data)
            
            # Рыночные условия
            market_conditions = self._analyze_market_conditions(candles_data, current_price)
            
            # Временные факторы
            time_factors = self._get_time_factors(now)
            
            # Уровни поддержки/сопротивления
            support_resistance = self._find_support_resistance(candles_data, current_price)
            
            # Ликвидность и спреды (мок-данные, можно заменить реальными)
            liquidity_data = self._estimate_liquidity(symbol, current_price)
            
            return MarketContext(
                timestamp=now,
                symbol=symbol,
                
                # Технические индикаторы
                rsi_14=tech_indicators['rsi_14'],
                rsi_7=tech_indicators['rsi_7'],
                macd=tech_indicators['macd'],
                macd_signal=tech_indicators['macd_signal'],
                bb_position=tech_indicators['bb_position'],
                sma_20=tech_indicators['sma_20'],
                ema_50=tech_indicators['ema_50'],
                atr_14=tech_indicators['atr_14'],
                volume_ratio=tech_indicators['volume_ratio'],
                
                # Рыночные условия
                volatility_percentile=market_conditions['volatility_percentile'],
                trend_strength=market_conditions['trend_strength'],
                market_regime=market_conditions['regime'],
                fear_greed_index=fear_greed_index,
                btc_dominance=btc_dominance,
                
                # Временные факторы
                hour_of_day=time_factors['hour'],
                day_of_week=time_factors['day_of_week'],
                session=time_factors['session'],
                
                # Уровни
                support_distance=support_resistance['support_distance'],
                resistance_distance=support_resistance['resistance_distance'],
                
                # Ликвидность
                bid_ask_spread=liquidity_data['spread'],
                order_book_imbalance=liquidity_data['imbalance']
            )
            
        except Exception as e:
            logger.error(f"❌ [CONTEXT_COLLECTOR] Error collecting context for {symbol}: {e}")
            return self._get_default_context(symbol)
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Рассчитывает технические индикаторы"""
        
        try:
            closes = pd.Series(df['close'].values)
            highs = pd.Series(df['high'].values)
            lows = pd.Series(df['low'].values)
            volumes = pd.Series(df['volume'].values) if 'volume' in df.columns else pd.Series(np.ones(len(closes)))
            
            # RSI
            rsi_14_series = TechnicalIndicators.rsi(closes, 14)
            rsi_14 = rsi_14_series.iloc[-1] if len(rsi_14_series) > 0 and not pd.isna(rsi_14_series.iloc[-1]) else 50.0
            
            rsi_7_series = TechnicalIndicators.rsi(closes, 7)
            rsi_7 = rsi_7_series.iloc[-1] if len(rsi_7_series) > 0 and not pd.isna(rsi_7_series.iloc[-1]) else 50.0
            
            # MACD
            macd_line, macd_signal_line, macd_histogram = TechnicalIndicators.macd(closes)
            macd = macd_line.iloc[-1] if len(macd_line) > 0 and not pd.isna(macd_line.iloc[-1]) else 0.0
            macd_signal = macd_signal_line.iloc[-1] if len(macd_signal_line) > 0 and not pd.isna(macd_signal_line.iloc[-1]) else 0.0
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(closes, 20)
            current_price = closes.iloc[-1]
            
            bb_upper_val = bb_upper.iloc[-1] if len(bb_upper) > 0 and not pd.isna(bb_upper.iloc[-1]) else current_price * 1.02
            bb_lower_val = bb_lower.iloc[-1] if len(bb_lower) > 0 and not pd.isna(bb_lower.iloc[-1]) else current_price * 0.98
            bb_position = (current_price - bb_lower_val) / (bb_upper_val - bb_lower_val) if bb_upper_val != bb_lower_val else 0.5
            
            # Moving Averages
            sma_20 = TechnicalIndicators.sma(closes, 20).iloc[-1] if len(closes) >= 20 else current_price
            ema_50 = TechnicalIndicators.ema(closes, 50).iloc[-1] if len(closes) >= 50 else current_price
            
            # ATR
            atr_14_series = TechnicalIndicators.atr(highs, lows, closes, 14)
            atr_14 = atr_14_series.iloc[-1] if len(atr_14_series) > 0 and not pd.isna(atr_14_series.iloc[-1]) else current_price * 0.02
            
            # Volume Ratio
            avg_volume = volumes.tail(20).mean() if len(volumes) >= 20 else volumes.iloc[-1] if len(volumes) > 0 else 1.0
            current_volume = volumes.iloc[-1] if len(volumes) > 0 else 1.0
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            return {
                'rsi_14': float(rsi_14),
                'rsi_7': float(rsi_7),
                'macd': float(macd),
                'macd_signal': float(macd_signal),
                'bb_position': float(bb_position),
                'sma_20': float(sma_20),
                'ema_50': float(ema_50),
                'atr_14': float(atr_14),
                'volume_ratio': float(volume_ratio)
            }
            
        except Exception as e:
            logger.error(f"❌ [TECH_INDICATORS] Error: {e}")
            return self._get_default_tech_indicators()
    
    def _analyze_market_conditions(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Анализирует рыночные условия"""
        
        try:
            closes = df['close'].values
            
            # Волатильность
            returns = np.diff(np.log(closes))
            current_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
            
            # Историческая волатильность для процентиля
            historical_volatility = []
            for i in range(20, len(returns), 5):  # Каждые 5 дней
                vol = np.std(returns[i-20:i])
                historical_volatility.append(vol)
            
            if len(historical_volatility) > 0:
                volatility_percentile = (np.sum(np.array(historical_volatility) < current_volatility) / 
                                       len(historical_volatility)) * 100
            else:
                volatility_percentile = 50.0
            
            # Сила тренда (на основе направленного движения)
            price_changes = np.diff(closes[-50:]) if len(closes) >= 50 else np.diff(closes)
            positive_moves = np.sum(price_changes > 0)
            total_moves = len(price_changes)
            trend_strength = abs(positive_moves / total_moves - 0.5) * 2 if total_moves > 0 else 0.0
            
            # Режим рынка
            if volatility_percentile > 80:
                regime = "volatile"
            elif trend_strength > 0.6:
                regime = "trending"
            else:
                regime = "ranging"
            
            return {
                'volatility_percentile': float(volatility_percentile),
                'trend_strength': float(trend_strength),
                'regime': regime
            }
            
        except Exception as e:
            logger.error(f"❌ [MARKET_CONDITIONS] Error: {e}")
            return {
                'volatility_percentile': 50.0,
                'trend_strength': 0.5,
                'regime': 'ranging'
            }
    
    def _get_time_factors(self, timestamp: datetime) -> Dict[str, Any]:
        """Получает временные факторы"""
        
        hour = timestamp.hour
        day_of_week = timestamp.weekday()  # 0 = Monday
        
        # Определяем торговую сессию
        if 0 <= hour < 8:
            session = "asian"
        elif 8 <= hour < 16:
            session = "european"
        else:
            session = "american"
        
        return {
            'hour': hour,
            'day_of_week': day_of_week,
            'session': session
        }
    
    def _find_support_resistance(self, df: pd.DataFrame, current_price: float) -> Dict[str, float]:
        """Находит уровни поддержки и сопротивления"""
        
        try:
            highs = df['high'].values[-100:]  # Последние 100 свечей
            lows = df['low'].values[-100:]
            
            # Простой алгоритм: локальные максимумы и минимумы
            support_levels = []
            resistance_levels = []
            
            # Находим локальные минимумы (поддержки)
            for i in range(2, len(lows) - 2):
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    support_levels.append(lows[i])
            
            # Находим локальные максимумы (сопротивления)
            for i in range(2, len(highs) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    resistance_levels.append(highs[i])
            
            # Ближайшие уровни
            support_levels = [s for s in support_levels if s < current_price]
            resistance_levels = [r for r in resistance_levels if r > current_price]
            
            nearest_support = max(support_levels) if support_levels else current_price * 0.95
            nearest_resistance = min(resistance_levels) if resistance_levels else current_price * 1.05
            
            support_distance = abs(current_price - nearest_support) / current_price * 100
            resistance_distance = abs(nearest_resistance - current_price) / current_price * 100
            
            return {
                'support_distance': float(support_distance),
                'resistance_distance': float(resistance_distance)
            }
            
        except Exception as e:
            logger.error(f"❌ [SUPPORT_RESISTANCE] Error: {e}")
            return {
                'support_distance': 2.0,
                'resistance_distance': 2.0
            }
    
    def _estimate_liquidity(self, symbol: str, current_price: float) -> Dict[str, float]:
        """Оценивает ликвидность (мок-данные)"""
        
        # В реальной системе здесь был бы запрос к order book
        base_spread = {
            'BTCUSDT': 0.01,
            'ETHUSDT': 0.02,
            'ADAUSDT': 0.05
        }.get(symbol, 0.03)
        
        return {
            'spread': base_spread,
            'imbalance': np.random.uniform(-0.1, 0.1)  # Случайный дисбаланс
        }
    

    
    def _get_default_context(self, symbol: str) -> MarketContext:
        """Возвращает контекст по умолчанию"""
        now = datetime.now(timezone.utc)
        
        return MarketContext(
            timestamp=now,
            symbol=symbol,
            rsi_14=50.0,
            rsi_7=50.0,
            macd=0.0,
            macd_signal=0.0,
            bb_position=0.5,
            sma_20=100.0,
            ema_50=100.0,
            atr_14=2.0,
            volume_ratio=1.0,
            volatility_percentile=50.0,
            trend_strength=0.5,
            market_regime="ranging",
            fear_greed_index=50,
            btc_dominance=50.0,
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            session="european",
            support_distance=2.0,
            resistance_distance=2.0,
            bid_ask_spread=0.03,
            order_book_imbalance=0.0
        )
    
    def _get_default_tech_indicators(self) -> Dict[str, float]:
        """Возвращает технические индикаторы по умолчанию"""
        return {
            'rsi_14': 50.0,
            'rsi_7': 50.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'bb_position': 0.5,
            'sma_20': 100.0,
            'ema_50': 100.0,
            'atr_14': 2.0,
            'volume_ratio': 1.0
        }
    
    def create_trade_outcome(self, 
                           trade_record: Any,
                           market_context: MarketContext,
                           exit_price: float,
                           exit_reason: str) -> TradeOutcome:
        """Создает TradeOutcome из завершенной сделки"""
        
        try:
            hold_time_minutes = trade_record.hold_time_seconds / 60.0
            
            # Рассчитываем дополнительные метрики
            sharpe_ratio = self._calculate_trade_sharpe(trade_record.pnl_pct, hold_time_minutes)
            
            # MFE/MAE (упрощенная версия)
            mfe = max(0, trade_record.pnl_pct)  # Максимальная прибыль
            mae = min(0, trade_record.pnl_pct)  # Максимальный убыток
            
            # Вероятность успеха (на основе исторических данных)
            win_probability = 0.6 if trade_record.pnl > 0 else 0.4
            
            return TradeOutcome(
                trade_id=trade_record.trade_id or "unknown",
                pnl=trade_record.pnl,
                pnl_pct=trade_record.pnl_pct,
                hold_time_minutes=hold_time_minutes,
                exit_reason=exit_reason,
                sharpe_ratio=sharpe_ratio,
                max_favorable_excursion=mfe,
                max_adverse_excursion=abs(mae),
                win_probability=win_probability,
                stress_level=self._calculate_stress_level(trade_record),
                confidence_decay=self._calculate_confidence_decay(hold_time_minutes)
            )
            
        except Exception as e:
            logger.error(f"❌ [TRADE_OUTCOME] Error creating outcome: {e}")
            return TradeOutcome(
                trade_id="error",
                pnl=0.0,
                pnl_pct=0.0,
                hold_time_minutes=30.0,
                exit_reason="error",
                sharpe_ratio=0.0,
                max_favorable_excursion=0.0,
                max_adverse_excursion=0.0,
                win_probability=0.5,
                stress_level=0.5,
                confidence_decay=0.1
            )
    
    def _calculate_trade_sharpe(self, pnl_pct: float, hold_time_minutes: float) -> float:
        """Рассчитывает Sharpe ratio для сделки"""
        if hold_time_minutes <= 0:
            return 0.0
        
        # Нормализуем к дневной доходности
        daily_return = pnl_pct * (24 * 60) / hold_time_minutes
        
        # Упрощенный Sharpe (предполагаем волатильность 2% в день)
        assumed_daily_vol = 2.0
        
        return daily_return / assumed_daily_vol if assumed_daily_vol > 0 else 0.0
    
    def _calculate_stress_level(self, trade_record: Any) -> float:
        """Рассчитывает уровень стресса сделки"""
        # Основан на размере позиции относительно капитала
        notional = trade_record.entry_price * trade_record.quantity
        
        # Предполагаем капитал $1000 (в реальности нужно передавать)
        capital = 1000.0
        exposure = notional / capital
        
        return min(1.0, exposure * 2)  # Нормализуем
    
    def _calculate_confidence_decay(self, hold_time_minutes: float) -> float:
        """Рассчитывает decay уверенности со временем"""
        # Чем дольше держим, тем больше decay
        decay_rate = 0.001  # 0.1% в минуту
        return min(0.5, hold_time_minutes * decay_rate)