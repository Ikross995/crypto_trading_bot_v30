#!/usr/bin/env python3
"""
Learning Visualizer - Система визуализации обучения торгового бота

🧠 ВОЗМОЖНОСТИ:
- Реальное время отслеживания адаптации параметров
- Визуализация метрик производительности
- Dashboard с графиками обучения
- Детальное логирование изменений
- Экспорт отчетов об обучении
"""

import asyncio
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from loguru import logger

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    logger.info("📊 [VISUALIZER] Matplotlib/Seaborn available for plotting")
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("📊 [VISUALIZER] Install matplotlib/seaborn for advanced visualization: pip install matplotlib seaborn")


@dataclass
class LearningSnapshot:
    """Расширенный снимок состояния обучения на определенный момент времени."""
    timestamp: datetime
    iteration: int
    
    # Текущие параметры
    confidence_threshold: float
    position_size_multiplier: float
    dca_enabled: bool
    
    # Базовые метрики производительности
    total_trades: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    
    # Недавние изменения
    recent_adaptations: List[Dict]
    ai_recommendations: Dict
    
    # Статистика обучения
    adaptations_count: int
    last_adaptation_trigger: str
    learning_confidence: float
    
    # Расширенные метрики производительности
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_hold_time: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    
    # Метрики риска
    risk_reward_ratio: float = 0.0
    kelly_criterion: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Реальные данные аккаунта
    account_balance: float = 0.0
    unrealized_pnl: float = 0.0
    total_wallet_balance: float = 0.0
    available_balance: float = 0.0
    margin_used: float = 0.0
    margin_ratio: float = 0.0
    
    # Данные о позициях
    open_positions: int = 0
    total_position_value: float = 0.0
    largest_position: float = 0.0
    
    # Рыночные данные
    market_volatility: float = 0.0
    market_trend: str = "neutral"
    price_change_24h: float = 0.0
    volume_24h: float = 0.0
    volume_profile: Dict[str, float] = None
    
    # Технические индикаторы
    rsi_current: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    
    # Метрики торговой стратегии
    signal_quality: float = 0.0
    signal_frequency: float = 0.0
    false_signals: int = 0
    missed_opportunities: int = 0
    
    # Адаптивные параметры
    volatility_adjustment: float = 1.0
    trend_factor: float = 1.0
    fear_greed_index: float = 50.0
    btc_dominance: float = 50.0
    
    def __post_init__(self):
        if self.volume_profile is None:
            self.volume_profile = {}


@dataclass
class AdaptationEvent:
    """Событие адаптации параметров."""
    timestamp: datetime
    trigger: str
    old_params: Dict[str, Any]
    new_params: Dict[str, Any]
    performance_before: Dict[str, float]
    performance_after: Optional[Dict[str, float]]
    reasoning: List[str]
    confidence: float


class LearningVisualizer:
    """
    Система визуализации и мониторинга обучения торгового бота.
    
    🎯 ФУНКЦИИ:
    1. Реальное время трекинг изменений параметров
    2. Визуализация метрик производительности  
    3. Генерация отчетов об обучении
    4. Dashboard для мониторинга AI
    """
    
    def __init__(self, output_dir: str = "data/learning_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # История обучения
        self.learning_history: List[LearningSnapshot] = []
        self.adaptation_events: List[AdaptationEvent] = []
        
        # Настройки визуализации
        self.update_interval = 60  # Обновление каждые 60 секунд
        self.max_history_points = 1000  # Максимум точек в истории
        
        # Файлы для сохранения
        self.history_file = self.output_dir / "learning_history.json"
        self.adaptations_file = self.output_dir / "adaptations.json"
        self.dashboard_file = self.output_dir / "learning_dashboard.html"
        
        logger.info("📊 [VISUALIZER] Learning visualizer initialized")
        logger.info(f"📁 [VISUALIZER] Output directory: {self.output_dir}")
    
    async def _get_real_account_data(self, trading_engine) -> Dict[str, Any]:
        """Получает реальные данные аккаунта из Binance."""
        try:
            if not hasattr(trading_engine, 'client') or not trading_engine.client:
                return {}
            
            # Информация об аккаунте
            account_info = trading_engine.client.get_account()
            
            # Балансы
            total_wallet_balance = float(account_info.get('totalWalletBalance', 0))
            available_balance = float(account_info.get('availableBalance', 0))
            total_unrealized_pnl = float(account_info.get('totalUnrealizedProfit', 0))
            total_margin_balance = float(account_info.get('totalMarginBalance', 0))
            total_position_initial_margin = float(account_info.get('totalPositionInitialMargin', 0))
            
            # Позиции
            positions = trading_engine.client.get_positions()
            active_positions = [pos for pos in positions if float(pos.get('positionAmt', 0)) != 0]
            
            open_positions_count = len(active_positions)
            total_position_value = sum(abs(float(pos.get('notional', 0))) for pos in active_positions)
            largest_position = max([abs(float(pos.get('notional', 0))) for pos in active_positions], default=0.0)
            
            # Margin ratio
            margin_ratio = 0.0
            if total_margin_balance > 0:
                margin_ratio = total_position_initial_margin / total_margin_balance
            
            return {
                'account_balance': total_margin_balance,
                'total_wallet_balance': total_wallet_balance,
                'available_balance': available_balance,
                'unrealized_pnl': total_unrealized_pnl,
                'margin_used': total_position_initial_margin,
                'margin_ratio': margin_ratio,
                'open_positions': open_positions_count,
                'total_position_value': total_position_value,
                'largest_position': largest_position
            }
            
        except Exception as e:
            logger.debug(f"[VISUALIZER] Failed to get account data: {e}")
            return {}
    
    async def _get_market_data(self, trading_engine) -> Dict[str, Any]:
        """Получает рыночные данные."""
        try:
            # Получаем данные по основной паре (BTCUSDT)
            symbol = "BTCUSDT"
            
            # 24h статистика
            ticker_24h = trading_engine.client.get_24hr_ticker(symbol=symbol)
            price_change_24h = float(ticker_24h.get('priceChangePercent', 0))
            
            # Волатильность (примерная через 24h изменение)
            volatility = abs(price_change_24h) / 100  # Нормализуем
            
            # Определяем тренд
            if price_change_24h > 2:
                trend = "bullish"
            elif price_change_24h < -2:
                trend = "bearish"
            else:
                trend = "neutral"
            
            return {
                'market_volatility': volatility,
                'market_trend': trend,
                'price_change_24h': price_change_24h,
                'volume_24h': float(ticker_24h.get('volume', 0))
            }
            
        except Exception as e:
            logger.debug(f"[VISUALIZER] Failed to get market data: {e}")
            return {
                'market_volatility': 0.0,
                'market_trend': 'neutral',
                'price_change_24h': 0.0,
                'volume_24h': 0.0
            }
    
    async def _calculate_extended_metrics(self, adaptive_learning_system) -> Dict[str, Any]:
        """Рассчитывает расширенные метрики производительности."""
        try:
            if not hasattr(adaptive_learning_system, 'trades_history') or not adaptive_learning_system.trades_history:
                return {}
            
            trades = adaptive_learning_system.trades_history
            
            # Базовые расчеты
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
            
            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            
            # Risk-reward ratio
            risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
            
            # Kelly criterion (упрощенный)
            win_rate = len(winning_trades) / len(trades) if trades else 0.0
            kelly_criterion = 0.0
            if avg_loss != 0 and win_rate > 0:
                b = abs(avg_win / avg_loss)  # odds received on the wager
                p = win_rate  # probability of winning
                kelly_criterion = (b * p - (1 - p)) / b
            
            # Sortino ratio (упрощенный)
            returns = [t.pnl_pct for t in trades]
            negative_returns = [r for r in returns if r < 0]
            
            sortino_ratio = 0.0
            if returns and negative_returns:
                avg_return = np.mean(returns)
                downside_deviation = np.std(negative_returns)
                if downside_deviation != 0:
                    sortino_ratio = avg_return / downside_deviation
            
            # Среднее время удержания
            avg_hold_time = np.mean([t.hold_time_seconds for t in trades]) if trades else 0.0
            
            return {
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_hold_time': avg_hold_time,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'risk_reward_ratio': risk_reward_ratio,
                'kelly_criterion': kelly_criterion,
                'sortino_ratio': sortino_ratio
            }
            
        except Exception as e:
            logger.debug(f"[VISUALIZER] Failed to calculate extended metrics: {e}")
            return {}

    async def capture_learning_snapshot(self, 
                                      adaptive_learning_system,
                                      trading_engine,
                                      iteration: int) -> LearningSnapshot:
        """Создает снимок текущего состояния обучения."""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Получаем реальные данные аккаунта
            account_data = await self._get_real_account_data(trading_engine)
            market_data = await self._get_market_data(trading_engine)
            extended_metrics = await self._calculate_extended_metrics(adaptive_learning_system)
            
            # Получаем текущие параметры
            current_metrics = getattr(adaptive_learning_system, 'current_metrics', None)
            if not current_metrics:
                # Создаем базовые метрики если их нет
                confidence_threshold = getattr(trading_engine.config, 'bt_conf_min', 0.45)
                position_size_multiplier = 1.0
                dca_enabled = True
                total_trades = 0
                win_rate = 0.0
                profit_factor = 1.0
                # Используем РЕАЛЬНЫЕ данные PnL из аккаунта, если доступны
                total_pnl = account_data.get('unrealized_pnl', 0.0) + extended_metrics.get('gross_profit', 0.0) - extended_metrics.get('gross_loss', 0.0)
                max_drawdown = 0.0
                sharpe_ratio = 0.0
            else:
                confidence_threshold = current_metrics.confidence_threshold
                position_size_multiplier = current_metrics.position_size_multiplier
                dca_enabled = current_metrics.dca_enabled
                total_trades = current_metrics.total_trades
                win_rate = current_metrics.win_rate
                profit_factor = current_metrics.profit_factor
                # Комбинируем исторический PnL с реальным unrealized PnL
                historical_pnl = current_metrics.total_pnl
                current_unrealized = account_data.get('unrealized_pnl', 0.0)
                total_pnl = historical_pnl + current_unrealized
                max_drawdown = current_metrics.max_drawdown
                sharpe_ratio = current_metrics.sharpe_ratio
            
            # Получаем недавние адаптации
            adaptations_log = getattr(adaptive_learning_system, 'adaptations_log', [])
            recent_adaptations = adaptations_log[-5:] if adaptations_log else []
            
            # AI рекомендации (если доступны)
            ai_recommendations = {}
            if hasattr(adaptive_learning_system, 'advanced_ai') and adaptive_learning_system.advanced_ai:
                try:
                    market_data_for_ai = {
                        'volatility': market_data.get('market_volatility', 1.0),
                        'trend_strength': 1.0 if market_data.get('market_trend') == 'bullish' else -1.0 if market_data.get('market_trend') == 'bearish' else 0.0,
                        'volume_trend': 1.0,
                        'price_change_24h': market_data.get('price_change_24h', 0.0)
                    }
                    ai_recommendations = await adaptive_learning_system.get_advanced_ai_recommendations(market_data_for_ai) or {}
                except Exception as ai_e:
                    logger.debug(f"[VISUALIZER] AI recommendations error: {ai_e}")
            
            # Получаем дополнительные метрики стратегии
            strategy_metrics = {}
            if hasattr(trading_engine, 'signaler'):
                try:
                    # Качество сигналов (примерное)
                    signal_quality = win_rate if total_trades > 0 else 0.5
                    signal_frequency = total_trades / max(1, iteration // 60)  # сигналов в минуту
                    strategy_metrics = {
                        'signal_quality': signal_quality,
                        'signal_frequency': signal_frequency,
                        'false_signals': max(0, total_trades - extended_metrics.get('winning_trades', 0) - extended_metrics.get('losing_trades', 0)),
                        'missed_opportunities': 0  # TODO: можно добавить логику
                    }
                except:
                    pass
            
            # Получаем адаптивные параметры
            adaptive_params = {}
            if hasattr(trading_engine, 'config'):
                try:
                    adaptive_params = {
                        'volatility_adjustment': getattr(trading_engine.config, 'volatility_factor', 1.0),
                        'trend_factor': 1.0,  # TODO: добавить реальную логику
                        'fear_greed_index': 50.0,  # TODO: интегрировать API
                        'btc_dominance': 50.0  # TODO: интегрировать API
                    }
                except:
                    pass
            
            # Создаем расширенный снимок
            snapshot = LearningSnapshot(
                timestamp=current_time,
                iteration=iteration,
                confidence_threshold=confidence_threshold,
                position_size_multiplier=position_size_multiplier,
                dca_enabled=dca_enabled,
                total_trades=total_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_pnl=total_pnl,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                recent_adaptations=recent_adaptations,
                ai_recommendations=ai_recommendations,
                adaptations_count=len(adaptations_log),
                last_adaptation_trigger=adaptations_log[-1].get('trigger', 'none') if adaptations_log else 'none',
                learning_confidence=ai_recommendations.get('confidence', 0.0),
                
                # Расширенные метрики
                **extended_metrics,
                
                # Реальные данные аккаунта
                **account_data,
                
                # Рыночные данные
                **market_data,
                
                # Метрики стратегии
                **strategy_metrics,
                
                # Адаптивные параметры
                **adaptive_params
            )
            
            # Добавляем в историю
            self.learning_history.append(snapshot)
            
            # Ограничиваем размер истории
            if len(self.learning_history) > self.max_history_points:
                self.learning_history = self.learning_history[-self.max_history_points:]
            
            # Сохраняем в файл
            await self._save_history()
            
            return snapshot
            
        except Exception as e:
            logger.error(f"❌ [VISUALIZER] Failed to capture learning snapshot: {e}")
            # Возвращаем базовый снимок
            return LearningSnapshot(
                timestamp=datetime.now(timezone.utc),
                iteration=iteration,
                confidence_threshold=0.45,
                position_size_multiplier=1.0,
                dca_enabled=True,
                total_trades=0,
                win_rate=0.0,
                profit_factor=1.0,
                total_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                recent_adaptations=[],
                ai_recommendations={},
                adaptations_count=0,
                last_adaptation_trigger='none',
                learning_confidence=0.0
            )
    
    async def log_adaptation_event(self, 
                                 trigger: str,
                                 old_params: Dict[str, Any],
                                 new_params: Dict[str, Any],
                                 performance_metrics: Dict[str, float],
                                 reasoning: List[str] = None,
                                 confidence: float = 0.0) -> None:
        """Логирует событие адаптации параметров."""
        try:
            event = AdaptationEvent(
                timestamp=datetime.now(timezone.utc),
                trigger=trigger,
                old_params=old_params.copy(),
                new_params=new_params.copy(),
                performance_before=performance_metrics.copy(),
                performance_after=None,  # Будет заполнено позже
                reasoning=reasoning or [],
                confidence=confidence
            )
            
            self.adaptation_events.append(event)
            
            # Детальное логирование
            logger.info("🔧 [ADAPTATION_EVENT] ============================================")
            logger.info(f"🕐 Timestamp: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            logger.info(f"🎯 Trigger: {trigger}")
            logger.info(f"📊 Confidence: {confidence:.2f}")
            
            logger.info("📈 PARAMETER CHANGES:")
            for param, old_val in old_params.items():
                new_val = new_params.get(param, old_val)
                if old_val != new_val:
                    change_pct = ((new_val - old_val) / old_val * 100) if isinstance(old_val, (int, float)) and old_val != 0 else 0
                    logger.info(f"   📊 {param}: {old_val} → {new_val} ({change_pct:+.1f}%)")
                else:
                    logger.info(f"   📊 {param}: {old_val} (unchanged)")
            
            logger.info("📈 PERFORMANCE METRICS:")
            for metric, value in performance_metrics.items():
                if isinstance(value, float):
                    logger.info(f"   📊 {metric}: {value:.3f}")
                else:
                    logger.info(f"   📊 {metric}: {value}")
            
            if reasoning:
                logger.info("🧠 REASONING:")
                for reason in reasoning:
                    logger.info(f"   💡 {reason}")
            
            logger.info("🔧 [ADAPTATION_EVENT] ============================================")
            
            # Сохраняем в файл
            await self._save_adaptations()
            
        except Exception as e:
            logger.error(f"❌ [VISUALIZER] Failed to log adaptation event: {e}")
    
    async def generate_real_time_report(self, snapshot: LearningSnapshot) -> str:
        """Генерирует отчет о текущем состоянии обучения в реальном времени."""
        try:
            report_lines = []
            
            report_lines.append("🧠 ========== LEARNING STATUS REPORT ==========")
            report_lines.append(f"🕐 Time: {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            report_lines.append(f"🔄 Iteration: {snapshot.iteration}")
            report_lines.append("")
            
            # Текущие параметры
            report_lines.append("📊 CURRENT PARAMETERS:")
            report_lines.append(f"   🎯 Confidence Threshold: {snapshot.confidence_threshold:.3f}")
            report_lines.append(f"   📏 Position Size Multiplier: {snapshot.position_size_multiplier:.2f}x")
            report_lines.append(f"   🔄 DCA Enabled: {'✅' if snapshot.dca_enabled else '❌'}")
            report_lines.append("")
            
            # Метрики производительности
            report_lines.append("📈 PERFORMANCE METRICS:")
            report_lines.append(f"   🎲 Total Trades: {snapshot.total_trades}")
            report_lines.append(f"   🏆 Win Rate: {snapshot.win_rate:.1%}")
            report_lines.append(f"   💰 Profit Factor: {snapshot.profit_factor:.2f}")
            report_lines.append(f"   💸 Total PnL: ${snapshot.total_pnl:.2f}")
            report_lines.append(f"   📉 Max Drawdown: {snapshot.max_drawdown:.1%}")
            report_lines.append(f"   📊 Sharpe Ratio: {snapshot.sharpe_ratio:.2f}")
            report_lines.append("")
            
            # Обучение
            report_lines.append("🧠 LEARNING STATUS:")
            report_lines.append(f"   🔧 Total Adaptations: {snapshot.adaptations_count}")
            report_lines.append(f"   🎯 Last Trigger: {snapshot.last_adaptation_trigger}")
            report_lines.append(f"   🤖 AI Confidence: {snapshot.learning_confidence:.2f}")
            report_lines.append("")
            
            # AI рекомендации
            if snapshot.ai_recommendations:
                report_lines.append("🤖 AI RECOMMENDATIONS:")
                for param, value in snapshot.ai_recommendations.get('recommendations', {}).items():
                    report_lines.append(f"   🎯 {param}: {value}")
                
                reasoning = snapshot.ai_recommendations.get('reasoning', [])
                if reasoning:
                    report_lines.append("   💡 Reasoning:")
                    for reason in reasoning:
                        report_lines.append(f"     • {reason}")
                report_lines.append("")
            
            # Недавние адаптации
            if snapshot.recent_adaptations:
                report_lines.append("🔧 RECENT ADAPTATIONS:")
                for adaptation in snapshot.recent_adaptations[-3:]:  # Последние 3
                    timestamp = adaptation.get('timestamp', 'unknown')
                    trigger = adaptation.get('trigger', 'unknown')
                    report_lines.append(f"   📅 {timestamp[:19]} - {trigger}")
                report_lines.append("")
            
            # Прогресс обучения
            if len(self.learning_history) >= 2:
                prev_snapshot = self.learning_history[-2]
                report_lines.append("📈 LEARNING PROGRESS (vs previous):")
                
                # Сравнение параметров
                conf_change = snapshot.confidence_threshold - prev_snapshot.confidence_threshold
                size_change = snapshot.position_size_multiplier - prev_snapshot.position_size_multiplier
                
                if conf_change != 0:
                    report_lines.append(f"   🎯 Confidence: {conf_change:+.3f}")
                if size_change != 0:
                    report_lines.append(f"   📏 Position Size: {size_change:+.2f}x")
                
                # Сравнение производительности
                wr_change = snapshot.win_rate - prev_snapshot.win_rate
                pf_change = snapshot.profit_factor - prev_snapshot.profit_factor
                
                if wr_change != 0:
                    report_lines.append(f"   🏆 Win Rate: {wr_change:+.1%}")
                if pf_change != 0:
                    report_lines.append(f"   💰 Profit Factor: {pf_change:+.2f}")
                
                report_lines.append("")
            
            report_lines.append("🧠 =============================================")
            
            report = "\n".join(report_lines)
            
            # Логируем отчет
            logger.info(report)
            
            # Сохраняем отчет в файл
            report_file = self.output_dir / f"real_time_report_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ [VISUALIZER] Failed to generate real-time report: {e}")
            return "Error generating report"
    
    async def _save_history(self) -> None:
        """Сохраняет историю обучения в JSON файл."""
        try:
            history_data = []
            for snapshot in self.learning_history:
                data = asdict(snapshot)
                data['timestamp'] = snapshot.timestamp.isoformat()
                history_data.append(data)
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"❌ [VISUALIZER] Failed to save history: {e}")
    
    async def _save_adaptations(self) -> None:
        """Сохраняет события адаптации в JSON файл."""
        try:
            adaptations_data = []
            for event in self.adaptation_events:
                data = asdict(event)
                data['timestamp'] = event.timestamp.isoformat()
                adaptations_data.append(data)
            
            with open(self.adaptations_file, 'w', encoding='utf-8') as f:
                json.dump(adaptations_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"❌ [VISUALIZER] Failed to save adaptations: {e}")
    
    async def create_learning_dashboard(self) -> str:
        """Создает HTML dashboard для мониторинга обучения."""
        try:
            if not self.learning_history:
                logger.info("📊 [DASHBOARD] No learning history - creating initial dashboard with current state")
                # Create minimal snapshot for initial dashboard
                from datetime import datetime, timezone
                initial_snapshot = LearningSnapshot(
                    timestamp=datetime.now(timezone.utc),
                    iteration=0,
                    confidence_threshold=1.2,
                    position_size_multiplier=1.0,
                    dca_enabled=True,
                    total_trades=0,
                    win_rate=0.0,
                    profit_factor=1.0,
                    total_pnl=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    recent_adaptations=[],
                    ai_recommendations={},
                    adaptations_count=0,
                    last_adaptation_trigger='none',
                    learning_confidence=0.0,
                    # Enhanced account data
                    account_balance=1000.0,
                    unrealized_pnl=0.0,
                    margin_used=0.0,
                    total_wallet_balance=1000.0,
                    available_balance=1000.0,
                    margin_ratio=0.0,
                    open_positions=0,
                    total_position_value=0.0,
                    largest_position=0.0
                )
                self.learning_history.append(initial_snapshot)
            
            # Создаем HTML dashboard
            html_content = self._generate_dashboard_html()
            
            # Сохраняем в файл
            with open(self.dashboard_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"📊 [DASHBOARD] Created learning dashboard: {self.dashboard_file}")
            return str(self.dashboard_file)
            
        except Exception as e:
            logger.error(f"❌ [DASHBOARD] Failed to create dashboard: {e}")
            return ""
    
    def _generate_dashboard_html(self) -> str:
        """Генерирует HTML код для dashboard."""
        if not self.learning_history:
            return "<html><body><h1>No learning data available</h1></body></html>"
        
        latest = self.learning_history[-1]
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Trading Bot - Learning Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; margin-bottom: 10px; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .adaptation-log {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .adaptation-item {{ border-left: 4px solid #667eea; padding-left: 15px; margin-bottom: 15px; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .status-indicator {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }}
        .status-good {{ background-color: #28a745; }}
        .status-warning {{ background-color: #ffc107; }}
        .status-danger {{ background-color: #dc3545; }}
        .learning-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .live-indicators {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .indicator {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .progress-bar {{ background-color: #e9ecef; border-radius: 5px; height: 20px; flex-grow: 1; margin-left: 10px; }}
        .progress-fill {{ height: 100%; border-radius: 5px; transition: width 0.3s ease; }}
        .update-info {{ text-align: center; color: #666; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 AI Trading Bot - Learning Dashboard</h1>
        <p>Real-time monitoring of bot adaptation and performance</p>
        <p>Last Update: {latest.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')} | Iteration: {latest.iteration}</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">🎯 Confidence Threshold</div>
            <div class="metric-value">{latest.confidence_threshold:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">📏 Position Size Multiplier</div>
            <div class="metric-value">{latest.position_size_multiplier:.2f}x</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🏆 Win Rate</div>
            <div class="metric-value {'positive' if latest.win_rate > 0.5 else 'negative'}">{latest.win_rate:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">💰 Total PnL (REAL)</div>
            <div class="metric-value {'positive' if latest.total_pnl > 0 else 'negative'}">${latest.total_pnl:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🎲 Total Trades</div>
            <div class="metric-value">{latest.total_trades}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🔧 Adaptations</div>
            <div class="metric-value">{latest.adaptations_count}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">💼 Account Balance</div>
            <div class="metric-value">${latest.account_balance:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">💸 Unrealized PnL</div>
            <div class="metric-value {'positive' if latest.unrealized_pnl > 0 else 'negative'}">${latest.unrealized_pnl:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">📊 Profit Factor</div>
            <div class="metric-value {'positive' if latest.profit_factor > 1 else 'negative'}">{latest.profit_factor:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🎯 Open Positions</div>
            <div class="metric-value">{latest.open_positions}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">💰 Avg Win</div>
            <div class="metric-value positive">${latest.avg_win:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">💸 Avg Loss</div>
            <div class="metric-value negative">${latest.avg_loss:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">⚖️ Risk/Reward</div>
            <div class="metric-value">{latest.risk_reward_ratio:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🎰 Kelly Criterion</div>
            <div class="metric-value">{latest.kelly_criterion:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">📈 Market Trend</div>
            <div class="metric-value">{latest.market_trend.upper()}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🌊 Volatility</div>
            <div class="metric-value">{latest.market_volatility:.3f}</div>
        </div>
    </div>
    
    <div class="learning-stats">
        <div class="live-indicators">
            <h3>📊 Live Learning Indicators</h3>
            <div class="indicator">
                <span>🤖 AI Confidence:</span>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {latest.learning_confidence*100:.0f}%; background-color: #667eea;"></div>
                </div>
                <span>{latest.learning_confidence:.2f}</span>
            </div>
            <div class="indicator">
                <span>📈 Profit Factor:</span>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {min(100, latest.profit_factor*50):.0f}%; background-color: {'#28a745' if latest.profit_factor > 1 else '#dc3545'};"></div>
                </div>
                <span>{latest.profit_factor:.2f}</span>
            </div>
            <div class="indicator">
                <span>🛡️ DCA Status:</span>
                <span class="status-indicator {'status-good' if latest.dca_enabled else 'status-danger'}"></span>
                <span>{'✅ Enabled' if latest.dca_enabled else '❌ Disabled'}</span>
            </div>
            <div class="indicator">
                <span>🎯 Last Trigger:</span>
                <span>{latest.last_adaptation_trigger}</span>
            </div>
        </div>
    </div>
    
    <div class="chart-container">
        <h3>📈 Parameter Evolution</h3>
        <div id="parameter-chart" style="height: 400px;"></div>
    </div>
    
    <div class="chart-container">
        <h3>📊 Performance Metrics</h3>
        <div id="performance-chart" style="height: 400px;"></div>
    </div>
    
    <div class="adaptation-log">
        <h3>🔧 Recent Adaptations</h3>
        {self._generate_adaptations_html()}
    </div>
    
    <div class="update-info">
        <p>🔄 Dashboard auto-refreshes every 60 seconds</p>
        <p>📁 Reports saved to: {self.output_dir}</p>
    </div>
    
    <script>
        // Parameter evolution chart
        {self._generate_parameter_chart_js()}
        
        // Performance metrics chart
        {self._generate_performance_chart_js()}
        
        // Auto-refresh every 60 seconds
        setTimeout(function() {{ location.reload(); }}, 60000);
    </script>
</body>
</html>
"""
        return html
    
    def _generate_adaptations_html(self) -> str:
        """Генерирует HTML для списка адаптаций."""
        if not self.adaptation_events:
            return "<p>No adaptations recorded yet.</p>"
        
        html_items = []
        for event in self.adaptation_events[-10:]:  # Последние 10 адаптаций
            changes = []
            for param, old_val in event.old_params.items():
                new_val = event.new_params.get(param, old_val)
                if old_val != new_val:
                    changes.append(f"{param}: {old_val} → {new_val}")
            
            changes_str = ", ".join(changes) if changes else "No changes"
            
            html_items.append(f"""
                <div class="adaptation-item">
                    <div class="timestamp">{event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</div>
                    <div><strong>Trigger:</strong> {event.trigger}</div>
                    <div><strong>Changes:</strong> {changes_str}</div>
                    <div><strong>Confidence:</strong> {event.confidence:.2f}</div>
                </div>
            """)
        
        return "\n".join(reversed(html_items))  # Новые сверху
    
    def _generate_parameter_chart_js(self) -> str:
        """Генерирует JavaScript для графика параметров."""
        if len(self.learning_history) < 2:
            return "// Not enough data for parameter chart"
        
        timestamps = [s.timestamp.isoformat() for s in self.learning_history]
        confidence_values = [s.confidence_threshold for s in self.learning_history]
        position_size_values = [s.position_size_multiplier for s in self.learning_history]
        
        return f"""
        var parameterData = [
            {{
                x: {timestamps},
                y: {confidence_values},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Confidence Threshold',
                line: {{color: '#667eea'}}
            }},
            {{
                x: {timestamps},
                y: {position_size_values},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Position Size Multiplier',
                yaxis: 'y2',
                line: {{color: '#764ba2'}}
            }}
        ];
        
        var parameterLayout = {{
            title: 'Parameter Evolution Over Time',
            xaxis: {{title: 'Time'}},
            yaxis: {{title: 'Confidence Threshold', side: 'left'}},
            yaxis2: {{
                title: 'Position Size Multiplier',
                side: 'right',
                overlaying: 'y'
            }},
            showlegend: true
        }};
        
        Plotly.newPlot('parameter-chart', parameterData, parameterLayout);
        """
    
    def _generate_performance_chart_js(self) -> str:
        """Генерирует JavaScript для графика производительности."""
        if len(self.learning_history) < 2:
            return "// Not enough data for performance chart"
        
        timestamps = [s.timestamp.isoformat() for s in self.learning_history]
        win_rates = [s.win_rate * 100 for s in self.learning_history]  # Convert to percentage
        profit_factors = [s.profit_factor for s in self.learning_history]
        total_pnls = [s.total_pnl for s in self.learning_history]
        
        return f"""
        var performanceData = [
            {{
                x: {timestamps},
                y: {win_rates},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Win Rate (%)',
                line: {{color: '#28a745'}}
            }},
            {{
                x: {timestamps},
                y: {profit_factors},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Profit Factor',
                yaxis: 'y2',
                line: {{color: '#ffc107'}}
            }},
            {{
                x: {timestamps},
                y: {total_pnls},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Total PnL ($)',
                yaxis: 'y3',
                line: {{color: '#dc3545'}}
            }}
        ];
        
        var performanceLayout = {{
            title: 'Performance Metrics Over Time',
            xaxis: {{title: 'Time'}},
            yaxis: {{title: 'Win Rate (%)', side: 'left'}},
            yaxis2: {{
                title: 'Profit Factor',
                side: 'right',
                overlaying: 'y',
                position: 1
            }},
            yaxis3: {{
                title: 'Total PnL ($)',
                side: 'right',
                overlaying: 'y',
                position: 0.85
            }},
            showlegend: true
        }};
        
        Plotly.newPlot('performance-chart', performanceData, performanceLayout);
        """