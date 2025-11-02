# work/runner/live.py
"""
Live trading engine (compat-safe).

This file provides a robust, defensive implementation of LiveTradingEngine and run_live_trading()
that tolerates different shapes of signals and missing market-data backends. It preserves the
public API expected by cli_integrated.py and runner.__init__.
"""
from __future__ import annotations

import asyncio
import logging
import numpy as np
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable

try:
    # Optional structured logging init (present in the user's project)
    from infra.logging import setup_structured_logging  # type: ignore
except Exception:  # pragma: no cover

    def setup_structured_logging() -> None:
        pass


try:
    from core.config import Config  # type: ignore
except Exception:  # pragma: no cover

    @dataclass
    class Config:
        mode: str = "live"
        dry_run: bool = True
        testnet: bool = True
        symbols: List[str] = None
        symbol: str = "BTCUSDT"
        timeframe: str = "1m"
        leverage: int = 5
        risk_per_trade_pct: float = 0.5
        max_daily_loss_pct: float = 5.0
        min_notional_usdt: float = 5.0
        maker_fee: float = 0.0002
        taker_fee: float = 0.0004


# Market data
try:
    from exchange.market_data import MarketDataProvider  # type: ignore
except Exception:  # pragma: no cover
    MarketDataProvider = None  # type: ignore

# Portfolio Tracker (optional)
try:
    from utils.portfolio_tracker import PortfolioTracker  # type: ignore
except Exception:  # pragma: no cover
    PortfolioTracker = None  # type: ignore

# Optional modules; we only use them when available.
try:
    from strategy.exits import ExitManager  # type: ignore
except Exception:  # pragma: no cover
    ExitManager = None  # type: ignore

try:
    from infra.metrics import MetricsCollector  # type: ignore
except Exception:  # pragma: no cover
    MetricsCollector = None  # type: ignore

# Enhanced Dashboard Integration
try:
    from strategy.enhanced_dashboard import EnhancedDashboardGenerator  # type: ignore
except Exception:  # pragma: no cover
    EnhancedDashboardGenerator = None  # type: ignore

# DCA and LSTM integration
try:
    from strategy.dca import DCAManager  # type: ignore
except Exception:  # pragma: no cover
    DCAManager = None  # type: ignore

try:
    from models.lstm import LSTMPredictor  # type: ignore
except Exception:  # pragma: no cover
    LSTMPredictor = None  # type: ignore

logger = logging.getLogger(__name__)


# --- Internal helper structures -------------------------------------------------------


@dataclass
class NormalizedSignal:
    symbol: str
    side: str  # "BUY" or "SELL"
    strength: float = 0.0  # 0..1
    entry_price: Optional[float] = None
    timestamp: datetime = datetime.now(timezone.utc)
    meta: Dict[str, Any] = None


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            xs = x.strip()
            if xs == "":
                return None
            return float(xs)
    except Exception:
        return None
    return None


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Try attributes and dict-keys in order."""
    for n in names:
        # attribute
        if hasattr(obj, n):
            try:
                return getattr(obj, n)
            except Exception:
                pass
        # dict-like
        try:
            return obj[n]  # type: ignore
        except Exception:
            pass
    return default


def _enum_value(x: Any) -> Any:
    if x is None:
        return None
    try:
        return x.value  # enum or pydantic types
    except Exception:
        return x


def normalize_signal_obj(
    raw: Any, symbol_default: Optional[str] = None
) -> Optional[NormalizedSignal]:
    """
    Accept signal in many shapes (dataclass, pydantic model, dict, plain strings).
    Returns a NormalizedSignal or None if cannot be understood.
    """
    if raw is None:
        return None

    # If signal is a plain string like "BUY"/"SELL", treat as no-trade fallback (skip).
    if isinstance(raw, str):
        s = raw.strip().upper()
        if s in {"BUY", "SELL"}:
            # No price/strength -> skip trading because we can't size risk sanely.
            logger.warning("Signal string (%s) has no price/strength — skipping.", s)
            return None
        logger.debug("Ignoring unknown string signal: %r", raw)
        return None

    # Convenience: Some generators return (side, strength) tuple
    if isinstance(raw, (tuple, list)) and len(raw) in (2, 3):
        side = str(raw[0]).upper()
        strength = float(raw[1]) if raw[1] is not None else 0.0
        price = _to_float(raw[2]) if len(raw) == 3 else None
        if side in {"BUY", "SELL"}:
            return NormalizedSignal(
                symbol=symbol_default or "UNKNOWN",
                side=side,
                strength=strength,
                entry_price=price,
                timestamp=datetime.now(timezone.utc),
                meta={"shape": "tuple"},
            )

    # General case: object/dict with fields
    side_raw = _get(raw, "side", "signal_type", "direction", default=None)
    side_raw = _enum_value(side_raw)
    side_str = str(side_raw).upper() if side_raw is not None else None
    if side_str and side_str not in {"BUY", "SELL"}:
        # Could be "SignalType.BUY"
        if "." in side_str:
            side_str = side_str.split(".")[-1]

    symbol = _get(raw, "symbol", default=symbol_default or "UNKNOWN")
    strength = _get(raw, "strength", "confidence", "score", default=0.0) or 0.0
    entry_price = _to_float(_get(raw, "entry_price", "price", "entry", default=None))
    ts = _get(raw, "timestamp", default=None) or datetime.now(timezone.utc)
    meta: Dict[str, Any] = {}

    # Sometimes generators add 'metadata' or 'meta'
    md = _get(raw, "metadata", "meta", default=None)
    if isinstance(md, dict):
        meta.update(md)

    # If we still don't have a valid side, skip.
    if side_str not in {"BUY", "SELL"}:
        logger.debug("Cannot normalize signal side from %r — skipping.", raw)
        return None

    # Coerce strength
    try:
        strength = float(strength)
    except Exception:
        strength = 0.0

    return NormalizedSignal(
        symbol=str(symbol),
        side=side_str,
        strength=strength,
        entry_price=entry_price,
        timestamp=ts if isinstance(ts, datetime) else datetime.now(timezone.utc),
        meta=meta,
    )


# --- Live engine ----------------------------------------------------------------------


class LiveTradingEngine:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("runner.live")
        self.running = False
        self.iteration = 0

        # Symbols
        symbols = getattr(config, "symbols", None)
        if not symbols:
            sym = getattr(config, "symbol", None)
            symbols = [sym] if sym else ["BTCUSDT"]
        self.symbols: List[str] = [str(s).upper() for s in symbols]

        # Market data provider
        self.market: Optional[MarketDataProvider] = None
        try:
            if MarketDataProvider is not None:
                self.market = MarketDataProvider()
        except Exception as e:  # pragma: no cover
            self.logger.warning("MarketDataProvider init failed: %s", e)

        # Signal generator
        self.signaler = self._init_signaler()

        # Exits / metrics (optional) - будет инициализирован после adaptive_learning
        self.exit_mgr = None

        self.metrics = None
        if MetricsCollector:
            try:
                self.metrics = MetricsCollector(config)  # type: ignore
            except Exception as e:
                self.logger.debug("MetricsCollector init failed: %s", e)

        # DCA Integration - ALWAYS enabled now for better trading
        self.dca_manager = None
        if DCAManager:
            try:
                # DCAManager needs client, so we'll initialize it after binance client
                self.dca_manager = "pending"  # Will initialize in _init_binance_client
                self.logger.info(
                    "🔄 [DCA] DCA integration enabled - will initialize with Binance client"
                )
            except Exception as e:
                self.logger.warning("🔄 [DCA] DCAManager init failed: %s", e)
        else:
            self.logger.warning("🔄 [DCA] DCAManager not available - check imports")

        # LSTM Integration
        self.lstm_predictor = None
        if LSTMPredictor and getattr(config, "lstm_enable", False):
            try:
                self.lstm_predictor = LSTMPredictor(config)  # type: ignore
                self.logger.info(
                    "LSTM predictor initialized for enhanced signal generation"
                )
            except Exception as e:
                self.logger.warning("LSTMPredictor init failed: %s", e)

        # Trailing Stop Loss Manager (initialized later with client)
        self.trailing_stop_manager = None
        self.use_trailing_stop = bool(
            getattr(config, "trail_enable", True)
        )  # Use correct config field
        if self.use_trailing_stop:
            self.logger.info(
                "🎯 [TRAIL_SL] Trailing stop loss enabled - will initialize with Binance client"
            )

        # Telegram Notifications
        self.telegram = None
        try:
            tg_token = getattr(config, "tg_bot_token", "")
            tg_chat_id = getattr(config, "tg_chat_id", "")
            if tg_token and tg_chat_id:
                from infra.telegram import init_notifier

                self.telegram = init_notifier(tg_token, tg_chat_id)
                self.logger.info("[TELEGRAM] Notifier initialized")
            else:
                self.logger.info("[TELEGRAM] Not configured - notifications disabled")
        except Exception as e:
            self.logger.warning("[TELEGRAM] Failed to initialize: %s", e)

        # Accounting
        self.equity_usdt = float(getattr(config, "paper_equity", 1000.0))
        self.min_notional = float(getattr(config, "min_notional_usdt", 5.0))
        self.leverage = int(getattr(config, "leverage", 5))
        self.risk_pct = float(getattr(config, "risk_per_trade_pct", 0.5))
        self.timeframe = str(getattr(config, "timeframe", "1m"))
        self.dry_run = bool(getattr(config, "dry_run", True))

        # CRITICAL FIX: Initialize portfolio_tracker as None to prevent AttributeError
        self.portfolio_tracker = None

        # Active positions tracking for DCA
        self.active_positions: Dict[str, Dict] = {}

        # Emergency Stop Loss tracking
        self.initial_equity: Optional[float] = None  # Set on first run
        self.emergency_stop_loss_pct = float(
            getattr(config, "emergency_stop_loss_pct", 20.0)
        )
        self.logger.info(
            f"Emergency Stop Loss enabled: -{self.emergency_stop_loss_pct}% equity loss"
        )

        # Dashboard tracking
        self._start_time: Optional[datetime] = None
        self.signals_generated: int = 0
        self.signals_executed: int = 0

        # Market Context Manager for pre-trading analysis
        self.market_context_manager = None

        # Adaptive Learning System for continuous improvement
        self.adaptive_learning = None
        self.learning_enabled = getattr(config, "enable_adaptive_learning", True)

        # Enhanced Dashboard Integration
        self.dashboard = None
        self.dashboard_enabled = getattr(config, "enable_dashboard", True)
        if self.dashboard_enabled and EnhancedDashboardGenerator:
            try:
                self.dashboard = EnhancedDashboardGenerator()
                self.dashboard_last_update = 0  # Track when dashboard was last updated
                self.logger.info("📊 [DASHBOARD] Enhanced dashboard initialized")
            except Exception as e:
                self.logger.warning(
                    "📊 [DASHBOARD] Failed to initialize dashboard: %s", e
                )
                self.dashboard = None
        else:
            self.logger.info("📊 [DASHBOARD] Dashboard disabled or not available")

        self.logger.info(
            "Live trading engine initialized with DCA, LSTM and Dashboard support"
        )
    
    def _get_market_session(self) -> str:
        """Get current market session based on UTC time"""
        from datetime import datetime, timezone
        
        now = datetime.now(timezone.utc)
        hour = now.hour
        
        # Market sessions in UTC
        if 0 <= hour < 8:
            return "asian"
        elif 8 <= hour < 16:
            return "european"
        else:
            return "american"

    def _init_signaler(self) -> Any:
        # Best-effort import; we tolerate absence
        try:
            from strategy.signals import SignalGenerator  # type: ignore

            try:
                sg = SignalGenerator(self.config)  # type: ignore
            except Exception:
                sg = SignalGenerator()  # type: ignore
            # Optional init
            for name in ("initialize", "init", "setup"):
                fn = getattr(sg, name, None)
                if callable(fn):
                    try:
                        r = fn()
                        if asyncio.iscoroutine(r):
                            # Don't block here
                            asyncio.create_task(r)  # fire and forget
                    except Exception as e:
                        self.logger.debug("SignalGenerator.%s failed: %s", name, e)
            return sg
        except Exception as e:
            self.logger.warning("SignalGenerator not available: %s", e)
            return object()

    async def _produce_raw_signal(self, symbol: str, market_data: Any) -> Any:
        """Try various method names / signatures to call user's signal generator."""
        sg = self.signaler
        # Try method variants
        cand: List[Tuple[str, Tuple[Any, ...]]] = [
            ("get_signal", (symbol, market_data, self.config)),
            ("get_signal", (symbol, market_data)),
            ("get_signal", (market_data,)),
            ("generate", (symbol, market_data, self.config)),
            ("generate", (symbol, market_data)),
            ("generate", (market_data,)),
            ("generate_signal", (symbol, market_data, self.config)),
            ("generate_signal", (symbol, market_data)),
            ("generate_signal", (market_data,)),
            ("compute", (symbol, market_data)),
            ("signal", (symbol, market_data)),
        ]
        for name, args in cand:
            fn = getattr(sg, name, None)
            if not callable(fn):
                continue
            try:
                res = fn(*args)
                if asyncio.iscoroutine(res):
                    res = await res
                return res
            except TypeError:
                # Signature mismatch, try next
                continue
            except Exception as e:
                self.logger.debug("SignalGenerator.%s error: %s", name, e)
                break  # avoid spamming
        return None

    async def _latest_price(self, symbol: str) -> Optional[float]:
        if not self.market:
            return None
        # Try explicit ticker first
        try:
            if hasattr(self.market, "get_ticker") and callable(
                getattr(self.market, "get_ticker")
            ):
                t = await self.market.get_ticker(symbol)
                price = _to_float(_get(t, "price", default=None))
                if price:
                    return price
        except Exception as e:
            self.logger.debug("get_ticker failed: %s", e)
        # Try small kline fetch
        try:
            if hasattr(self.market, "get_candles"):
                kl = await self.market.get_candles(symbol, self.timeframe, limit=2)  # type: ignore
                if isinstance(kl, list) and kl:
                    last = kl[-1]
                    price = _to_float(_get(last, "close", "c", "price", default=None))
                    return price
        except Exception as e:
            self.logger.debug("get_candles failed: %s", e)
        return None

    def _position_size_qty(
        self, price: Optional[float], strength: float = 0.5
    ) -> Optional[float]:
        """
        FIXED: Conservative position sizing with proper risk management.
        Risk is calculated as % of equity, NOT multiplied by leverage.
        """
        p = _to_float(price)
        if not p or p <= 0:
            return None

        equity = float(self.equity_usdt)

        # Base risk per trade (% of total equity)
        base_risk_pct = self.risk_pct  # e.g., 0.5% means risk 0.5% of equity

        # FIXED: Cap strength multiplier to avoid huge positions
        # Strength > 1.0 is already high, so use conservative scaling
        strength_clamped = min(1.0, max(0.1, strength))  # Clamp between 0.1 and 1.0
        strength_multiplier = 0.5 + (strength_clamped * 0.5)  # Range: 0.5x to 1.0x

        adjusted_risk_pct = base_risk_pct * strength_multiplier

        # Apply volatility adjustment (simplified)
        volatility_factor = getattr(self.config, "volatility_factor", 1.0)
        risk_pct = adjusted_risk_pct / max(0.5, volatility_factor)

        # CRITICAL: Calculate risk amount as % of equity (NOT leveraged yet)
        risk_amount = equity * (risk_pct / 100.0)  # Amount we're willing to lose

        # Stop loss distance (default 2%)
        sl_distance_pct = getattr(self.config, "sl_pct", 2.0)

        # FIXED: Position value = Risk Amount / SL Distance (as decimal)
        # This gives us the NOTIONAL value we should trade
        position_value = risk_amount / (sl_distance_pct / 100.0)

        # CRITICAL FIX: Leverage is already applied by exchange, not here!
        # We only specify notional value, exchange applies leverage
        # Example: $10 risk, 2% SL = $500 position (which exchange makes 5x leverage)

        # Ensure minimum notional
        final_value = max(self.min_notional, position_value)

        # SAFETY: Cap maximum position to 20% of equity (even with leverage)
        max_position = equity * 0.2 * self.leverage
        final_value = min(final_value, max_position)

        # Convert to quantity
        qty = final_value / p

        if qty <= 0:
            return None

        self.logger.info(
            "[POSITION_SIZE] Equity=%.2f, Risk=%.2f%%, Strength=%.2f (clamped=%.2f), Value=%.2f (max=%.2f), Qty=%.6f",
            equity,
            risk_pct,
            strength,
            strength_clamped,
            final_value,
            max_position,
            qty,
        )

        return qty

    async def start(self) -> None:
        if callable(setup_structured_logging):
            try:
                setup_structured_logging()
            except Exception:
                pass
        self.running = True
        self.logger.info("Starting live trading engine...")

        # 🧪 ENHANCED: Market Context Analysis & Pre-Trading Backtest
        await self._initialize_market_context()

        # Preload historical data if configured
        preload_candles = getattr(self.config, "preload_candles", 0)
        if preload_candles > 0:
            await self._preload_historical_data(preload_candles)

        if self.metrics:
            try:
                self.metrics.start()  # type: ignore
            except Exception:
                pass

        # 📊 Initialize and launch Enhanced Dashboard on startup
        if self.dashboard:
            try:
                self.logger.info("📊 [DASHBOARD] Generating initial dashboard...")
                dashboard_path = await self.dashboard.update_dashboard(
                    trading_engine=self, adaptive_learning=self.adaptive_learning
                )

                if dashboard_path:
                    # Auto-open dashboard in browser on startup
                    try:
                        import webbrowser
                        from pathlib import Path

                        abs_path = Path(dashboard_path).resolve()
                        file_url = f"file://{abs_path}"

                        webbrowser.open(file_url)
                        self.logger.info(
                            "📊 [DASHBOARD] 🌐 Enhanced dashboard opened in browser: %s",
                            file_url,
                        )
                        self.logger.info(
                            "📊 [DASHBOARD] 🔄 Will auto-update every 30 seconds during trading"
                        )
                    except Exception as browser_e:
                        self.logger.warning(
                            "📊 [DASHBOARD] Failed to auto-open browser: %s", browser_e
                        )
                        self.logger.info(
                            "📊 [DASHBOARD] 📂 Manual open: %s", dashboard_path
                        )
                else:
                    self.logger.warning("📊 [DASHBOARD] Initial generation failed")
            except Exception as dash_e:
                self.logger.warning(
                    "📊 [DASHBOARD] Failed to initialize on startup: %s", dash_e
                )

        await self._run_trading_loop()

    async def _initialize_market_context(self) -> None:
        """Initialize market context analysis before trading starts."""
        try:
            logger.info("🔍 [MARKET_CONTEXT] Initializing market context analysis...")

            # Import MarketContextManager
            from strategy.market_context import MarketContextManager
            from data.fetchers import HistoricalDataFetcher
            from exchange.client import BinanceClient
            from copy import deepcopy

            # Create data fetcher for market analysis
            data_config = deepcopy(self.config)
            data_config.testnet = False  # Always use mainnet for historical data

            binance_client = BinanceClient(data_config)
            csv_file = getattr(self.config, "csv_data_file", None)
            data_fetcher = HistoricalDataFetcher(binance_client, csv_file=csv_file)

            # Initialize market context manager
            self.market_context_manager = MarketContextManager(
                self.config, data_fetcher
            )

            # Run market analysis
            context = await self.market_context_manager.initialize_market_context()

            # Apply recommendations to trading engine
            if context:
                logger.info("🎛️ [MARKET_CONTEXT] Applying trading recommendations...")

                # Apply adaptive confidence threshold
                original_threshold = getattr(self.config, "bt_conf_min", 0.45)
                recommended_threshold = context.suggested_confidence_threshold

                logger.info(
                    f"📊 [ADAPTIVE_THRESHOLD] Original: {original_threshold:.3f} → Recommended: {recommended_threshold:.3f}"
                )

                # Apply position size multiplier
                original_risk = self.risk_pct
                adjusted_risk = (
                    original_risk * context.suggested_position_size_multiplier
                )
                self.risk_pct = max(
                    0.1, min(2.0, adjusted_risk)
                )  # Cap between 0.1% and 2%

                logger.info(
                    f"📊 [ADAPTIVE_SIZING] Risk per trade: {original_risk:.2f}% → {self.risk_pct:.2f}%"
                )

                # Apply recommendations to signal generator if available
                if hasattr(self.signaler, "apply_market_context"):
                    try:
                        self.signaler.apply_market_context(context)
                        logger.info(
                            "🎯 [SIGNAL_TUNING] Applied market context to signal generator"
                        )
                    except Exception as sg_e:
                        logger.warning(
                            f"🎯 [SIGNAL_TUNING] Failed to apply context to signals: {sg_e}"
                        )

                # Log risk warnings
                if context.risk_warning_level in ["HIGH", "CRITICAL"]:
                    logger.warning(
                        f"⚠️ [RISK_WARNING] Market risk level: {context.risk_warning_level}"
                    )
                    logger.warning(
                        "⚠️ [RISK_WARNING] Consider reduced position sizes or pausing trading"
                    )

                logger.info(
                    "✅ [MARKET_CONTEXT] Market analysis complete - trading recommendations applied"
                )
            else:
                logger.warning(
                    "❌ [MARKET_CONTEXT] No context data received - using default settings"
                )

            # 🤖 Initialize Enhanced ML-Powered Adaptive Learning System
            if self.learning_enabled:
                try:
                    # 🚀 NEW: Enhanced ML system with real machine learning
                    from strategy.enhanced_adaptive_learning import EnhancedAdaptiveLearningSystem

                    self.enhanced_ai = EnhancedAdaptiveLearningSystem(self.config)
                    
                    # Backward compatibility: Keep old interface for existing code
                    self.adaptive_learning = self.enhanced_ai
                    
                    logger.info("🧠 [ENHANCED_AI] Advanced ML learning system initialized with:")
                    logger.info("    ✅ 4 ML models (PnL, Win Prob, Hold Time, Risk)")
                    logger.info("    ✅ Rich feature engineering (12+ market features)")
                    logger.info("    ✅ Real-time online learning")
                    logger.info("    ✅ Predictive analytics & AI recommendations")
                    
                    # 🚀 Initialize AI Status Monitor for clear visibility
                    try:
                        from strategy.ai_status_monitor import ai_monitor
                        self.ai_monitor = ai_monitor
                        logger.info("🧠 [AI_MONITOR] AI Status Monitor initialized - Enhanced logging enabled!")
                        logger.info("    ✅ Real-time ML prediction tracking")
                        logger.info("    ✅ Learning event visualization")
                        logger.info("    ✅ Performance accuracy monitoring")
                        logger.info("    ✅ Clear AI decision indicators")
                        
                        # Start periodic status reporting
                        self._ai_status_interval = 0
                        
                    except ImportError:
                        logger.warning("🧠 [AI_MONITOR] AI Status Monitor not available")
                        self.ai_monitor = None

                    # ✅ ExitManager будет инициализирован в _init_binance_client после создания клиента

                    # Apply any existing adaptive parameters
                    adaptive_params = (
                        self.adaptive_learning.get_current_adaptive_params()
                    )
                    if adaptive_params:
                        # Override settings with learned parameters
                        if adaptive_params.get("confidence_threshold"):
                            logger.info(
                                f"🧠 [LEARNING] Applying learned confidence threshold: {adaptive_params['confidence_threshold']:.3f}"
                            )

                        if adaptive_params.get("position_size_multiplier"):
                            original_risk = self.risk_pct
                            learned_risk = (
                                original_risk
                                * adaptive_params["position_size_multiplier"]
                            )
                            self.risk_pct = max(0.1, min(2.0, learned_risk))
                            logger.info(
                                f"🧠 [LEARNING] Applying learned position sizing: {original_risk:.2f}% → {self.risk_pct:.2f}%"
                            )

                        # Log learning status
                        if adaptive_params.get("is_ab_testing"):
                            logger.info(
                                f"🧪 [A/B_TEST] Currently running variant {adaptive_params.get('ab_variant')}"
                            )

                    # 🧠 Initialize Advanced AI features if available
                    await self._initialize_advanced_ai()

                    # 📊 FORCE INITIAL DASHBOARD GENERATION
                    logger.info(
                        "📊 [DASHBOARD_CHECK] Checking dashboard initialization..."
                    )
                    if self.adaptive_learning:
                        logger.info("📊 [DASHBOARD_CHECK] ✅ adaptive_learning exists")

                        if hasattr(self.adaptive_learning, "learning_visualizer"):
                            logger.info(
                                "📊 [DASHBOARD_CHECK] ✅ learning_visualizer attribute exists"
                            )

                            if self.adaptive_learning.learning_visualizer:
                                logger.info(
                                    "📊 [DASHBOARD_CHECK] ✅ learning_visualizer is initialized"
                                )

                                try:
                                    logger.info(
                                        "📊 [DASHBOARD] Force generating initial dashboard with current data..."
                                    )

                                    # Create initial snapshot
                                    snapshot = await self.adaptive_learning.learning_visualizer.capture_learning_snapshot(
                                        adaptive_learning_system=self.adaptive_learning,
                                        trading_engine=self,
                                        iteration=0,
                                    )

                                    # Generate dashboard immediately
                                    dashboard_path = (
                                        await self.adaptive_learning.learning_visualizer.create_learning_dashboard()
                                    )
                                    if dashboard_path:
                                        logger.info(
                                            f"📊 [DASHBOARD] ✅ Initial dashboard created: {dashboard_path}"
                                        )
                                        logger.info(
                                            f"📊 [DASHBOARD] 🌐 Open in browser: file://{Path(dashboard_path).absolute()}"
                                        )
                                    else:
                                        logger.warning(
                                            "📊 [DASHBOARD] ❌ Dashboard creation returned empty path"
                                        )

                                except Exception as dash_e:
                                    logger.error(
                                        f"📊 [DASHBOARD] ❌ Failed to generate initial dashboard: {dash_e}"
                                    )
                                    import traceback

                                    logger.error(
                                        f"📊 [DASHBOARD] Traceback: {traceback.format_exc()}"
                                    )
                            else:
                                logger.warning(
                                    "📊 [DASHBOARD_CHECK] ❌ learning_visualizer is None"
                                )
                        else:
                            logger.warning(
                                "📊 [DASHBOARD_CHECK] ❌ learning_visualizer attribute missing"
                            )
                    else:
                        logger.warning(
                            "📊 [DASHBOARD_CHECK] ❌ adaptive_learning is None"
                        )

                except ImportError:
                    logger.warning(
                        "🤖 [ADAPTIVE_LEARNING] AdaptiveLearningSystem not available"
                    )
                    self.adaptive_learning = None
                except Exception as learning_e:
                    logger.error(
                        f"🤖 [ADAPTIVE_LEARNING] Failed to initialize: {learning_e}"
                    )
                    self.adaptive_learning = None
            else:
                logger.info("🤖 [ADAPTIVE_LEARNING] Learning disabled by configuration")

        except Exception as e:
            logger.error(
                f"❌ [MARKET_CONTEXT] Failed to initialize market context: {e}"
            )
            logger.warning(
                "⚠️ [MARKET_CONTEXT] Continuing with default trading settings"
            )
            # Don't fail the startup - continue with default settings

    async def _preload_historical_data(self, limit: int) -> None:
        """Preload historical candle data for all symbols before trading starts."""
        self.logger.info(
            "[PRELOAD] Loading %d historical candles for IMBA signals...", limit
        )

        for symbol in self.symbols:
            try:
                self.logger.info(
                    "[PRELOAD] Loading %d candles for %s...", limit, symbol
                )

                # Use market data provider to fetch historical candles
                if self.market and hasattr(self.market, "get_candles"):
                    candles = await self.market.get_candles(
                        symbol, self.timeframe, limit=limit
                    )

                    if candles and len(candles) > 0:
                        self.logger.info(
                            "[PRELOAD] SUCCESS: Loaded %d candles for %s",
                            len(candles),
                            symbol,
                        )

                        # Convert candles to DataFrame for IMBA compatibility
                        try:
                            import pandas as pd
                            from datetime import datetime

                            # Extract OHLCV data from candles
                            if hasattr(candles, "close"):
                                # Already MarketData object - convert to DataFrame
                                df = pd.DataFrame(
                                    {
                                        "timestamp": candles.timestamp,
                                        "open": candles.open,
                                        "high": candles.high,
                                        "low": candles.low,
                                        "close": candles.close,
                                        "volume": candles.volume,
                                    }
                                )
                                df.set_index("timestamp", inplace=True)
                            elif isinstance(candles, list) and len(candles) > 0:
                                # CRITICAL FIX: Handle both dictionary and raw array formats
                                if len(candles) > 0 and isinstance(candles[0], dict):
                                    # Dictionary format from MarketDataProvider._normalize_klines
                                    df = pd.DataFrame(
                                        [
                                            {
                                                "timestamp": candle.get(
                                                    "open_time",
                                                    candle.get("timestamp", 0),
                                                ),
                                                "open": candle.get("open", 0),
                                                "high": candle.get("high", 0),
                                                "low": candle.get("low", 0),
                                                "close": candle.get("close", 0),
                                                "volume": candle.get("volume", 0),
                                            }
                                            for candle in candles
                                        ]
                                    )
                                else:
                                    # Raw array format: [timestamp, open, high, low, close, volume, ...]
                                    df = pd.DataFrame(
                                        candles,
                                        columns=[
                                            "timestamp",
                                            "open",
                                            "high",
                                            "low",
                                            "close",
                                            "volume",
                                            "close_time",
                                            "quote_volume",
                                            "trades",
                                            "taker_buy_base",
                                            "taker_buy_quote",
                                            "ignore",
                                        ],
                                    )
                                    df = df[
                                        [
                                            "timestamp",
                                            "open",
                                            "high",
                                            "low",
                                            "close",
                                            "volume",
                                        ]
                                    ]

                                # CRITICAL FIX: Safe timestamp conversion to avoid overflow
                                try:
                                    # Convert to numeric first and filter out invalid values
                                    df["timestamp"] = pd.to_numeric(
                                        df["timestamp"], errors="coerce"
                                    )

                                    # Filter out NaN/invalid timestamps
                                    valid_mask = df["timestamp"].notna()
                                    if not valid_mask.all():
                                        invalid_count = (~valid_mask).sum()
                                        self.logger.warning(
                                            f"[PRELOAD] {symbol}: Filtered {invalid_count} invalid timestamps"
                                        )
                                        df = df[valid_mask]

                                    # Check for reasonable timestamp range (2010-2030)
                                    min_timestamp = 1262304000000  # Jan 1, 2010 in ms
                                    max_timestamp = 1893456000000  # Jan 1, 2030 in ms

                                    range_mask = (df["timestamp"] >= min_timestamp) & (
                                        df["timestamp"] <= max_timestamp
                                    )
                                    if not range_mask.all():
                                        out_of_range = (~range_mask).sum()
                                        self.logger.warning(
                                            f"[PRELOAD] {symbol}: Filtered {out_of_range} out-of-range timestamps"
                                        )
                                        df = df[range_mask]

                                    if df.empty:
                                        self.logger.error(
                                            f"[PRELOAD] {symbol}: No valid timestamps after filtering"
                                        )
                                        df = None
                                    else:
                                        # Safe conversion to datetime
                                        df["timestamp"] = pd.to_datetime(
                                            df["timestamp"], unit="ms", errors="coerce"
                                        )
                                        df.set_index("timestamp", inplace=True)

                                except Exception as e:
                                    self.logger.error(
                                        f"[PRELOAD] {symbol}: Timestamp conversion failed: {e}"
                                    )
                                    df = None

                                # CRITICAL FIX: Safe conversion of numeric columns
                                if df is not None and not df.empty:
                                    try:
                                        for col in [
                                            "open",
                                            "high",
                                            "low",
                                            "close",
                                            "volume",
                                        ]:
                                            # Convert to numeric with error handling
                                            df[col] = pd.to_numeric(
                                                df[col], errors="coerce"
                                            )

                                            # Check for invalid values (NaN, inf, negative prices)
                                            invalid_mask = (
                                                df[col].isna()
                                                | np.isinf(df[col])
                                                | (df[col] <= 0)
                                            )
                                            if invalid_mask.any():
                                                invalid_count = invalid_mask.sum()
                                                self.logger.warning(
                                                    f"[PRELOAD] {symbol}: {col} has {invalid_count} invalid values, forward filling"
                                                )

                                                # Forward fill invalid values
                                                df[col] = df[col].fillna(method="ffill")
                                                df[col] = df[col].replace(
                                                    [float("inf"), float("-inf")],
                                                    method="ffill",
                                                )

                                                # If still invalid after forward fill, use reasonable defaults
                                                if (
                                                    col == "volume"
                                                    and df[col].isna().any()
                                                ):
                                                    df[col] = df[col].fillna(0)
                                                elif df[col].isna().any():
                                                    # For OHLC, use previous valid price or a default
                                                    df[col] = df[col].fillna(
                                                        df[col].iloc[-1]
                                                        if len(df) > 0
                                                        else 1.0
                                                    )

                                        # Final validation: ensure all data is finite
                                        numeric_cols = [
                                            "open",
                                            "high",
                                            "low",
                                            "close",
                                            "volume",
                                        ]
                                        valid_data_mask = (
                                            df[numeric_cols]
                                            .apply(lambda x: x.notna() & np.isfinite(x))
                                            .all(axis=1)
                                        )

                                        if not valid_data_mask.all():
                                            invalid_rows = (~valid_data_mask).sum()
                                            self.logger.warning(
                                                f"[PRELOAD] {symbol}: Removing {invalid_rows} rows with invalid data"
                                            )
                                            df = df[valid_data_mask]

                                        if df.empty:
                                            self.logger.error(
                                                f"[PRELOAD] {symbol}: No valid data after cleaning"
                                            )
                                            df = None

                                    except Exception as e:
                                        self.logger.error(
                                            f"[PRELOAD] {symbol}: Numeric conversion failed: {e}"
                                        )
                                        df = None
                            else:
                                df = None

                            if df is not None and not df.empty:
                                # Store DataFrame in SignalGenerator cache
                                if hasattr(self.signaler, "_historical_data"):
                                    if not isinstance(
                                        self.signaler._historical_data, dict
                                    ):
                                        self.signaler._historical_data = {}
                                    self.signaler._historical_data[symbol] = df

                                    # ENHANCED LOGGING: Show detailed data verification
                                    try:
                                        first_ts = df.index[0] if len(df) > 0 else "N/A"
                                        last_ts = df.index[-1] if len(df) > 0 else "N/A"
                                        memory_mb = (
                                            df.memory_usage(deep=True).sum()
                                            / 1024
                                            / 1024
                                        )

                                        # Sample LATEST data values to prove it's real and current
                                        if len(df) >= 3:
                                            # CRITICAL FIX: Show LATEST candle, not first (oldest) candle
                                            latest_idx = (
                                                -1
                                            )  # Use last (most recent) candle instead of first (oldest)
                                            sample_data = f"Latest OHLC: {df['open'].iloc[latest_idx]:.2f}/{df['high'].iloc[latest_idx]:.2f}/{df['low'].iloc[latest_idx]:.2f}/{df['close'].iloc[latest_idx]:.2f}"
                                        else:
                                            sample_data = "Insufficient data for sample"

                                        self.logger.info(
                                            f"[PRELOAD] ✅ VERIFIED DATA LOADED for {symbol}:"
                                        )
                                        self.logger.info(
                                            f"[PRELOAD]   📊 Records: {len(df)} candles"
                                        )
                                        self.logger.info(
                                            f"[PRELOAD]   📅 Period: {first_ts} → {last_ts}"
                                        )
                                        self.logger.info(
                                            f"[PRELOAD]   💾 Memory: {memory_mb:.2f} MB"
                                        )
                                        self.logger.info(
                                            f"[PRELOAD]   🎯 {sample_data}"
                                        )
                                        self.logger.info(
                                            f"[PRELOAD]   ✅ CACHED: Ready for IMBA analysis"
                                        )

                                    except Exception as detail_error:
                                        self.logger.info(
                                            "[PRELOAD] SUCCESS: Stored %d candles as DataFrame in cache for %s",
                                            len(df),
                                            symbol,
                                        )
                                        self.logger.debug(
                                            f"[PRELOAD] Detail logging failed: {detail_error}"
                                        )
                                else:
                                    self.logger.warning(
                                        "[PRELOAD] SignalGenerator has no _historical_data attribute"
                                    )
                            else:
                                self.logger.warning(
                                    "[PRELOAD] Failed to convert candles to DataFrame for %s",
                                    symbol,
                                )

                        except Exception as e:
                            self.logger.error(
                                "[PRELOAD] Failed to convert candles to DataFrame: %s",
                                e,
                            )
                            import traceback

                            self.logger.error(traceback.format_exc())
                    else:
                        self.logger.warning(
                            "[PRELOAD] WARNING: No candles returned for %s", symbol
                        )
                else:
                    self.logger.warning(
                        "[PRELOAD] WARNING: MarketDataProvider not available, skipping preload"
                    )

            except Exception as e:
                self.logger.error(
                    "[PRELOAD] ERROR: Failed to load candles for %s: %s", symbol, e
                )

        self.logger.info("[PRELOAD] COMPLETE: Historical data preload finished!")

    async def stop(self) -> None:
        self.running = False
        if self.metrics:
            try:
                self.metrics.stop()  # type: ignore
            except Exception:
                pass
        self.logger.info("Live trading engine stopped")

    async def _run_trading_loop(self) -> None:
        self.logger.info(
            "Starting main trading loop for %d symbols: %s",
            len(self.symbols),
            ", ".join(self.symbols),
        )
        # Default: iterate forever; we will sleep 1s between cycles to be gentle.
        while self.running:
            self.iteration += 1
            try:
                # === EMERGENCY STOP LOSS CHECK ===
                # Check account equity and shut down if emergency threshold breached
                if await self._check_emergency_stop_loss():
                    self.logger.critical(
                        "🚨 EMERGENCY STOP LOSS TRIGGERED! Halting all trading operations."
                    )
                    self.running = False
                    break

                for symbol in self.symbols:
                    await self._process_symbol(symbol)

                # Monitor trailing stops after processing all symbols
                if self.trailing_stop_manager:
                    try:
                        await self.trailing_stop_manager.monitor_all_positions()
                    except Exception as trail_e:
                        self.logger.error("[TRAIL_SL] Monitoring error: %s", trail_e)

                # Update PnL for all active positions periodically (every 10 iterations ~ 10 seconds)
                if self.iteration % 10 == 0:
                    await self._update_all_positions_pnl()

                # Advanced AI periodic learning and optimization (every 20 iterations ~ 20 seconds)
                if self.adaptive_learning and self.iteration % 20 == 0:
                    try:
                        await self._run_ai_optimization_cycle()
                    except Exception as ai_e:
                        self.logger.debug(
                            "[AI_CYCLE] Failed to run AI optimization: %s", ai_e
                        )

                # 📊 Generate learning visualization reports (every 60 iterations ~ 60 seconds)
                if (
                    self.adaptive_learning
                    and hasattr(self.adaptive_learning, "learning_visualizer")
                    and self.adaptive_learning.learning_visualizer
                    and self.iteration % 60 == 0
                ):
                    try:
                        await self._generate_learning_visualization()
                    except Exception as viz_e:
                        self.logger.debug(
                            "[LEARNING_VIZ] Failed to generate visualization: %s", viz_e
                        )

                # 📊 Update Enhanced Dashboard (every 30 iterations ~ 30 seconds)
                if self.dashboard and self.iteration % 30 == 0:
                    try:
                        await self._update_enhanced_dashboard()
                    except Exception as dash_e:
                        self.logger.debug(
                            "[ENHANCED_DASHBOARD] Failed to update dashboard: %s",
                            dash_e,
                        )

                # Log portfolio summary periodically (every 60 iterations ~ 1 minute)
                if self.portfolio_tracker and self.iteration % 60 == 0:
                    try:
                        self.portfolio_tracker.log_portfolio_summary()
                    except Exception as pt_e:
                        self.logger.debug(
                            "[PORTFOLIO] Failed to log portfolio summary: %s", pt_e
                        )

            except Exception as e:
                self.logger.error("Error in trading loop: %s", e)
            # Fast polling for responsive trading - 1 second intervals
            # User requested fast response time for signal detection
            await asyncio.sleep(
                1.0
            )  # Fast 1-second polling for immediate signal response

    async def _process_symbol(self, symbol: str) -> None:
        # Set current symbol for proper price rounding
        self._current_symbol = symbol

        # Fetch market data for the signaler; if fails, pass None (signaler will fallback)
        md: Any = None
        if self.market and hasattr(self.market, "get_candles"):
            try:
                md = await self.market.get_candles(
                    symbol, self.timeframe, limit=250
                )  # IMBA needs 250+ candles
                
                # 🧠 NEW: Cache candles data for ML system
                if not hasattr(self, '_last_candles_data'):
                    self._last_candles_data = {}
                self._last_candles_data[symbol] = md
                
            except Exception as e:
                self.logger.debug("get_candles(%s) error: %s", symbol, e)

        raw = await self._produce_raw_signal(symbol, md)
        sig = normalize_signal_obj(raw, symbol_default=symbol)
        if not sig:
            # Log that we checked this symbol but got no signal
            self.logger.debug(
                "[SYMBOL_CHECK] %s: No actionable signal (wait/rejected)", symbol
            )
            return  # nothing actionable

        # Ensure we have a price to size the order
        price = sig.entry_price or await self._latest_price(symbol)
        if not price:
            self.logger.debug("Skip %s: missing price", symbol)
            return

        # 🧠 NEW: Enhanced ML Analysis of Signal Context
        enhanced_analysis = None
        if self.enhanced_ai and md is not None:
            try:
                # Convert market data to DataFrame if needed
                import pandas as pd
                
                if hasattr(md, 'close'):
                    # Already a structured object
                    candles_df = pd.DataFrame({
                        'timestamp': md.timestamp,
                        'open': md.open,
                        'high': md.high,
                        'low': md.low,
                        'close': md.close,
                        'volume': md.volume
                    })
                elif isinstance(md, list) and len(md) > 0:
                    # List of candles - convert to DataFrame
                    if isinstance(md[0], dict):
                        candles_df = pd.DataFrame(md)
                    else:
                        # Raw array format
                        candles_df = pd.DataFrame(md, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume'
                        ])
                else:
                    candles_df = None
                
                if candles_df is not None and len(candles_df) > 20:
                    # Get ML analysis with predictions and recommendations
                    import time
                    start_time = time.time()
                    
                    enhanced_analysis = await self.enhanced_ai.analyze_signal_context(
                        symbol=symbol,
                        candles_data=candles_df,
                        current_price=price,
                        signal_strength=sig.strength,
                        additional_context={
                            'timeframe': self.timeframe,
                            'iteration': self.iteration,
                            'market_session': self._get_market_session(),
                            'volatility_factor': getattr(self.config, 'volatility_factor', 1.0)
                        }
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # 🧠 NEW: AI Status Monitor - Log ML prediction with clear indicators
                    if enhanced_analysis and hasattr(self, 'ai_monitor') and self.ai_monitor:
                        try:
                            ml_pred = enhanced_analysis.get('ml_predictions', {})
                            ai_rec = enhanced_analysis.get('ai_recommendations', {})
                            trading_decision = enhanced_analysis.get('trading_decision', {})
                            risk_assessment = enhanced_analysis.get('risk_assessment', {})
                            
                            # Log prediction with AI Status Monitor for maximum visibility
                            self.ai_monitor.log_prediction(
                                symbol=symbol,
                                predictions=ml_pred,
                                decision=trading_decision,
                                processing_time=processing_time
                            )
                            
                        except Exception as monitor_e:
                            self.logger.warning("🧠 [AI_MONITOR] Failed to log prediction: %s", monitor_e)
                    
                    # Log ML analysis results
                    if enhanced_analysis:
                        ml_pred = enhanced_analysis.get('ml_predictions', {})
                        ai_rec = enhanced_analysis.get('ai_recommendations', {})
                        trading_decision = enhanced_analysis.get('trading_decision', {})
                        risk_assessment = enhanced_analysis.get('risk_assessment', {})
                        
                        self.logger.info(
                            "🧠 [ML_ANALYSIS] %s: Expected PnL %+.2f%% | Win Prob %.0f%% | Risk %s | Confidence %.2f",
                            symbol,
                            ml_pred.get('expected_pnl_pct', 0),
                            ml_pred.get('win_probability', 0.5) * 100,
                            risk_assessment.get('risk_level', 'unknown'),
                            ml_pred.get('prediction_confidence', 0)
                        )
                        
                        self.logger.info("🎯 [TRADING_DECISION] %s", trading_decision.get('reasoning', 'No reasoning'))
                        
                        # Show AI recommendations
                        recommendations = ai_rec.get('recommendations', [])
                        if recommendations:
                            for rec in recommendations[:3]:  # Show top 3
                                self.logger.info(
                                    "💡 [AI_REC] %s (confidence: %.0f%%)",
                                    rec.get('action', 'unknown'), 
                                    rec.get('confidence', 0) * 100
                                )
                        
                        # Apply ML-driven position sizing
                        if trading_decision.get('should_trade', True):
                            ml_size_multiplier = trading_decision.get('position_size_multiplier', 1.0)
                            original_strength = sig.strength
                            # Combine original signal strength with ML recommendations
                            enhanced_strength = original_strength * ml_size_multiplier
                            sig.strength = min(2.0, max(0.1, enhanced_strength))  # Cap between 0.1 and 2.0
                            
                            self.logger.info(
                                "📊 [ML_SIZING] Signal strength: %.2f → %.2f (ML multiplier: %.2f)",
                                original_strength, sig.strength, ml_size_multiplier
                            )
                            
                            # 🧠 AI Status Monitor - Log position adjustment
                            if hasattr(self, 'ai_monitor') and self.ai_monitor:
                                try:
                                    self.ai_monitor.log_position_adjustment(
                                        symbol=symbol,
                                        original_strength=original_strength,
                                        ml_multiplier=ml_size_multiplier,
                                        new_strength=sig.strength
                                    )
                                except Exception as monitor_adj_e:
                                    self.logger.warning("🧠 [AI_MONITOR] Failed to log position adjustment: %s", monitor_adj_e)
                        else:
                            # ML system recommends not trading
                            self.logger.warning(
                                "🚫 [ML_BLOCK] ML system recommends SKIPPING trade: %s",
                                trading_decision.get('reasoning', 'Low confidence/high risk')
                            )
                            return  # Skip this trade
                    
            except Exception as ml_e:
                self.logger.warning("🧠 [ML_ANALYSIS] Error in ML analysis: %s", ml_e)
                enhanced_analysis = None
        
        # Calculate position size with potentially ML-adjusted strength
        qty = self._position_size_qty(price, sig.strength)
        if not qty:
            self.logger.debug(
                "Skip %s: missing qty (price=%s strength=%s)", symbol, price, sig.strength
            )
            return

        # Log the intended action
        self.logger.info(
            "[SIGNAL] %s %s @ %.2f (strength=%.2f) -> qty=%.6f [DRY-RUN=%s]",
            sig.side,
            symbol,
            price,
            sig.strength,
            qty,
            self.dry_run,
        )

        # Place real orders if not in dry-run mode
        if not self.dry_run:
            try:
                # Check if we already have an open position BEFORE placing new order
                has_position = await self._check_existing_position(symbol, sig.side)
                if has_position:
                    self.logger.info(
                        "[POSITION_EXISTS] Already have %s position for %s, skipping new entry",
                        sig.side,
                        symbol,
                    )
                    # Check DCA conditions instead of opening new position
                    if self.dca_manager and symbol in self.active_positions:
                        try:
                            # FIXED: Update position PnL from exchange before DCA check
                            await self._update_position_pnl(symbol)
                            current_position = self.active_positions[symbol]

                            # Log current position status for debugging
                            pnl = current_position.get("unrealized_pnl", 0.0)
                            pnl_pct = (
                                pnl
                                / (
                                    current_position.get("entry_price", 1)
                                    * current_position.get("quantity", 1)
                                )
                            ) * 100
                            self.logger.info(
                                "[POSITION_STATUS] %s: PnL=$%.2f (%.2f%%), checking DCA conditions...",
                                symbol,
                                pnl,
                                pnl_pct,
                            )

                            should_dca = await self.dca_manager.should_dca(
                                symbol, current_position
                            )
                            if should_dca:
                                self.logger.info(
                                    "[DCA_TRIGGER] Conditions met for DCA on %s", symbol
                                )
                                await self._handle_dca_trigger(symbol, current_position)
                            else:
                                self.logger.debug(
                                    "[DCA_CHECK] No DCA needed for %s (PnL: %.2f%%)",
                                    symbol,
                                    pnl_pct,
                                )
                        except Exception as dca_e:
                            self.logger.warning("[DCA] Error checking DCA: %s", dca_e)
                    return

                # CRITICAL: Check for OPPOSITE direction position
                opposite_side = "SELL" if sig.side == "BUY" else "BUY"
                has_opposite = await self._check_existing_position(
                    symbol, opposite_side
                )
                if has_opposite:
                    self.logger.warning(
                        "[POSITION_CONFLICT] ⚠️  Signal %s but have %s position for %s - BLOCKING TRADE!",
                        sig.side,
                        opposite_side,
                        symbol,
                    )
                    self.logger.warning(
                        "[POSITION_CONFLICT] Refusing to enter against existing position direction"
                    )
                    return

                await self._place_order(symbol, sig.side, qty, price, sig.strength)

                # Update active positions for DCA tracking
                if symbol not in self.active_positions:
                    self.active_positions[symbol] = {
                        "side": sig.side,
                        "entry_price": price,
                        "quantity": qty,
                        "unrealized_pnl": 0.0,
                        "created_time": sig.timestamp,
                    }

                    # Start DCA position if enabled
                    if self.dca_manager and hasattr(
                        self.dca_manager, "start_dca_position"
                    ):
                        try:
                            from core.types import OrderSide

                            order_side = (
                                OrderSide.BUY if sig.side == "BUY" else OrderSide.SELL
                            )
                            # CRITICAL FIX: Handle both sync and async DCA methods
                            result = self.dca_manager.start_dca_position(
                                symbol,
                                order_side,
                                price,
                                f"Signal strength: {sig.strength:.2f}",
                            )
                            if asyncio.iscoroutine(result):
                                await result
                            self.logger.info(
                                "[DCA] Started DCA position for %s", symbol
                            )
                        except Exception as dca_e:
                            self.logger.warning(
                                "[DCA] Failed to start DCA position: %s", dca_e
                            )

            except Exception as e:
                self.logger.error("Failed to place order for %s: %s", symbol, e)

        # DCA is now ONLY checked inside the "has_position" block above
        # This prevents DCA from triggering on every signal
        # DCA should only trigger when:
        #   1. We have an existing position
        #   2. Price has moved significantly against us
        #   3. We haven't exceeded max_levels

        # Call exit manager hooks if available (they will no-op in dry-run).
        if self.exit_mgr and hasattr(self.exit_mgr, "on_new_signal"):
            try:
                r = self.exit_mgr.on_new_signal(symbol, sig.side, price, qty)  # type: ignore
                if asyncio.iscoroutine(r):
                    await r
            except Exception as e:
                self.logger.debug("ExitManager.on_new_signal error: %s", e)

    async def _place_order(
        self, symbol: str, side: str, qty: float, price: float, strength: float
    ) -> None:
        """Place a real order on the exchange."""
        try:
            # Initialize Binance client if not already done
            if not hasattr(self, "_binance_client") or not self._binance_client:
                await self._init_binance_client()

            # Ensure minimum notional value
            notional = qty * price
            if notional < self.min_notional:
                self.logger.warning(
                    "[ORDER_SKIP] Notional %.2f < min %.2f USDT",
                    notional,
                    self.min_notional,
                )
                return

            # Place market order for immediate execution
            order_side = side.upper()  # "BUY" or "SELL"

            # Round quantity to proper precision for Binance Futures (per symbol)
            rounded_qty = self._round_quantity(qty, symbol)

            # Symbol-specific minimum quantities and step sizes
            min_qty_map = {
                "BTCUSDT": 0.001,
                "ETHUSDT": 0.001,
                "SOLUSDT": 0.1,  # CRITICAL FIX: SOLUSDT step size is 0.1, not 0.01
                "BNBUSDT": 0.01,
                "AVAXUSDT": 1.0,  # FIXED: AVAX needs whole numbers like ADA/DOGE
                "XRPUSDT": 0.1,
                "ADAUSDT": 1.0,  # Whole numbers only
                "DOGEUSDT": 1.0,  # Whole numbers only
                "MATICUSDT": 1.0,  # Whole numbers only
                "LINKUSDT": 0.1,
            }

            min_qty = min_qty_map.get(symbol, 0.001)  # Default to 0.001

            if rounded_qty < min_qty:
                rounded_qty = min_qty
                self.logger.info(
                    "[ORDER] Quantity %.3f too small, using minimum %.3f",
                    round(qty, 3),
                    min_qty,
                )

            # Get symbol-specific decimal places for accurate logging
            precision_map = {
                "BTCUSDT": 3,
                "ETHUSDT": 3,
                "BNBUSDT": 2,
                "SOLUSDT": 1,  # CRITICAL FIX: SOLUSDT step size 0.1 = 1 decimal
                "XRPUSDT": 1,
                "ADAUSDT": 0,
                "DOGEUSDT": 0,
                "AVAXUSDT": 0,  # FIXED: AVAX needs 0 decimals like ADA/DOGE
                "LINKUSDT": 1,
                "MATICUSDT": 0,
            }
            decimal_places = precision_map.get(symbol, 3)
            qty_format = f"{{:.{decimal_places}f}}"

            self.logger.info(
                "[ORDER] Placing %s order: %s %s @ %.2f (notional: %.2f USDT)",
                order_side,
                qty_format.format(rounded_qty),
                symbol,
                price,
                notional,
            )

            # Use the Binance client to place order
            if hasattr(self, "client") and self.client:
                try:
                    # CRITICAL FIX: Force proper decimal places for API
                    # Convert float to string with proper precision, then back to float
                    # This prevents 2.77 from being sent as 2.770 to Binance
                    api_quantity = float(f"{rounded_qty:.{decimal_places}f}")

                    # Get current bid/ask to calculate expected execution price
                    try:
                        ticker = self.client.get_ticker_price(symbol)
                        current_market_price = float(ticker.get("price", price))

                        # Calculate expected slippage for market order
                        price_diff = abs(current_market_price - price)
                        slippage_pct = (price_diff / price) * 100 if price > 0 else 0

                        # CRITICAL WARNING: Alert if slippage is too high (>2%)
                        if slippage_pct > 2.0:
                            self.logger.warning(
                                f"🚨 [HIGH_SLIPPAGE] Expected: ${price:.4f}, Market: ${current_market_price:.4f}, Slippage: {slippage_pct:.2f}%"
                            )

                            # SAFETY: Block trade if slippage is extreme (>5%)
                            if slippage_pct > 5.0:
                                self.logger.error(
                                    f"❌ [EXTREME_SLIPPAGE] Blocking trade due to {slippage_pct:.2f}% slippage (>5% threshold)"
                                )
                                return  # Don't place order with extreme slippage

                        # Use LIMIT order with slight price adjustment for better execution
                        if order_side == "BUY":
                            # For BUY orders, add small premium to ensure execution
                            limit_price = current_market_price * 1.001  # 0.1% premium
                        else:
                            # For SELL orders, subtract small discount to ensure execution
                            limit_price = current_market_price * 0.999  # 0.1% discount

                        # Round limit price to proper precision
                        limit_price = self._round_price(limit_price, symbol)

                        self.logger.info(
                            f"[ORDER_PRICE] Signal: ${price:.4f}, Market: ${current_market_price:.4f}, Limit: ${limit_price:.4f}"
                        )

                    except Exception as price_e:
                        self.logger.warning(
                            f"[PRICE_CHECK] Failed to get market price: {price_e}, using MARKET order"
                        )
                        limit_price = None

                    # Place order with better execution strategy
                    if limit_price:
                        order_result = self.client.place_order(
                            symbol=symbol,
                            side=order_side,
                            type="LIMIT",
                            quantity=api_quantity,
                            price=limit_price,
                            timeInForce="IOC",  # Immediate or Cancel - fills immediately or cancels
                        )
                    else:
                        # Fallback to market order
                        order_result = self.client.place_order(
                            symbol=symbol,
                            side=order_side,
                            type="MARKET",
                            quantity=api_quantity,
                        )

                    # CRITICAL: Verify order was actually filled!
                    order_id = order_result.get("orderId", "N/A")
                    order_status = order_result.get("status", "UNKNOWN")
                    executed_qty = float(order_result.get("executedQty", 0))
                    fills = order_result.get("fills", [])

                    self.logger.info(
                        "[ORDER_RESULT] %s order %s: status=%s, executed=%.3f/%.3f",
                        order_side,
                        order_id,
                        order_status,
                        executed_qty,
                        rounded_qty,
                    )

                    # CRITICAL DEBUG: Log exact values for debugging
                    self.logger.info(
                        "[ORDER_DEBUG] status='%s', executed_qty=%s (type: %s), zero_check=%s",
                        order_status,
                        executed_qty,
                        type(executed_qty),
                        executed_qty == 0,
                    )

                    # CRITICAL FIX: Handle MARKET order execution delay
                    # Market orders on Binance Futures can have status=NEW initially, then execute
                    # Check for both 0 and 0.0 and also small float precision issues
                    if order_status == "NEW" and (
                        executed_qty == 0 or executed_qty == 0.0 or executed_qty < 0.001
                    ):
                        self.logger.info(
                            "[ORDER_WAIT] Market order placed, waiting for execution..."
                        )

                        # Wait up to 5 seconds for market order to execute
                        for attempt in range(10):  # 10 attempts * 0.5s = 5 seconds max
                            await asyncio.sleep(0.5)

                            # Check order status again - use get_open_orders since get_order doesn't exist
                            try:
                                # Check if order is still in open orders (if not, it was likely filled OR cancelled)
                                open_orders = self.client.get_open_orders(symbol=symbol)
                                order_still_open = any(
                                    order.get("orderId") == order_id
                                    for order in open_orders
                                )

                                if not order_still_open:
                                    # Order not in open orders - check position to see if it was filled vs cancelled
                                    positions = self.client.get_positions()
                                    position_found = False
                                    
                                    for pos in positions:
                                        if pos.get("symbol") == symbol:
                                            pos_amt = float(pos.get("positionAmt", 0))
                                            if abs(pos_amt) > 0:
                                                # Position exists - order was filled
                                                order_status = "FILLED"
                                                executed_qty = abs(pos_amt)
                                                fills = []  # We don't have individual fills info
                                                position_found = True
                                                break
                                    
                                    if not position_found:
                                        # No position found - order was cancelled/rejected
                                        order_status = "CANCELLED"
                                        executed_qty = 0.0
                                else:
                                    # Order still open - not filled yet
                                    order_status = "NEW"
                                    executed_qty = 0.0

                                self.logger.debug(
                                    "[ORDER_CHECK] Attempt %d: status=%s, executed=%.3f",
                                    attempt + 1,
                                    order_status,
                                    executed_qty,
                                )

                                if order_status == "FILLED" and executed_qty > 0:
                                    break
                            except Exception as check_e:
                                self.logger.warning(
                                    "[ORDER_CHECK] Failed to check order %s: %s",
                                    order_id,
                                    check_e,
                                )
                                continue

                    # Check if order was actually filled (after waiting if necessary)
                    if order_status == "FILLED" and executed_qty > 0:
                        # Calculate actual fill price from fills
                        if fills:
                            total_value = sum(
                                float(fill["price"]) * float(fill["qty"])
                                for fill in fills
                            )
                            avg_fill_price = total_value / executed_qty
                            self.logger.info(
                                "[ORDER_FILLED] ✅ %s %.3f %s @ $%.2f (avg fill price)",
                                order_side,
                                executed_qty,
                                symbol,
                                avg_fill_price,
                            )
                        else:
                            avg_fill_price = price  # Fallback to market price

                        # Verify position was created
                        await self._verify_position_created(
                            symbol, order_side, executed_qty
                        )

                        # Setup Take Profit and Stop Loss orders with actual fill data
                        await self._setup_tp_sl_orders(
                            symbol, order_side, executed_qty, avg_fill_price, strength
                        )

                        # Register TP/SL orders with Exit Tracker if available
                        if (
                            self.exit_tracker
                            and hasattr(self, "pending_trades")
                            and symbol in self.pending_trades
                        ):
                            try:
                                trade_id = self.pending_trades[symbol].get("trade_id")
                                if trade_id:
                                    # Note: We would register actual order IDs here if we captured them
                                    # For now, exit tracker will use fallback history monitoring
                                    self.logger.info(
                                        "🎯 [EXIT_TRACKER] Will monitor position exits for trade %s",
                                        trade_id,
                                    )
                            except Exception as et_e:
                                self.logger.warning(
                                    "🎯 [EXIT_TRACKER] Failed to register orders: %s",
                                    et_e,
                                )

                        # 🧠 NEW: Record trade opening with Enhanced ML system
                        if self.enhanced_ai:
                            try:
                                from strategy.adaptive_learning import TradeRecord
                                from datetime import datetime, timezone
                                import uuid

                                # Generate unique trade ID
                                trade_id = f"T_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

                                # Calculate TP/SL levels for record (matches existing method signature)
                                tp_prices, sl_price = self._calculate_tp_sl_levels(
                                    avg_fill_price, order_side, strength
                                )

                                # 🚀 NEW: Create trade record with ML prediction data
                                trade_record = TradeRecord(
                                    timestamp=datetime.now(timezone.utc),
                                    symbol=symbol,
                                    side=order_side,
                                    entry_price=avg_fill_price,
                                    exit_price=avg_fill_price,  # Will be updated on close
                                    quantity=executed_qty,
                                    pnl=0.0,  # Will be calculated on close
                                    pnl_pct=0.0,  # Will be calculated on close
                                    hold_time_seconds=0.0,  # Will be calculated on close
                                    signal_strength=strength,
                                    market_conditions={
                                        "volatility": getattr(
                                            self.config, "volatility_factor", 1.0
                                        ),
                                        "volume_ratio": 1.0,
                                        "risk_level": "medium",
                                    },
                                    was_dca=False,
                                    exit_reason="pending",  # Will be updated on close
                                    trade_id=trade_id,  # Added trade_id field we just created
                                )
                                
                                # 🧠 NEW: Store ML predictions for later validation
                                if enhanced_analysis and 'ml_predictions' in enhanced_analysis:
                                    ml_predictions = enhanced_analysis.get('ml_predictions', {})
                                    trade_record.ml_prediction = ml_predictions
                                    
                                    self.logger.info(
                                        "🧠 [ML_RECORD] Storing ML predictions: PnL %+.2f%%, Win prob %.0f%%",
                                        ml_predictions.get('expected_pnl_pct', 0),
                                        ml_predictions.get('win_probability', 0.5) * 100
                                    )

                                # 🚀 ENHANCED: Record with ML system using both old and new methods
                                # Use new enhanced method if available
                                if hasattr(self.enhanced_ai, 'record_trade_with_ml'):
                                    # Get market data for ML recording
                                    candles_data = None
                                    if hasattr(self, '_last_candles_data'):
                                        candles_data = getattr(self, '_last_candles_data', {}).get(symbol)
                                    
                                    await self.enhanced_ai.record_trade_with_ml(trade_record, candles_data)
                                    self.logger.info("🧠 [ENHANCED_ML] ✅ Trade recorded with ML context")
                                else:
                                    # Fallback to old method
                                    await self.enhanced_ai.record_trade(trade_record)
                                    self.logger.info("🤖 [BASIC_ML] ✅ Trade recorded (basic method)")

                                # Also keep pending trades for position tracking
                                self.pending_trades = getattr(
                                    self, "pending_trades", {}
                                )
                                self.pending_trades[symbol] = {
                                    "entry_time": datetime.now(timezone.utc),
                                    "entry_price": avg_fill_price,
                                    "quantity": executed_qty,
                                    "side": order_side,
                                    "signal_strength": strength,
                                    "trade_record": trade_record,  # Reference to AI record
                                    "trade_id": trade_id,  # Store trade ID for exit tracking
                                    "ml_analysis": enhanced_analysis,  # Store ML analysis for exit processing
                                }

                                self.logger.info(
                                    "🧠 [ENHANCED_LEARNING] ✅ Trade recorded in advanced ML system: %s %.3f @ %.4f",
                                    order_side,
                                    executed_qty,
                                    avg_fill_price,
                                )
                            except Exception as learning_e:
                                self.logger.error(
                                    f"🧠 [ENHANCED_LEARNING] ❌ Failed to record trade in ML system: {learning_e}"
                                )
                                import traceback

                                self.logger.error(
                                    f"🧠 [ENHANCED_LEARNING] Traceback: {traceback.format_exc()}"
                                )

                        self.logger.info(
                            "[ORDER_SUCCESS] ✅ %s position opened: %.3f %s",
                            order_side,
                            executed_qty,
                            symbol,
                        )
                    else:
                        # CRITICAL: If order still not filled, cancel it and exit
                        if order_status in ["NEW", "PARTIALLY_FILLED"]:
                            self.logger.warning(
                                "[ORDER_CANCEL] Cancelling unfilled order %s", order_id
                            )
                            try:
                                self.client.cancel_order(
                                    symbol=symbol, orderId=order_id
                                )
                                self.logger.info(
                                    "[ORDER_CANCEL] ✅ Order %s cancelled", order_id
                                )
                            except Exception as cancel_e:
                                self.logger.error(
                                    "[ORDER_CANCEL] Failed to cancel order %s: %s",
                                    order_id,
                                    cancel_e,
                                )

                        self.logger.error(
                            "[ORDER_FAILED] ❌ Order not filled: status=%s, executed=%.3f",
                            order_status,
                            executed_qty,
                        )
                        if order_result:
                            self.logger.error(
                                "[ORDER_DETAILS] Full result: %s", order_result
                            )
                        return  # Don't setup TP/SL for unfilled orders

                except Exception as order_error:
                    self.logger.error(
                        "[ORDER_EXCEPTION] ❌ Failed to place %s order: %s",
                        order_side,
                        order_error,
                    )
                    # Try to get account info for debugging
                    try:
                        account = self.client.futures_account()
                        balance = account.get("totalWalletBalance", "unknown")
                        self.logger.error(
                            "[ACCOUNT_DEBUG] Wallet balance: %s USDT", balance
                        )
                    except Exception:
                        self.logger.error("[ACCOUNT_DEBUG] Could not get account info")
                    raise order_error

                # NOTE: Equity tracking removed for FUTURES trading
                # In futures, equity doesn't change when opening positions
                # Equity changes only when positions are closed (realized PnL)
                # Unrealized PnL is tracked separately through position monitoring
            else:
                self.logger.warning(
                    "[ORDER_SKIP] No Binance client available, order not placed"
                )

        except Exception as e:
            self.logger.error(
                "[ORDER_FAILED] Failed to place %s order for %s: %s", side, symbol, e
            )
            raise

    async def _verify_position_created(
        self, symbol: str, side: str, expected_qty: float
    ) -> bool:
        """Verify that a position was actually created on the exchange after order execution."""
        try:
            if not hasattr(self, "client") or not self.client:
                self.logger.warning(
                    "[POSITION_VERIFY] No client available for verification"
                )
                return False

            # Wait a moment for position to be updated on exchange
            await asyncio.sleep(0.5)

            positions = self.client.get_positions()

            for pos in positions:
                if pos.get("symbol") == symbol:
                    pos_amt = float(pos.get("positionAmt", 0))
                    if pos_amt != 0:
                        # Check if position side matches and quantity is reasonable
                        actual_side = "BUY" if pos_amt > 0 else "SELL"
                        if actual_side == side:
                            self.logger.info(
                                "[POSITION_VERIFY] ✅ Position confirmed: %.3f %s (%s)",
                                abs(pos_amt),
                                symbol,
                                side,
                            )

                            # Check if quantity matches (allow small differences due to rounding)
                            qty_diff = abs(abs(pos_amt) - expected_qty)
                            if (
                                qty_diff > expected_qty * 0.1
                            ):  # More than 10% difference
                                self.logger.warning(
                                    "[POSITION_VERIFY] ⚠️ Quantity mismatch: expected %.3f, got %.3f",
                                    expected_qty,
                                    abs(pos_amt),
                                )
                            return True

            self.logger.error(
                "[POSITION_VERIFY] ❌ No position found after order execution!"
            )
            return False

        except Exception as e:
            self.logger.error("[POSITION_VERIFY] Failed to verify position: %s", e)
            return False

    async def _check_existing_position(self, symbol: str, side: str) -> bool:
        """Check if we already have a SIGNIFICANT open position for this symbol with same side."""
        try:
            if not hasattr(self, "client") or not self.client:
                # No client, check local tracking
                return (
                    symbol in self.active_positions
                    and self.active_positions[symbol].get("side") == side
                )

            # Get position from Binance (get all positions, then filter)
            try:
                positions = self.client.get_positions()

                # Define minimum position thresholds to be considered "significant"
                # Positions smaller than these are considered dust/residual
                min_position_thresholds = {
                    "BTCUSDT": 0.001,  # $10+ at $100k
                    "ETHUSDT": 0.01,  # $40+ at $4k
                    "BNBUSDT": 0.01,  # $12+ at $1200
                    "SOLUSDT": 0.1,  # $18+ at $180
                    "LINKUSDT": 1.0,  # $18+ at $18
                    "ADAUSDT": 10.0,  # $6+ at $0.6
                    "XRPUSDT": 10.0,  # $6+ at $0.6
                    "DOGEUSDT": 50.0,  # $10+ at $0.2
                    "AVAXUSDT": 0.1,  # $4+ at $40
                    "MATICUSDT": 10.0,  # $6+ at $0.6
                }

                min_threshold = min_position_thresholds.get(symbol, 0.001)

                # Check if any position has SIGNIFICANT amount with matching side
                for pos in positions:
                    pos_amt = float(pos.get("positionAmt", 0))
                    if pos_amt != 0:
                        # Positive amount = LONG, negative = SHORT
                        pos_side = "BUY" if pos_amt > 0 else "SELL"
                        if pos.get("symbol") == symbol and pos_side == side:
                            abs_pos_amt = abs(pos_amt)

                            if abs_pos_amt >= min_threshold:
                                self.logger.debug(
                                    "[POSITION_CHECK] Found SIGNIFICANT %s position: %.3f %s (>= %.3f threshold)",
                                    side,
                                    abs_pos_amt,
                                    symbol,
                                    min_threshold,
                                )
                                return True
                            else:
                                # Found position but it's too small - consider it dust
                                self.logger.info(
                                    "[POSITION_DUST] Found tiny %s position: %.6f %s (< %.3f threshold) - IGNORING as dust",
                                    side,
                                    abs_pos_amt,
                                    symbol,
                                    min_threshold,
                                )

                                # TODO: Optionally close dust positions automatically
                                # await self._close_dust_position(symbol, pos_amt)

                return False

            except Exception as e:
                self.logger.debug(
                    "[POSITION_CHECK] Failed to get positions from exchange: %s", e
                )
                # Fallback to local tracking
                return (
                    symbol in self.active_positions
                    and self.active_positions[symbol].get("side") == side
                )

        except Exception as e:
            self.logger.warning("[POSITION_CHECK] Error checking position: %s", e)
            return False

    async def _update_position_pnl(self, symbol: str) -> None:
        """Update the unrealized PnL for an active position from Binance."""
        try:
            if not hasattr(self, "client") or not self.client:
                self.logger.debug(
                    "[PNL_UPDATE] No client available, skipping PnL update"
                )
                return

            if symbol not in self.active_positions:
                self.logger.debug(
                    "[PNL_UPDATE] No active position found for %s", symbol
                )
                return

            # Get current position from Binance
            positions = self.client.get_positions()

            for pos in positions:
                if pos.get("symbol") == symbol:
                    pos_amt = float(pos.get("positionAmt", 0))
                    if pos_amt != 0:  # Active position found
                        unrealized_pnl = float(pos.get("unRealizedPnl", 0))
                        mark_price = float(pos.get("markPrice", 0))
                        entry_price = float(pos.get("entryPrice", 0))

                        # Update our local tracking
                        self.active_positions[symbol].update(
                            {
                                "unrealized_pnl": unrealized_pnl,
                                "mark_price": mark_price,
                                "entry_price": entry_price,  # Use actual entry price from exchange
                                "quantity": abs(pos_amt),
                                "side": "BUY" if pos_amt > 0 else "SELL",
                            }
                        )

                        pnl_pct = (
                            (unrealized_pnl / (entry_price * abs(pos_amt))) * 100
                            if entry_price > 0 and pos_amt != 0
                            else 0
                        )

                        self.logger.debug(
                            "[PNL_UPDATE] %s: PnL=$%.2f (%.2f%%), Entry=$%.4f, Mark=$%.4f",
                            symbol,
                            unrealized_pnl,
                            pnl_pct,
                            entry_price,
                            mark_price,
                        )
                        return

            # Position not found on exchange - it might have been closed
            self.logger.warning(
                "[PNL_UPDATE] Position for %s not found on exchange - might be closed",
                symbol,
            )

        except Exception as e:
            self.logger.warning("[PNL_UPDATE] Error updating PnL for %s: %s", symbol, e)

    async def _update_all_positions_pnl(self) -> None:
        """Update PnL for all active positions periodically."""
        try:
            if not self.active_positions:
                return

            for symbol in list(self.active_positions.keys()):
                await self._update_position_pnl(symbol)

                # Log summary of updated positions every 10th update
                if self.iteration % 100 == 0:  # Every ~100 seconds
                    position = self.active_positions.get(symbol)
                    if position:
                        pnl = position.get("unrealized_pnl", 0.0)
                        pnl_pct = (
                            pnl
                            / (
                                position.get("entry_price", 1)
                                * position.get("quantity", 1)
                            )
                        ) * 100
                        self.logger.info(
                            "[PNL_SUMMARY] %s: $%.2f (%.2f%%)", symbol, pnl, pnl_pct
                        )

        except Exception as e:
            self.logger.debug("[PNL_UPDATE_ALL] Error updating all positions: %s", e)

    async def _run_ai_optimization_cycle(self) -> None:
        """Run AI optimization and learning cycle periodically."""
        try:
            if not self.adaptive_learning:
                return

            # Check if we have enough data for optimization
            trade_count = len(getattr(self.adaptive_learning, "trades_history", []))

            # 🧠 NEW: AI Status Monitor - Periodic status reporting with clear visibility
            if hasattr(self, 'ai_monitor') and self.ai_monitor:
                # Show periodic AI status every 300 iterations (~5 minutes) 
                if self.iteration % 300 == 0:
                    try:
                        self.ai_monitor.log_periodic_status()
                    except Exception as monitor_status_e:
                        self.logger.warning("🧠 [AI_MONITOR] Failed to log periodic status: %s", monitor_status_e)
            
            # Log AI status periodically (backup/fallback)
            if self.iteration % 200 == 0:  # Every ~3 minutes
                self.logger.info(
                    "🧠 [AI_STATUS] Trades recorded: %d, Learning active: %s",
                    trade_count,
                    bool(self.adaptive_learning),
                )

                # Show current AI parameters
                if hasattr(self.adaptive_learning, "current_params"):
                    params = self.adaptive_learning.current_params
                    self.logger.info("🧠 [AI_PARAMS] Current parameters: %s", params)

            # Run optimization if we have enough trades
            if trade_count >= 20 and self.iteration % 1000 == 0:  # Every ~15 minutes
                self.logger.info(
                    "🎯 [AI_OPTIMIZATION] Running periodic optimization with %d trades",
                    trade_count,
                )

                # Run Bayesian optimization
                if (
                    hasattr(self.adaptive_learning, "advanced_ai")
                    and self.adaptive_learning.advanced_ai
                ):
                    optimal_params = (
                        await self.adaptive_learning.optimize_parameters_with_ai(
                            "sharpe_ratio"
                        )
                    )

                    if optimal_params:
                        self.logger.info(
                            "🎯 [AI_OPTIMIZATION] New optimal parameters found"
                        )

                        # Apply optimizations
                        if "confidence_threshold" in optimal_params:
                            self.logger.info(
                                "🎯 [AI_TUNED] Confidence: %.3f",
                                optimal_params["confidence_threshold"],
                            )

                        if "position_size_multiplier" in optimal_params:
                            old_risk = self.risk_pct
                            adjusted_risk = (
                                old_risk * optimal_params["position_size_multiplier"]
                            )
                            self.risk_pct = max(0.1, min(2.0, adjusted_risk))
                            self.logger.info(
                                "🎯 [AI_TUNED] Position size: %.2f%% → %.2f%%",
                                old_risk,
                                self.risk_pct,
                            )

                        # Log reasoning if available
                        if "reasoning" in optimal_params:
                            self.logger.info(
                                "🧠 [AI_REASONING] %s", optimal_params["reasoning"]
                            )

                # Run A/B testing analysis
                if hasattr(self.adaptive_learning, "analyze_ab_test_results"):
                    ab_results = await self.adaptive_learning.analyze_ab_test_results()
                    if ab_results and ab_results.get("winner"):
                        self.logger.info(
                            "🧪 [A/B_TEST] Winner variant: %s (confidence: %.2f)",
                            ab_results["winner"],
                            ab_results.get("confidence", 0),
                        )

            # Pattern recognition every 500 iterations
            if trade_count >= 50 and self.iteration % 500 == 0:  # Every ~8 minutes
                self.logger.info("🧩 [PATTERN] Running pattern recognition analysis...")

                if hasattr(self.adaptive_learning, "discover_patterns"):
                    patterns = await self.adaptive_learning.discover_patterns()
                    if patterns:
                        for pattern in patterns[:3]:  # Log top 3 patterns
                            self.logger.info(
                                "🧩 [PATTERN] Found: %s (strength: %.2f)",
                                pattern.get("description", "Unknown"),
                                pattern.get("strength", 0),
                            )

        except Exception as e:
            self.logger.debug("[AI_CYCLE] AI optimization cycle failed: %s", e)

    async def _init_binance_client(self) -> None:
        """Initialize Binance client for order placement."""
        try:
            if hasattr(self, "client") and self.client:
                self._binance_client = self.client
                self.logger.info("[BINANCE] Using existing client for orders")
            else:
                # Import and create Binance client
                try:
                    from exchange.client import BinanceClient

                    self.client = BinanceClient(
                        api_key=getattr(self.config, "api_key", None),
                        api_secret=getattr(self.config, "api_secret", None),
                        testnet=getattr(self.config, "testnet", True),
                    )
                    self._binance_client = self.client
                    self.logger.info(
                        "[BINANCE] Initialized trading client (testnet=%s)",
                        getattr(self.config, "testnet", True),
                    )

                    # ✅ Теперь инициализируем ExitManager с готовой системой адаптивного обучения и клиентом
                    if ExitManager and not self.exit_mgr and self.adaptive_learning:
                        try:
                            self.exit_mgr = ExitManager(
                                self.client, self.config, self.adaptive_learning
                            )
                            self.logger.info(
                                "🎯 [EXIT_MANAGER] Exit tracking integrated with AI learning system"
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"🎯 [EXIT_MANAGER] Failed to initialize with AI integration: {e}"
                            )
                            # Fallback без AI интеграции
                            try:
                                self.exit_mgr = ExitManager(self.client, self.config)
                                self.logger.info(
                                    "🎯 [EXIT_MANAGER] Exit tracking initialized (without AI integration)"
                                )
                            except Exception as e2:
                                self.logger.error(
                                    f"🎯 [EXIT_MANAGER] Complete initialization failed: {e2}"
                                )
                except Exception as init_e:
                    self.logger.error(
                        "[BINANCE] Failed to initialize trading client: %s", init_e
                    )
                    self.client = None
                    self._binance_client = None

            # Initialize DCA Manager with Binance client
            if self.dca_manager == "pending" and self.client and DCAManager:
                try:
                    # Import DCAConfig here to avoid circular imports
                    from strategy.dca import DCAConfig

                    # Convert Config to DCAConfig with proper mapping
                    dca_config = DCAConfig(
                        max_levels=getattr(self.config, "max_levels", 3),
                        level_spacing_pct=getattr(
                            self.config, "level_spacing_pct", 1.5
                        ),
                        level_multipliers=getattr(
                            self.config, "level_multipliers", [1.0, 1.5, 2.0, 2.5, 3.0]
                        ),
                        base_quantity_usd=100.0,  # Default base size
                        max_total_investment=500.0,  # Max per position
                        max_drawdown_pct=15.0,  # Stop at 15% loss
                    )

                    self.dca_manager = DCAManager(self.client, dca_config)  # type: ignore
                    self.logger.info(
                        "[DCA] DCA Manager initialized with Binance client"
                    )
                    self.logger.info(
                        f"[DCA] Config: max_levels={dca_config.max_levels}, spacing={dca_config.level_spacing_pct}%"
                    )
                except Exception as dca_e:
                    self.logger.warning(
                        "[DCA] Failed to initialize DCA Manager: %s", dca_e
                    )
                    self.dca_manager = None

            # Initialize Trailing Stop Manager with Binance client
            if self.use_trailing_stop and self.client:
                try:
                    from exchange.trailing_stop import (
                        TrailingStopManager,
                        TrailingStopConfig,
                    )
                    from exchange.orders import OrderManager

                    # Get or create order manager
                    if not hasattr(self, "order_manager"):
                        self.order_manager = OrderManager(self.client)

                    # Create trailing stop manager with default config
                    trailing_config = TrailingStopConfig()
                    self.trailing_stop_manager = TrailingStopManager(
                        client=self.client,
                        order_manager=self.order_manager,
                        config=trailing_config,
                    )

                    self.logger.info("[TRAIL_SL] Trailing stop manager initialized")

                except Exception as trail_e:
                    self.logger.warning(
                        "[TRAIL_SL] Failed to initialize trailing stop manager: %s",
                        trail_e,
                    )
                    self.trailing_stop_manager = None

            # Initialize Portfolio Tracker with Binance client
            if self.client and PortfolioTracker:
                try:
                    self.portfolio_tracker = PortfolioTracker(self.client, self.config)
                    self.logger.info(
                        "[PORTFOLIO] Portfolio Tracker initialized - Balance/P&L monitoring enabled"
                    )
                except Exception as pt_e:
                    self.logger.warning(
                        "[PORTFOLIO] Failed to initialize Portfolio Tracker: %s", pt_e
                    )
                    self.portfolio_tracker = None
            else:
                self.portfolio_tracker = None

            # Initialize Exit Tracker for real-time TP/SL monitoring
            if self.client:
                try:
                    from exchange.exit_tracker import (
                        create_exit_tracker,
                        integrate_exit_tracker_callbacks,
                    )

                    self.exit_tracker = await create_exit_tracker(
                        self.client, self.adaptive_learning
                    )
                    integrate_exit_tracker_callbacks(self.exit_tracker, self)

                    self.logger.info(
                        "🎯 [EXIT_TRACKER] Real-time TP/SL tracking initialized"
                    )

                except Exception as et_e:
                    self.logger.warning(
                        "🎯 [EXIT_TRACKER] Failed to initialize exit tracker: %s", et_e
                    )
                    self.exit_tracker = None
            else:
                self.exit_tracker = None

        except Exception as e:
            self.logger.error("[BINANCE] Failed to initialize client: %s", e)
            self._binance_client = None

    async def _setup_tp_sl_orders(
        self, symbol: str, side: str, qty: float, entry_price: float, strength: float
    ) -> None:
        """Setup comprehensive Take Profit and Stop Loss orders with advanced monolith logic."""
        try:
            # Calculate TP/SL levels based on signal strength and market conditions
            tp_levels, sl_level = self._calculate_tp_sl_levels(
                entry_price, side, strength
            )

            if not tp_levels and not sl_level:
                self.logger.info("[TP_SL] No TP/SL levels calculated for %s", symbol)
                return

            # Try advanced exit integration from monolith first
            success = await self._try_advanced_exit_setup(
                symbol, side, qty, entry_price, sl_level, tp_levels
            )

            if not success:
                # Fallback to direct order placement
                self.logger.info(
                    "[TP_SL] Using direct order placement (advanced exits not available)"
                )

                # Setup Stop Loss order
                if sl_level:
                    await self._place_stop_loss_order(symbol, side, qty, sl_level)

                # Setup Take Profit orders (multiple levels)
                if tp_levels:
                    await self._place_take_profit_orders(symbol, side, qty, tp_levels)

            self.logger.info(
                "[TP_SL] Setup complete for %s: %d TP levels, SL at %.2f",
                symbol,
                len(tp_levels),
                sl_level or 0.0,
            )

            # Register position with trailing stop manager
            if self.trailing_stop_manager and tp_levels and sl_level:
                try:
                    self.trailing_stop_manager.register_position(
                        symbol=symbol,
                        entry_price=entry_price,
                        side=side,
                        tp_levels=tp_levels,
                        initial_sl=sl_level,
                    )
                    self.logger.info("[TRAIL_SL] Position registered for %s", symbol)
                except Exception as trail_e:
                    self.logger.warning(
                        "[TRAIL_SL] Failed to register position: %s", trail_e
                    )

        except Exception as e:
            self.logger.error(
                "[TP_SL_FAILED] Failed to setup TP/SL for %s: %s", symbol, e
            )

    async def _try_advanced_exit_setup(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        sl_level: float,
        tp_levels: list[float],
    ) -> bool:
        """Try to use advanced exit management from monolith."""
        try:
            # Import advanced exit functions from monolith
            from exchange.exits import (
                ensure_exits_on_exchange,
                ensure_sl_on_exchange,
                ensure_tp_on_exchange,
            )

            # Determine position sign for monolith compatibility
            pos_sign = 1 if side.upper() == "BUY" else -1

            # Calculate TP shares for distribution
            tp_shares = (
                [0.4, 0.35, 0.25]
                if len(tp_levels) == 3
                else [1.0 / len(tp_levels)] * len(tp_levels)
            )

            self.logger.info("[ADVANCED_TP_SL] Using monolith exit management system")

            # Use monolith's comprehensive exit setup
            ensure_exits_on_exchange(
                symbol=symbol,
                pos_sign_or_side=pos_sign,
                qty=qty,
                sl=sl_level,
                tps=tp_levels,
                tp_shares=tp_shares,
            )

            self.logger.info("[ADVANCED_TP_SL] Monolith exit setup successful")
            return True

        except ImportError as e:
            self.logger.debug("[ADVANCED_TP_SL] Monolith exits not available: %s", e)
            return False
        except Exception as e:
            self.logger.warning("[ADVANCED_TP_SL] Monolith exit setup failed: %s", e)
            return False

    def _calculate_tp_sl_levels(
        self, entry_price: float, side: str, strength: float
    ) -> tuple[list[float], float]:
        """Calculate Take Profit and Stop Loss levels using configuration settings with leverage consideration."""
        try:
            # Get configuration values
            leverage = getattr(self.config, "leverage", 5.0)

            # FIXED: Use correct config attribute for SL
            base_sl_pct = getattr(self.config, "sl_fixed_pct", 2.0)

            # FIXED: Use configuration TP levels instead of hardcoded ones
            config_tp_levels = (
                self.config.parse_tp_levels()
            )  # Get from config: [1.5, 3.0, 5.0]

            # SMART TP ADJUSTMENT: Apply strength multiplier but keep TPs reasonable for leverage
            # Higher strength = slightly more aggressive targets, but not extreme
            if strength >= 2.0:
                # Very strong signals: +20% more aggressive
                strength_multiplier = 1.2
            elif strength >= 1.5:
                # Strong signals: +10% more aggressive
                strength_multiplier = 1.1
            elif strength >= 1.0:
                # Normal signals: use base levels
                strength_multiplier = 1.0
            else:
                # Weak signals: -10% more conservative
                strength_multiplier = 0.9

            # Apply strength adjustment to TP levels
            tp_levels_pct = [tp * strength_multiplier for tp in config_tp_levels]

            # Ensure TP levels don't exceed safe limits for leverage (avoid liquidation)
            max_safe_tp = 8.0 if leverage <= 5 else 5.0 if leverage <= 10 else 3.0
            tp_levels_pct = [min(tp, max_safe_tp) for tp in tp_levels_pct]

            # SL remains consistent - no strength adjustment for better risk management
            sl_pct = base_sl_pct

            # Calculate actual price levels with proper symbol precision
            symbol = getattr(
                self, "_current_symbol", "BTCUSDT"
            )  # Get current symbol for rounding

            if side.upper() == "BUY":
                # Long position: SL below entry, TP above entry
                sl_level = self._round_price(entry_price * (1 - sl_pct / 100), symbol)
                tp_levels = [
                    self._round_price(entry_price * (1 + tp_pct / 100), symbol)
                    for tp_pct in tp_levels_pct
                ]
            else:
                # Short position: SL above entry, TP below entry
                sl_level = self._round_price(entry_price * (1 + sl_pct / 100), symbol)
                tp_levels = [
                    self._round_price(entry_price * (1 - tp_pct / 100), symbol)
                    for tp_pct in tp_levels_pct
                ]

            # Calculate profit percentages with leverage for logging
            profit_pcts = [tp_pct * leverage for tp_pct in tp_levels_pct]

            self.logger.info(
                "[TP_SL_CALC] 🎯 %s %dx: Entry=$%.4f, SL=$%.4f(-%.1f%%), TPs=%s",
                side,
                leverage,
                entry_price,
                sl_level,
                sl_pct,
                [
                    f"${tp:.4f}({profit:.1f}%)"
                    for tp, profit in zip(tp_levels, profit_pcts)
                ],
            )

            # Verify all levels are different and valid
            if len(set(tp_levels)) != len(tp_levels):
                self.logger.warning(
                    "[TP_SL_CALC] ⚠️ Some TP levels are identical after rounding!"
                )

            return tp_levels, sl_level

        except Exception as e:
            self.logger.error("[TP_SL_CALC_FAILED] Failed to calculate levels: %s", e)
            # Fallback to safe defaults
            default_tps = (
                [entry_price * 1.015, entry_price * 1.03, entry_price * 1.05]
                if side.upper() == "BUY"
                else [entry_price * 0.985, entry_price * 0.97, entry_price * 0.95]
            )
            default_sl = (
                entry_price * 0.98 if side.upper() == "BUY" else entry_price * 1.02
            )
            return default_tps, default_sl

    async def _cancel_exit_orders(self, symbol: str) -> None:
        """Cancel all existing stop loss and take profit orders for symbol."""
        try:
            if not hasattr(self, "client") or not self.client:
                return

            # Get all open orders for this symbol
            open_orders = self.client.get_open_orders(symbol=symbol)

            cancelled_count = 0
            for order in open_orders:
                order_type = order.get("type", "").upper()
                # AGGRESSIVE CLEANUP: Cancel ANY reduceOnly, STOP, or TP order
                is_exit_order = (
                    order_type
                    in ("STOP_MARKET", "STOP", "TAKE_PROFIT_MARKET", "TAKE_PROFIT")
                    or order.get("reduceOnly", False)
                    or str(order.get("closePosition", "")).lower() == "true"
                )

                if is_exit_order:
                    try:
                        self.client.cancel_order(
                            symbol=symbol, orderId=order["orderId"]
                        )
                        cancelled_count += 1
                        self.logger.info(
                            "[EXIT_CANCEL] ✅ Cancelled %s order %s (reduceOnly=%s)",
                            order_type,
                            order["orderId"],
                            order.get("reduceOnly", False),
                        )
                    except Exception as e:
                        self.logger.warning(
                            "[EXIT_CANCEL_FAILED] ❌ Could not cancel order %s: %s",
                            order["orderId"],
                            e,
                        )
                else:
                    self.logger.debug(
                        "[EXIT_SKIP] Skipping non-exit order %s (type=%s)",
                        order["orderId"],
                        order_type,
                    )

            if cancelled_count > 0:
                self.logger.info(
                    "[EXIT_CANCEL] Cancelled %d existing exit orders for %s",
                    cancelled_count,
                    symbol,
                )

        except Exception as e:
            self.logger.warning(
                "[EXIT_CANCEL_FAILED] Failed to cancel exit orders for %s: %s",
                symbol,
                e,
            )

    async def _place_stop_loss_order(
        self, symbol: str, side: str, qty: float, sl_price: float
    ) -> None:
        """Place stop loss order (cancels existing exit orders first)."""
        try:
            # Cancel existing exit orders before placing new ones
            await self._cancel_exit_orders(symbol)

            # Determine SL order side (opposite of entry)
            sl_side = "SELL" if side == "BUY" else "BUY"

            # Round SL price to tick size
            rounded_sl_price = self._round_price(sl_price)

            self.logger.info(
                "[SL_ORDER] Placing %s STOP_MARKET (closePosition): %s @ %.2f",
                sl_side,
                symbol,
                rounded_sl_price,
            )

            if hasattr(self, "client") and self.client:
                # Skip time sync - method not available in BinanceClient
                # Binance client handles timestamp internally
                self.logger.debug(
                    "[TIME_SYNC] Using client internal timestamp handling"
                )

                # Place STOP_MARKET order with closePosition=true
                # This closes entire position when stop price is hit
                sl_result = self.client.place_order(
                    symbol=symbol,
                    side=sl_side,
                    type="STOP_MARKET",
                    stopPrice=rounded_sl_price,
                    closePosition="true",  # Close all position
                    workingType="MARK_PRICE",  # Use mark price to avoid manipulation
                )

                self.logger.info(
                    "[SL_SUCCESS] Stop loss placed: %s (closePosition=true)",
                    sl_result.get("orderId", "N/A"),
                )

        except Exception as e:
            self.logger.error("[SL_FAILED] Failed to place stop loss: %s", e)

    async def _place_take_profit_orders(
        self, symbol: str, side: str, total_qty: float, tp_levels: list[float]
    ) -> None:
        """Place multiple take profit orders with quantity distribution."""
        try:
            # Distribute quantity across TP levels (per symbol precision)
            tp_quantities = self._distribute_tp_quantities(
                total_qty, len(tp_levels), symbol
            )

            # Determine TP order side (opposite of entry)
            tp_side = "SELL" if side == "BUY" else "BUY"

            for i, (tp_price, tp_qty) in enumerate(zip(tp_levels, tp_quantities)):
                try:
                    # Round values to proper precision (per symbol)
                    rounded_tp_qty = self._round_quantity(tp_qty, symbol)
                    rounded_tp_price = self._round_price(tp_price)

                    self.logger.info(
                        "[TP_ORDER] Placing %s take profit %d: %.3f %s @ %.2f",
                        tp_side,
                        i + 1,
                        rounded_tp_qty,
                        symbol,
                        rounded_tp_price,
                    )

                    if hasattr(self, "client") and self.client:
                        # Place LIMIT order with reduceOnly=true for take profit
                        # This ensures TP orders only close position, never open new ones
                        tp_result = self.client.place_order(
                            symbol=symbol,
                            side=tp_side,
                            type="LIMIT",
                            quantity=rounded_tp_qty,
                            price=rounded_tp_price,
                            timeInForce="GTC",
                            reduceOnly="true",  # Only close position, don't reverse
                        )

                        self.logger.info(
                            "[TP_SUCCESS] Take profit %d placed: %s",
                            i + 1,
                            tp_result.get("orderId", "N/A"),
                        )

                except Exception as e:
                    self.logger.error(
                        "[TP_FAILED] Failed to place take profit %d: %s", i + 1, e
                    )

        except Exception as e:
            self.logger.error(
                "[TP_ORDERS_FAILED] Failed to place take profit orders: %s", e
            )

    def _distribute_tp_quantities(
        self, total_qty: float, num_levels: int, symbol: str = "BTCUSDT"
    ) -> list[float]:
        """Distribute quantity across TP levels with smart allocation."""
        if num_levels <= 0:
            return []

        # Smart distribution: More quantity on earlier TPs
        # Example: [40%, 35%, 25%] for 3 levels
        if num_levels == 1:
            ratios = [1.0]
        elif num_levels == 2:
            ratios = [0.6, 0.4]  # 60%, 40%
        elif num_levels == 3:
            ratios = [0.4, 0.35, 0.25]  # 40%, 35%, 25%
        else:
            # For more levels, create decreasing distribution
            ratios = [0.5 / (i + 1) for i in range(num_levels)]
            total_ratio = sum(ratios)
            ratios = [r / total_ratio for r in ratios]

        # Calculate actual quantities and round appropriately (per symbol)
        quantities = []
        remaining_qty = total_qty

        for i, ratio in enumerate(ratios[:-1]):
            qty = self._round_quantity(total_qty * ratio, symbol)
            quantities.append(qty)
            remaining_qty -= qty

        # Last quantity gets the remainder to avoid rounding errors
        quantities.append(self._round_quantity(remaining_qty, symbol))

        self.logger.info(
            "[TP_DIST] Quantity distribution: %s (total: %.3f)",
            [f"{q:.3f}" for q in quantities],
            sum(quantities),
        )

        return quantities

    def _round_price(self, price: float, symbol: str = "BTCUSDT") -> float:
        """Round price to proper Binance tick size per symbol."""
        # Symbol-specific tick sizes for Binance futures (CORRECTED for current prices)
        tick_size_map = {
            "BTCUSDT": 0.10,  # High-value BTC: tick size 0.10
            "ETHUSDT": 0.01,  # ETH: tick size 0.01
            "BNBUSDT": 0.01,  # BNB: tick size 0.01
            "SOLUSDT": 0.01,  # SOL: tick size 0.01
            "XRPUSDT": 0.0001,  # XRP: tick size 0.0001 (current price ~$0.60)
            "ADAUSDT": 0.001,  # FIXED: ADA tick size 0.001 (current price ~$0.70, was 0.0001)
            "DOGEUSDT": 0.00001,  # DOGE: tick size 0.00001 (5 decimals)
            "AVAXUSDT": 0.01,  # FIXED: AVAX tick size 0.01 (current price ~$40, was 0.001)
            "LINKUSDT": 0.001,  # LINK: tick size 0.001
            "MATICUSDT": 0.0001,  # MATIC: tick size 0.0001
        }

        # Get tick size for this symbol, default to 0.01
        tick_size = tick_size_map.get(symbol, 0.01)

        # Round to nearest tick
        rounded_price = round(price / tick_size) * tick_size

        # Determine number of decimal places from tick size
        if tick_size >= 0.1:
            decimals = 1
        elif tick_size >= 0.01:
            decimals = 2
        elif tick_size >= 0.001:
            decimals = 3
        elif tick_size >= 0.0001:
            decimals = 4
        else:
            decimals = 5

        return round(rounded_price, decimals)

    def _round_quantity(self, quantity: float, symbol: str = "BTCUSDT") -> float:
        """Round quantity to proper Binance precision per symbol."""
        # Different symbols have different quantity precision:
        # BTCUSDT, ETHUSDT: 3 decimals (0.001)
        # ADAUSDT, DOGEUSDT, MATICUSDT: 0 decimals (whole numbers only)
        # XRPUSDT, LINKUSDT: 1 decimal (0.1)
        # SOLUSDT, BNBUSDT, AVAXUSDT: 2 decimals (0.01)

        # Map symbols to their precision (decimals)
        # CRITICAL: SOLUSDT step size is 0.1, so precision=1 decimal place
        precision_map = {
            "BTCUSDT": 3,
            "ETHUSDT": 3,
            "BNBUSDT": 2,
            "SOLUSDT": 1,  # CRITICAL FIX: SOLUSDT step size is 0.1, precision=1
            "XRPUSDT": 1,
            "ADAUSDT": 0,
            "DOGEUSDT": 0,
            "AVAXUSDT": 0,  # FIXED: AVAX needs whole numbers like ADA/DOGE
            "LINKUSDT": 1,
            "MATICUSDT": 0,
        }

        # Get precision for this symbol, default to 3
        precision = precision_map.get(symbol, 3)

        return round(quantity, precision)

    def _calculate_tp_sl_levels(
        self, entry_price: float, side: str, signal_strength: float
    ) -> tuple:
        """Calculate TP/SL levels for trade record."""
        try:
            # Use same logic as _setup_tp_sl_orders
            leverage = getattr(self.config, "leverage", 10)

            # SL calculation (2% against position)
            sl_pct = 0.02
            if side == "BUY":
                sl_price = entry_price * (1 - sl_pct)
            else:
                sl_price = entry_price * (1 + sl_pct)

            # TP calculation (dynamic based on signal strength)
            base_tp_pct = 0.014  # 1.4% base
            strength_multiplier = min(2.0, max(1.0, signal_strength))

            tp_distances = [
                base_tp_pct * strength_multiplier * 1.0,  # TP1: 1.4%
                base_tp_pct * strength_multiplier * 1.5,  # TP2: 2.1%
                base_tp_pct * strength_multiplier * 2.0,  # TP3: 2.8%
            ]

            tp_prices = []
            for tp_dist in tp_distances:
                if side == "BUY":
                    tp_price = entry_price * (1 + tp_dist)
                else:
                    tp_price = entry_price * (1 - tp_dist)
                tp_prices.append(tp_price)

            return tp_prices, sl_price

        except Exception as e:
            self.logger.error(f"Failed to calculate TP/SL levels: {e}")
            return [], entry_price

    async def _handle_dca_trigger(self, symbol: str, position: Dict) -> None:
        """Handle DCA trigger by placing additional orders."""
        try:
            self.logger.info("[DCA_TRIGGER] Processing DCA trigger for %s", symbol)

            if not self.dca_manager or not hasattr(
                self.dca_manager, "active_positions"
            ):
                self.logger.warning("[DCA] DCA manager not properly initialized")
                return

            # Get current market price
            current_price = await self._latest_price(symbol)
            if not current_price:
                self.logger.warning("[DCA] Could not get current price for %s", symbol)
                return

            # Update unrealized PnL
            entry_price = position.get("entry_price", 0)
            quantity = position.get("quantity", 0)
            side = position.get("side", "BUY")

            if side == "BUY":
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity

            position["unrealized_pnl"] = pnl
            position["current_price"] = current_price

            # Calculate DCA order size (typically same as original or larger)
            original_qty = position.get("quantity", 0)

            # Get DCA multiplier from configuration or default
            dca_multiplier = getattr(self.config, "dca_multiplier", 1.0)
            dca_qty = original_qty * dca_multiplier

            # Place DCA order (same side as original position to average down/up)
            await self._place_order(
                symbol, side, dca_qty, current_price, 0.8
            )  # Lower strength for DCA

            # Update position tracking
            self.active_positions[symbol] = {
                "side": side,
                "entry_price": (entry_price * quantity + current_price * dca_qty)
                / (quantity + dca_qty),  # New weighted average
                "quantity": quantity + dca_qty,
                "unrealized_pnl": pnl,
                "created_time": position.get("created_time"),
                "dca_count": position.get("dca_count", 0) + 1,
            }

            self.logger.info(
                "[DCA_SUCCESS] DCA order placed for %s: %.3f @ %.2f (total qty: %.3f)",
                symbol,
                dca_qty,
                current_price,
                quantity + dca_qty,
            )

        except Exception as e:
            self.logger.error(
                "[DCA_FAILED] Failed to handle DCA trigger for %s: %s", symbol, e
            )

    async def _update_position_pnl(self, symbol: str) -> None:
        """Update position PnL using real Binance API data."""
        try:
            if not self.client:
                return

            # Get real position data from Binance
            positions = self.client.get_positions()

            for pos in positions:
                if pos["symbol"] == symbol:
                    position_amt = float(pos.get("positionAmt", 0))
                    if abs(position_amt) > 0:  # Active position
                        unrealized_pnl = float(pos.get("unRealizedProfit", 0))
                        mark_price = float(pos.get("markPrice", 0))
                        entry_price = float(pos.get("entryPrice", 0))

                        # Update our position tracking
                        if symbol in self.active_positions:
                            self.active_positions[symbol].update(
                                {
                                    "unrealized_pnl": unrealized_pnl,
                                    "current_price": mark_price,
                                    "entry_price": entry_price,
                                    "quantity": abs(position_amt),
                                    "side": "BUY" if position_amt > 0 else "SELL",
                                }
                            )

                            self.logger.debug(
                                f"[PNL_UPDATE] {symbol}: {unrealized_pnl:+.2f} USDT @ {mark_price:.4f}"
                            )
                        break

        except Exception as e:
            self.logger.debug(f"[PNL_UPDATE] Failed to update PnL for {symbol}: {e}")

    async def _update_all_positions_pnl(self) -> None:
        """Update PnL for all active positions and detect closed positions."""
        try:
            if not self.active_positions:
                return

            # Get all positions from Binance in one call
            if not self.client:
                return

            positions = self.client.get_positions()

            # Create set of symbols that still have active positions
            active_symbols = set()

            for pos in positions:
                symbol = pos["symbol"]
                position_amt = float(pos.get("positionAmt", 0))

                if abs(position_amt) > 0:
                    active_symbols.add(symbol)

                    if symbol in self.active_positions:
                        unrealized_pnl = float(pos.get("unRealizedProfit", 0))
                        mark_price = float(pos.get("markPrice", 0))
                        entry_price = float(pos.get("entryPrice", 0))

                        # Update position tracking
                        self.active_positions[symbol].update(
                            {
                                "unrealized_pnl": unrealized_pnl,
                                "current_price": mark_price,
                                "entry_price": entry_price,
                                "quantity": abs(position_amt),
                                "side": "BUY" if position_amt > 0 else "SELL",
                            }
                        )

                        # Check DCA triggers with real PnL
                        if self.dca_manager:
                            await self.dca_manager.check_dca_trigger(
                                symbol, self.active_positions[symbol]
                            )

            # CRITICAL FIX: Detect closed positions and notify AI system
            closed_symbols = set(self.active_positions.keys()) - active_symbols

            for closed_symbol in closed_symbols:
                try:
                    await self._handle_position_closed(closed_symbol)
                except Exception as close_e:
                    self.logger.error(
                        f"[POSITION_CLOSED] Error handling closed position {closed_symbol}: {close_e}"
                    )

        except Exception as e:
            self.logger.debug(f"[PNL_UPDATE_ALL] Failed: {e}")

    async def _handle_position_closed(self, symbol: str) -> None:
        """Handle when a position is detected as closed."""
        try:
            # Get pending trade info if available
            pending_trade = getattr(self, "pending_trades", {}).get(symbol)

            if pending_trade and "trade_id" in pending_trade:
                trade_id = pending_trade["trade_id"]
                entry_price = pending_trade.get("entry_price", 0.0)

                # Get current market price as estimated exit price
                current_price = await self._latest_price(symbol)
                if not current_price:
                    current_price = entry_price  # Fallback

                # Determine exit reason (we don't know exact reason, so estimate)
                side = pending_trade.get("side", "BUY")
                if side == "BUY":
                    if current_price > entry_price:
                        exit_reason = "tp_estimated"
                    else:
                        exit_reason = "sl_estimated"
                else:  # SELL
                    if current_price < entry_price:
                        exit_reason = "tp_estimated"
                    else:
                        exit_reason = "sl_estimated"

                self.logger.info(
                    f"🎯 [POSITION_CLOSED] Detected closed position: {symbol} @ ~${current_price:.2f}"
                )

                # Notify Exit Manager (which will notify AI system)
                if self.exit_mgr and hasattr(self.exit_mgr, "notify_position_closed"):
                    await self.exit_mgr.notify_position_closed(
                        trade_id=trade_id,
                        symbol=symbol,
                        exit_price=current_price,
                        exit_reason=exit_reason,
                        fees_paid=0.0,  # Estimate
                    )

                # Clean up pending trade
                if hasattr(self, "pending_trades") and symbol in self.pending_trades:
                    del self.pending_trades[symbol]

            else:
                self.logger.debug(
                    f"[POSITION_CLOSED] No pending trade info for {symbol}"
                )

            # Remove from active positions
            if symbol in self.active_positions:
                del self.active_positions[symbol]

        except Exception as e:
            self.logger.error(f"[POSITION_CLOSED] Error handling {symbol}: {e}")

    async def _run_ai_optimization_cycle(self) -> None:
        """Run AI optimization and learning cycle."""
        try:
            if not self.adaptive_learning:
                return

            # Update learning system with latest data
            if hasattr(self.adaptive_learning, "update_performance_metrics"):
                await self.adaptive_learning.update_performance_metrics()

            # Run advanced AI optimization if available
            if (
                hasattr(self.adaptive_learning, "advanced_ai")
                and self.adaptive_learning.advanced_ai
            ):
                # Get current market data for AI analysis
                market_data = {
                    "volatility": 1.0,
                    "trend_strength": 0.0,
                    "volume_trend": 1.0,
                    "price_change_24h": 0.0,
                }

                recommendations = (
                    await self.adaptive_learning.get_advanced_ai_recommendations(
                        market_data
                    )
                )

                if recommendations and recommendations.get("confidence", 0) > 0.5:
                    self.logger.info(
                        f"🧠 [AI_CYCLE] Applied recommendations with {recommendations['confidence']:.2f} confidence"
                    )

        except Exception as e:
            self.logger.debug(f"[AI_OPTIMIZATION] Failed: {e}")

    async def _check_emergency_stop_loss(self) -> bool:
        """
        Check if emergency stop loss has been triggered.

        Returns:
            True if emergency stop loss triggered, False otherwise
        """
        try:
            # Get current equity using Binance client
            if not hasattr(self, "client") or not self.client:
                # Client not initialized yet, skip check
                return False

            # BinanceClient.get_balance() is synchronous
            current_equity = float(self.client.get_balance())

            # Set initial equity on first run
            if self.initial_equity is None:
                self.initial_equity = current_equity
                self.logger.info(f"📊 Initial equity set: ${current_equity:.2f}")
                return False

            # Calculate loss percentage
            if self.initial_equity > 0:
                loss_pct = (
                    (current_equity - self.initial_equity) / self.initial_equity
                ) * 100

                # Log equity status every iteration (for monitoring)
                if self.iteration % 60 == 0:  # Every ~60 seconds
                    self.logger.info(
                        f"💰 Equity: ${current_equity:.2f} (Initial: ${self.initial_equity:.2f}, "
                        f"Change: {loss_pct:+.2f}%)"
                    )

                # Check if emergency threshold breached
                if loss_pct <= -self.emergency_stop_loss_pct:
                    self.logger.critical(
                        f"🚨🚨🚨 EMERGENCY STOP LOSS TRIGGERED! 🚨🚨🚨\n"
                        f"Initial Equity: ${self.initial_equity:.2f}\n"
                        f"Current Equity: ${current_equity:.2f}\n"
                        f"Loss: {loss_pct:.2f}% (Threshold: -{self.emergency_stop_loss_pct}%)\n"
                        f"🛑 HALTING ALL TRADING IMMEDIATELY!"
                    )

                    # Send Telegram alert if available
                    if self.telegram:
                        try:
                            await self.telegram.send_alert(
                                f"🚨🚨🚨 EMERGENCY STOP LOSS TRIGGERED! 🚨🚨🚨\n\n"
                                f"Initial: ${self.initial_equity:.2f}\n"
                                f"Current: ${current_equity:.2f}\n"
                                f"Loss: {loss_pct:.2f}%\n\n"
                                f"🛑 Bot halted automatically!"
                            )
                        except Exception as tg_e:
                            self.logger.error(f"Failed to send Telegram alert: {tg_e}")

                    return True

            return False

        except Exception as e:
            self.logger.error(
                f"[EMERGENCY_SL] Failed to check emergency stop loss: {e}"
            )
            return False

    async def _initialize_advanced_ai(self) -> None:
        """Initialize advanced AI features for the trading bot."""
        try:
            if not self.adaptive_learning or not self.adaptive_learning.advanced_ai:
                logger.debug(
                    "🧠 [ADVANCED_AI] Not available - basic adaptive learning only"
                )
                return

            logger.info(
                "🧠 [ADVANCED_AI] Initializing advanced intelligence features..."
            )

            # Get AI recommendations based on current market data
            market_data = {
                "volatility": getattr(self.config, "volatility_factor", 1.0),
                "trend_strength": 0.0,  # Will be updated with real data
                "volume_trend": 1.0,
                "price_change_24h": 0.0,
            }

            recommendations = (
                await self.adaptive_learning.get_advanced_ai_recommendations(
                    market_data
                )
            )

            if recommendations and recommendations.get("confidence", 0) > 0.5:
                logger.info(
                    f"🎯 [AI_RECOMMENDATIONS] Confidence: {recommendations['confidence']:.2f}"
                )

                for param, value in recommendations.get("recommendations", {}).items():
                    if param == "confidence_threshold":
                        logger.info(f"🎯 [AI_TUNING] Confidence threshold: {value:.3f}")
                    elif param == "position_size_multiplier":
                        adjusted_risk = self.risk_pct * value
                        self.risk_pct = max(0.1, min(2.0, adjusted_risk))
                        logger.info(
                            f"🎯 [AI_TUNING] Position sizing: {self.risk_pct:.2f}%"
                        )

                reasoning = recommendations.get("reasoning", [])
                if reasoning:
                    logger.info(f"🧠 [AI_REASONING] Applied: {', '.join(reasoning)}")

            # Start advanced A/B testing if configured
            ab_variants = getattr(self.config, "advanced_ab_variants", None)
            if ab_variants:
                await self._start_advanced_ab_testing(ab_variants)

            # Run parameter optimization periodically
            if (
                hasattr(self.adaptive_learning, "trades_history")
                and len(self.adaptive_learning.trades_history) >= 20
            ):
                await self._run_periodic_optimization()

            logger.info("🧠 [ADVANCED_AI] Initialization complete")

        except Exception as e:
            logger.error(f"❌ [ADVANCED_AI] Initialization failed: {e}")

    async def _update_enhanced_dashboard(self) -> None:
        """Update the enhanced dashboard with current trading data."""
        try:
            if not self.dashboard:
                return

            # Update dashboard with current trading engine and adaptive learning data
            dashboard_path = await self.dashboard.update_dashboard(
                trading_engine=self, adaptive_learning=self.adaptive_learning
            )

            if dashboard_path:
                # Open dashboard in browser on first update
                if not hasattr(self, "_dashboard_opened"):
                    try:
                        import webbrowser
                        from pathlib import Path

                        abs_path = Path(dashboard_path).resolve()
                        file_url = f"file://{abs_path}"

                        webbrowser.open(file_url)
                        self._dashboard_opened = True

                        self.logger.info(
                            "📊 [ENHANCED_DASHBOARD] Dashboard opened in browser: %s",
                            file_url,
                        )
                    except Exception as browser_e:
                        self.logger.warning(
                            "📊 [ENHANCED_DASHBOARD] Failed to open browser: %s",
                            browser_e,
                        )
                        self.logger.info(
                            "📊 [ENHANCED_DASHBOARD] Manual open: %s", dashboard_path
                        )

                self.logger.debug(
                    "📊 [ENHANCED_DASHBOARD] Updated successfully: %s", dashboard_path
                )
            else:
                self.logger.warning(
                    "📊 [ENHANCED_DASHBOARD] Update returned empty path"
                )

        except Exception as e:
            self.logger.warning("📊 [ENHANCED_DASHBOARD] Update failed: %s", e)

    async def _start_advanced_ab_testing(self, variants: List[Dict]) -> None:
        """Start advanced A/B testing with multiple variants."""
        try:
            if not self.adaptive_learning:
                return

            logger.info(
                f"🧪 [ADVANCED_AB] Starting testing with {len(variants)} variants"
            )

            result = await self.adaptive_learning.start_advanced_ab_testing(variants)

            if result.get("test_started"):
                logger.info(
                    f"🧪 [ADVANCED_AB] Test started: {result['current_allocation']} allocation"
                )
                logger.info(
                    f"🧪 [ADVANCED_AB] Min trades required: {result['min_trades_required']}"
                )

        except Exception as e:
            logger.error(f"❌ [ADVANCED_AB] Failed to start testing: {e}")

    async def _run_periodic_optimization(self) -> None:
        """Run Bayesian optimization on trading parameters."""
        try:
            if not self.adaptive_learning:
                return

            logger.info(
                "🎯 [AI_OPTIMIZATION] Running Bayesian parameter optimization..."
            )

            optimal_params = await self.adaptive_learning.optimize_parameters_with_ai(
                "sharpe_ratio"
            )

            if optimal_params:
                logger.info(
                    "🎯 [AI_OPTIMIZATION] Optimization complete - parameters updated"
                )

                # Apply optimized parameters
                if "confidence_threshold" in optimal_params:
                    logger.info(
                        f"🎯 [OPTIMIZED] Confidence: {optimal_params['confidence_threshold']:.3f}"
                    )

                if "position_size_multiplier" in optimal_params:
                    adjusted_risk = (
                        self.risk_pct * optimal_params["position_size_multiplier"]
                    )
                    self.risk_pct = max(0.1, min(2.0, adjusted_risk))
                    logger.info(f"🎯 [OPTIMIZED] Position sizing: {self.risk_pct:.2f}%")

        except Exception as e:
            logger.error(f"❌ [AI_OPTIMIZATION] Failed: {e}")

    async def _generate_learning_visualization(self) -> None:
        """Generate learning visualization and reports."""
        try:
            if not self.adaptive_learning or not hasattr(
                self.adaptive_learning, "learning_visualizer"
            ):
                return

            visualizer = self.adaptive_learning.learning_visualizer
            if not visualizer:
                return

            # Capture current learning snapshot
            snapshot = await visualizer.capture_learning_snapshot(
                adaptive_learning_system=self.adaptive_learning,
                trading_engine=self,
                iteration=self.iteration,
            )

            # Generate real-time report every minute
            await visualizer.generate_real_time_report(snapshot)

            # Generate HTML dashboard every 10 minutes
            if self.iteration % 600 == 0:  # Every 10 minutes
                dashboard_path = await visualizer.create_learning_dashboard()
                if dashboard_path:
                    self.logger.info(
                        f"📊 [DASHBOARD] Updated learning dashboard: {dashboard_path}"
                    )

            # Log learning status for immediate visibility
            if snapshot.adaptations_count > 0:
                self.logger.info(
                    "📊 [LEARNING_STATUS] ========================================"
                )
                self.logger.info(f"🔧 Total Adaptations: {snapshot.adaptations_count}")
                self.logger.info(
                    f"🎯 Current Confidence: {snapshot.confidence_threshold:.3f}"
                )
                self.logger.info(
                    f"📏 Position Multiplier: {snapshot.position_size_multiplier:.2f}x"
                )
                self.logger.info(
                    f"🔄 DCA Enabled: {'✅' if snapshot.dca_enabled else '❌'}"
                )

                if snapshot.total_trades > 0:
                    self.logger.info(
                        f"📈 Performance: {snapshot.total_trades} trades, {snapshot.win_rate:.1%} WR, ${snapshot.total_pnl:+.2f} PnL"
                    )

                self.logger.info(
                    "📊 [LEARNING_STATUS] ========================================"
                )

        except Exception as e:
            self.logger.debug(f"[LEARNING_VIZ] Failed to generate visualization: {e}")


# --- Entry point ----------------------------------------------------------------------


async def run_live_trading(config: Config) -> None:
    engine = LiveTradingEngine(config)
    try:
        await engine.start()
    finally:
        await engine.stop()
