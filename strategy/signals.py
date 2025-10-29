
# --- INPUT ADAPTER (idempotent) ---
def _coerce_market_input(symbol, market_data):
    """
    Превращаем разношёрстный market_data в (price, klines_like) без падений.
    price: float|None
    klines_like: list[(ts,o,h,l,c)]|None
    """
    sym = str(symbol or "UNKNOWN").upper()
    price = None
    klike = None

    try:
        # dict c 'price' или last
        if isinstance(market_data, dict):
            for k in ("price","last","last_price","close","c"):
                if k in market_data:
                    price = float(market_data[k]); break
            if "kline" in market_data and isinstance(market_data["kline"], (list,tuple)):
                kl = market_data["kline"]
                if kl and len(kl[0])>=5:
                    klike = [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in kl]
        # list свечей
        elif isinstance(market_data, (list,tuple)) and market_data:
            # может быть список свечей [ [ts,o,h,l,c,...], ... ]
            if isinstance(market_data[0], (list,tuple)) and len(market_data[0])>=5:
                try:
                    klike = [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in market_data]
                    price = float(klike[-1][4])
                except Exception:
                    pass
        # одиночные типы
        elif isinstance(market_data, (int,float)):
            price = float(market_data)
        elif isinstance(market_data, str):
            try: price = float(market_data)
            except Exception: pass
    except Exception:
        # не роняем генератор — просто возвращаем None
        pass

    return price, klike
# --- /INPUT ADAPTER ---

"""
Trading signal generation - Market Data Compatible Version

This version properly handles market data from the trading engine
and generates signals based on REAL prices, not synthetic ones.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass

import numpy as np

from core.config import Config

# LSTM Integration
try:
    from models.lstm import LSTMPredictor  # type: ignore
except ImportError:
    LSTMPredictor = None  # type: ignore
try:
    from core.types import MarketData
except ImportError:
    # If MarketData doesn't exist, create a simple version
    @dataclass
    class MarketData:
        symbol: str
        timestamp: List[datetime]
        open: List[float]
        high: List[float]
        low: List[float]
        close: List[float]
        volume: List[float]

try:
    from core.constants import SignalType
except ImportError:
    # If SignalType doesn't exist, create it
    from enum import Enum
    class SignalType(Enum):
        BUY = "BUY"
        SELL = "SELL"
        HOLD = "HOLD"

# Create our own TradingSignal class since it's missing from core.types
@dataclass
class TradingSignal:
    """Trading signal data structure."""
    symbol: str
    signal_type: SignalType
    strength: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SignalGenerator:
    """Generates trading signals based on REAL market data."""
    
    def __init__(self, config: Config):
        """Initialize signal generator with AGGRESSIVE configuration and LSTM integration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Check if IMBA signals should be used
        self.use_imba = getattr(config, "use_imba_signals", False)
        self.imba_integration = None
        
        if self.use_imba:
            try:
                from strategy.imba_integration import IMBASignalIntegration
                self.imba_integration = IMBASignalIntegration(config)
                self.logger.info("IMBA Research Signals ENABLED (9 signals + regime detection)")
                self.logger.info(f"  Min confidence: {config.bt_conf_min}")
                self.logger.info(f"  Signal aggregation with voting system")
            except Exception as e:
                self.logger.error(f"Failed to initialize IMBA integration: {e}")
                self.use_imba = False
        
        # ULTRA AGGRESSIVE Signal parameters - GUARANTEED to generate signals
        self.fast_ma_period = 5      # Very fast MA
        self.slow_ma_period = 10     # Very slow MA  
        self.min_signal_strength = 0.01  # Only 1% strength needed!
        
        # LSTM Integration for enhanced signal generation
        self.lstm_predictor = None
        if LSTMPredictor and getattr(config, "lstm_enable", False):
            try:
                self.lstm_predictor = LSTMPredictor(config)  # type: ignore
                self.logger.info("LSTM predictor integrated for enhanced signal generation")
            except Exception as e:
                self.logger.warning("LSTM predictor initialization failed: %s", e)
                self.lstm_predictor = None
        
        # State tracking
        self.last_signal: Optional[TradingSignal] = None
        self.last_signal_time: Optional[datetime] = None
        self.signal_count = 0
        
        # Price history for LSTM predictions
        self.price_history: List[float] = []
        self.max_history_length = getattr(config, "lstm_input", 16) * 2  # Keep 2x LSTM input length
        
        # Historical data cache for IMBA signals (accumulates candles over time)
        self._historical_data: Dict[str, List] = {}
        
        # Signal cooldown tracking per symbol (prevents spamming)
        self._last_signal_time: Dict[str, datetime] = {}
        self._signal_cooldown_seconds = getattr(config, "signal_cooldown_seconds", 60)  # 60 seconds default
        
        # Windows-compatible logging (no emoji)
        if not self.use_imba:
            lstm_status = "WITH LSTM" if self.lstm_predictor else "WITHOUT LSTM"
            self.logger.info(f"ULTRA AGGRESSIVE SignalGenerator initialized (MARKET DATA COMPATIBLE) {lstm_status}")
            self.logger.info(f"Min signal strength: {self.min_signal_strength}")
            self.logger.info(f"MA periods: {self.fast_ma_period}/{self.slow_ma_period}")
            if self.lstm_predictor:
                self.logger.info(f"LSTM input length: {getattr(config, 'lstm_input', 16)}")
                self.logger.info(f"LSTM signal threshold: {getattr(config, 'lstm_signal_threshold', 0.0015)}")
            self.logger.info("Ready to process REAL market data from API with AI enhancement")
    
    async def initialize(self) -> None:
        """Initialize the signal generator - ASYNC VERSION."""
        self.logger.info("Signal generator initialized for REAL API TRADING")
    
    def generate_signal(self, symbol: str, market_data, config=None) -> Optional[TradingSignal]:
        """
        Generate trading signals based on REAL market data.
        
        Args:
            symbol: Trading symbol (BTCUSDT, ETHUSDT, etc.)
            market_data: Market data (DataFrame, dict, or MarketData object)
            config: Optional config override
        """
        self.signal_count += 1

        # 🔍 DIAGNOSTIC: Log what data we're receiving
        if isinstance(market_data, list) and market_data:
            self.logger.debug(f"[SIGNAL_DATA] {symbol}: Received {len(market_data)} candles, latest close: {market_data[-1].get('close', 'N/A')}")
        elif isinstance(market_data, dict):
            self.logger.debug(f"[SIGNAL_DATA] {symbol}: Received dict with keys: {list(market_data.keys())}")
        else:
            self.logger.debug(f"[SIGNAL_DATA] {symbol}: Received {type(market_data)}")

        # 🎯 IMBA Signal Generation - if enabled, use advanced multi-signal aggregation
        if self.use_imba and self.imba_integration:
            try:
                # DIAGNOSTIC: Log confidence threshold being used
                aggregator = self.imba_integration.aggregator
                actual_threshold = getattr(aggregator, 'min_confidence', 'unknown')
                self.logger.debug(f"[IMBA_THRESHOLD] {symbol}: Using min_confidence={actual_threshold} (config.bt_conf_min={self.config.bt_conf_min})")
                
                return self._generate_imba_signal(symbol, market_data)
            except Exception as e:
                self.logger.error(f"IMBA signal generation failed for {symbol}: {e}, falling back to default")
                # Fall through to default signal generation

        # ✅ Новый блок: обработка если пришел список
        if isinstance(market_data, list):
            if not market_data:
                self.logger.warning(f"Signal attempt #{self.signal_count}: market_data list is empty")
                return None
            
            # 🔧 FIX: Check if list contains prices (floats/ints) or market_data objects
            first_item = market_data[0]
            
            # If list contains numbers (prices), treat as price array
            if isinstance(first_item, (int, float)):
                self.logger.debug(f"Signal attempt #{self.signal_count}: received price list ({len(market_data)} prices)")
                # Convert to dict format that we can process
                market_data = {
                    'close': market_data,
                    'timestamp': [datetime.now() for _ in market_data],
                    'symbol': symbol
                }
                # Continue to normal processing below (don't return, fall through)
            
            # If list contains dicts (OHLCV candles), aggregate them
            elif isinstance(first_item, dict):
                self.logger.debug(f"Signal attempt #{self.signal_count}: received list of candle dicts ({len(market_data)} items)")
                
                # Try to extract symbol from first item
                extracted_symbol = first_item.get('symbol', symbol) if 'symbol' in first_item else symbol
                
                # Check if dicts contain OHLCV data (candles)
                if 'close' in first_item:
                    # Extract all prices from the list
                    try:
                        prices = []
                        timestamps = []
                        
                        for item in market_data:
                            if isinstance(item, dict) and 'close' in item:
                                close_val = item.get('close')
                                if isinstance(close_val, (int, float)):
                                    prices.append(float(close_val))
                                    # Try to get timestamp
                                    ts = item.get('timestamp', item.get('time', datetime.now()))
                                    if isinstance(ts, (int, float)):
                                        ts = datetime.fromtimestamp(ts / 1000 if ts > 1e10 else ts)
                                    timestamps.append(ts)
                        
                        if len(prices) > 0:
                            # Convert aggregated data to processable format
                            market_data = {
                                'close': prices,
                                'timestamp': timestamps if timestamps else [datetime.now() for _ in prices],
                                'symbol': extracted_symbol
                            }
                            # Fall through to normal processing
                            self.logger.debug(f"Aggregated {len(prices)} candles from dict list")
                        else:
                            self.logger.warning(f"No prices extracted from dict list")
                            return None
                            
                    except Exception as e:
                        self.logger.error(f"Error aggregating dict list: {e}")
                        return None
                else:
                    # Dicts without 'close' - unknown format
                    self.logger.warning(f"Dict list without 'close' field - unknown format")
                    return None
            
            # If list contains complex objects with __dict__, try each one
            elif hasattr(first_item, '__dict__') and not isinstance(first_item, type):
                self.logger.debug(f"Signal attempt #{self.signal_count}: received list of complex objects ({len(market_data)} items)")
                for idx, item in enumerate(market_data):
                    try:
                        if hasattr(item, '__dict__') and not isinstance(item, (int, float, str, type)):
                            signal = self.generate_signal(symbol, item, config)
                            if signal:
                                self.logger.debug(f"Signal generated from object[{idx}]")
                                return signal
                    except Exception as e:
                        self.logger.warning(f"Error processing object[{idx}]: {e}")
                self.logger.warning(f"Signal attempt #{self.signal_count}: no valid signal from object list")
                return None
            else:
                # Unknown list type, log and skip
                self.logger.warning(f"Signal attempt #{self.signal_count}: unknown list type, first item: {type(first_item)}")
                return None
        
        
        # Log what we received for debugging
        if market_data is None:
            self.logger.warning(f"Signal attempt #{self.signal_count}: market_data is None!")
            return self._generate_fallback_signal()
        
        # Extract symbol first to improve logging
        symbol = 'UNKNOWN'
        try:
            if hasattr(market_data, 'symbol'):
                symbol = market_data.symbol
            elif isinstance(market_data, dict):
                symbol = market_data.get('symbol', 'UNKNOWN')
            else:
                self.logger.warning(f"Signal attempt #{self.signal_count}: Unexpected market_data type: {type(market_data)}")
        except Exception as e:
            self.logger.warning(f"Error extracting symbol: {e}")
        
        self.logger.debug(f"Signal attempt #{self.signal_count} for {symbol}")
        
        # Handle different market_data structures - PREFER REAL DATA
        prices = None
        timestamps = None
        
        try:
            if hasattr(market_data, 'close') and hasattr(market_data, 'timestamp'):
                # This looks like real MarketData object
                prices = market_data.close
                timestamps = market_data.timestamp
                self.logger.debug(f"Using real MarketData: {len(prices)} prices for {symbol}")
                
            elif isinstance(market_data, dict):
                # Dictionary format market data
                prices = market_data.get('close', [])
                timestamps = market_data.get('timestamp', [])
                
                # 🔧 FIX: Check if 'close' is a single value (float) or list
                if isinstance(prices, (int, float)):
                    # Single price - convert to list
                    prices = [float(prices)]
                    if isinstance(timestamps, datetime):
                        timestamps = [timestamps]
                    elif not timestamps:
                        timestamps = [datetime.now()]
                
                if prices and isinstance(prices, (list, tuple)):
                    self.logger.debug(f"Using dict market data: {len(prices)} prices for {symbol}")
                
            elif hasattr(market_data, '__dict__'):
                # Try to extract from any object with attributes
                attrs = vars(market_data)
                prices = attrs.get('close', attrs.get('prices', []))
                timestamps = attrs.get('timestamp', attrs.get('time', []))
                if prices:
                    self.logger.debug(f"Using object attributes: {len(prices)} prices for {symbol}")
            
            if not prices:
                self.logger.info(f"No price data available for {symbol}, checking fallback options")
                # Only use fallback for demo after trying real data
                if symbol == 'UNKNOWN' and self.signal_count % 5 == 0:
                    return self._generate_fallback_signal()
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting market data: {e}")
            return None
        
        # Validate we have enough data
        if not prices or len(prices) < max(self.slow_ma_period, 3):
            self.logger.debug(f"Insufficient price data for {symbol}: need {max(self.slow_ma_period, 3)}, got {len(prices)}")
            return None
            
        try:
            # Use recent prices for calculation
            recent_prices = list(prices[-20:]) if len(prices) >= 20 else list(prices)
            # Ensure numerical values are floats (market data can provide Decimal)
            recent_prices = [float(p) for p in recent_prices]
            current_timestamp = timestamps[-1] if timestamps and len(timestamps) > 0 else datetime.now()
            
            # Calculate moving averages with real data
            if len(recent_prices) >= self.slow_ma_period:
                fast_ma = np.mean(recent_prices[-self.fast_ma_period:])
                slow_ma = np.mean(recent_prices[-self.slow_ma_period:])
            else:
                # Use what we have
                fast_ma = np.mean(recent_prices[-min(self.fast_ma_period, len(recent_prices)):])
                slow_ma = np.mean(recent_prices)
            
            current_price = recent_prices[-1]
            
            # Update price history for LSTM
            self._update_price_history(recent_prices)
            
            # Log REAL price data (not synthetic!)
            self.logger.debug(f"REAL PRICES - {symbol}: fast_ma={fast_ma:.4f}, slow_ma={slow_ma:.4f}, current={current_price:.4f}")
            
            # Check cooldown period
            if self._is_in_cooldown(current_timestamp):
                self.logger.debug(f"Still in cooldown for {symbol}")
                return None
            
            # ULTRA SENSITIVE signal detection using REAL prices
            signal_type = None
            strength = 0.0
            
            # Calculate MA difference percentage
            ma_diff_pct = abs(fast_ma - slow_ma) / slow_ma if slow_ma > 0 else 0.0

            if fast_ma > slow_ma:
                signal_type = SignalType.BUY
                strength = min(0.9, 0.5 + float(ma_diff_pct) * 100)  # At least 50% strength

            elif fast_ma < slow_ma:
                signal_type = SignalType.SELL
                strength = min(0.9, 0.5 + float(ma_diff_pct) * 100)  # At least 50% strength
            
            # Secondary signal: price momentum with real data
            elif len(recent_prices) >= 2:
                price_change = (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2]
                if abs(price_change) > 0.0001:  # 0.01% movement
                    signal_type = SignalType.BUY if price_change > 0 else SignalType.SELL
                    strength = min(0.85, 0.6 + abs(price_change) * 1000)
            
            # LSTM Enhancement: Improve signal with AI predictions
            lstm_enhanced = False
            if self.lstm_predictor and len(self.price_history) >= getattr(self.config, "lstm_input", 16):
                try:
                    # Get LSTM prediction
                    lstm_prediction = self._get_lstm_prediction()
                    if lstm_prediction is not None:
                        # Combine traditional signal with LSTM prediction
                        original_strength = strength
                        strength = self._enhance_signal_with_lstm(signal_type, strength, lstm_prediction)
                        lstm_enhanced = True
                        self.logger.debug(f"LSTM enhanced signal for {symbol}: {original_strength:.3f} -> {strength:.3f} (prediction: {lstm_prediction:.4f})")
                except Exception as e:
                    self.logger.warning(f"LSTM enhancement failed for {symbol}: {e}")
            
            # Check minimum strength (very low threshold)
            if not signal_type or strength < self.min_signal_strength:
                self.logger.debug(f"Signal rejected for {symbol}: type={signal_type}, strength={strength:.3f}, min_req={self.min_signal_strength}")
                return None
                
            # Create trading signal with REAL data and LSTM enhancement
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                timestamp=current_timestamp,
                metadata={
                    'fast_ma': float(fast_ma),
                    'slow_ma': float(slow_ma),
                    'current_price': float(current_price),
                    'ma_diff_pct': float(ma_diff_pct) * 100,
                    'strategy': 'REAL_MARKET_DATA',
                    'signal_attempt': self.signal_count,
                    'data_points': len(recent_prices)
                }
            )
            
            self.last_signal = signal
            self.last_signal_time = current_timestamp
            
            # Log successful signal generation with REAL price
            self.logger.info(f"GENERATED {signal_type.value} signal for {symbol} "
                           f"(strength: {strength:.2f}, REAL_PRICE: {current_price:.4f}, attempt: #{self.signal_count})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error processing real market data for {symbol}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _generate_fallback_signal(self) -> Optional[TradingSignal]:
        """Generate a fallback signal ONLY when real data is unavailable."""
        try:
            # Only use as last resort
            if self._is_in_cooldown(datetime.now()):
                return None
                
            signal_type = SignalType.BUY if (self.signal_count % 2) == 0 else SignalType.SELL
            
            # Use current BTC price range for more realistic fallback
            base_price = 67000.0  # Approximate current BTC price
            price_variation = (self.signal_count % 200) - 100  # ±100 variation
            fallback_price = base_price + price_variation
            
            signal = TradingSignal(
                symbol='BTCUSDT',
                signal_type=signal_type,
                strength=0.3,  # Lower strength for fallback signals
                timestamp=datetime.now(),
                metadata={
                    'fallback': True,
                    'fallback_price': fallback_price,
                    'strategy': 'FALLBACK_ONLY',
                    'signal_attempt': self.signal_count,
                    'note': 'Generated when real market data unavailable'
                }
            )
            
            self.last_signal = signal
            self.last_signal_time = datetime.now()
            
            # Clear indication this is fallback
            self.logger.warning(f"FALLBACK {signal_type.value} signal generated "
                              f"(strength: 0.3, fallback_price: {fallback_price:.2f}) - REAL DATA PREFERRED!")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Even fallback signal generation failed: {e}")
            return None
    
    def _is_in_cooldown(self, current_time: datetime) -> bool:
        """Check if we're still in cooldown period from last signal."""
        if not self.last_signal_time:
            return False
            
        cooldown_seconds = getattr(self.config, 'cooldown_sec', 10)  # Default 10 seconds
        
        # Handle timezone issues gracefully
        try:
            if current_time.tzinfo is None and self.last_signal_time.tzinfo is not None:
                current_time = current_time.replace(tzinfo=timezone.utc)
            elif current_time.tzinfo is not None and self.last_signal_time.tzinfo is None:
                self.last_signal_time = self.last_signal_time.replace(tzinfo=timezone.utc)
        except:
            pass  # Ignore timezone errors
        
        try:
            time_since_last = (current_time - self.last_signal_time).total_seconds()
        except:
            return False  # If time calculation fails, don't block
        
        is_cooling = time_since_last < cooldown_seconds
        if is_cooling:
            self.logger.debug(f"Cooldown: {time_since_last:.1f}s < {cooldown_seconds}s")
        
        return is_cooling
    
    def get_signal_summary(self) -> dict:
        """Get summary of signal generator state."""
        return {
            'fast_ma_period': self.fast_ma_period,
            'slow_ma_period': self.slow_ma_period,
            'min_signal_strength': self.min_signal_strength,
            'signal_count': self.signal_count,
            'market_data_compatible': True,
            'prefers_real_data': True,
            'lstm_enabled': self.lstm_predictor is not None,
            'price_history_length': len(self.price_history),
            'last_signal': {
                'type': self.last_signal.signal_type.value if self.last_signal else None,
                'strength': self.last_signal.strength if self.last_signal else None,
                'timestamp': self.last_signal.timestamp.isoformat() if self.last_signal else None,
                'was_fallback': self.last_signal.metadata.get('fallback', False) if self.last_signal else False
            } if self.last_signal else None
        }

    def _update_price_history(self, recent_prices: List[float]) -> None:
        """Update price history for LSTM predictions."""
        try:
            # Add recent prices to history
            self.price_history.extend(recent_prices)
            
            # Keep only the most recent prices (don't let history grow too large)
            if len(self.price_history) > self.max_history_length:
                self.price_history = self.price_history[-self.max_history_length:]
                
            self.logger.debug(f"Price history updated: {len(self.price_history)} prices (max: {self.max_history_length})")
            
        except Exception as e:
            self.logger.warning(f"Failed to update price history: {e}")

    def _get_lstm_prediction(self) -> Optional[float]:
        """Get LSTM prediction for next price movement."""
        try:
            if not self.lstm_predictor or len(self.price_history) < getattr(self.config, "lstm_input", 16):
                return None
                
            # Prepare data for LSTM
            lstm_input_length = getattr(self.config, "lstm_input", 16)
            input_prices = self.price_history[-lstm_input_length:]
            
            # Get prediction from LSTM model
            prediction = self.lstm_predictor.predict(input_prices)
            
            if prediction is not None:
                self.logger.debug(f"LSTM prediction: {prediction:.6f} (input: {len(input_prices)} prices)")
                return float(prediction)
                
            return None
            
        except Exception as e:
            self.logger.warning(f"LSTM prediction failed: {e}")
            return None

    def _enhance_signal_with_lstm(self, signal_type: Optional[SignalType], 
                                  original_strength: float, lstm_prediction: float) -> float:
        """Enhance traditional signal strength with LSTM prediction."""
        try:
            if not signal_type or lstm_prediction is None:
                return original_strength
                
            # Get LSTM threshold from config
            lstm_threshold = getattr(self.config, "lstm_signal_threshold", 0.0015)  # 0.15% default
            
            # Determine if LSTM agrees with traditional signal
            lstm_bullish = lstm_prediction > lstm_threshold
            lstm_bearish = lstm_prediction < -lstm_threshold
            
            signal_bullish = (signal_type == SignalType.BUY)
            signal_bearish = (signal_type == SignalType.SELL)
            
            # Calculate enhancement factor
            enhancement_factor = 1.0
            
            if (signal_bullish and lstm_bullish) or (signal_bearish and lstm_bearish):
                # LSTM agrees - boost signal strength
                lstm_confidence = min(abs(lstm_prediction) / lstm_threshold, 3.0)  # Cap at 3x
                enhancement_factor = 1.0 + (lstm_confidence - 1.0) * 0.3  # Up to 30% boost
                
            elif (signal_bullish and lstm_bearish) or (signal_bearish and lstm_bullish):
                # LSTM disagrees - reduce signal strength
                lstm_confidence = min(abs(lstm_prediction) / lstm_threshold, 2.0)  # Cap at 2x
                enhancement_factor = max(0.3, 1.0 - lstm_confidence * 0.2)  # Up to 40% reduction, min 30%
                
            # Apply enhancement
            enhanced_strength = original_strength * enhancement_factor
            
            # Ensure within valid range
            enhanced_strength = max(0.01, min(0.95, enhanced_strength))
            
            self.logger.debug(f"LSTM enhancement: {original_strength:.3f} * {enhancement_factor:.3f} = {enhanced_strength:.3f}")
            
            return enhanced_strength
            
        except Exception as e:
            self.logger.warning(f"Signal enhancement failed: {e}")
            return original_strength
    def _generate_imba_signal(self, symbol: str, market_data) -> Optional[TradingSignal]:
        """
        Generate IMBA trading signal using advanced multi-signal aggregation.
        
        Args:
            symbol: Trading symbol (BTCUSDT, ETHUSDT, etc.)
            market_data: Market data (DataFrame, dict, or MarketData object)
        
        Returns TradingSignal compatible with existing engine.
        """
        try:
            # Convert market_data to DataFrame if needed
            import pandas as pd
            
            df = None
            
            # Extract symbol from multiple sources (override if provided in data)
            if hasattr(market_data, 'symbol'):
                symbol = market_data.symbol
            elif isinstance(market_data, dict) and 'symbol' in market_data:
                symbol = market_data.get('symbol', symbol)
            
            # Convert incoming market_data to DataFrame
            new_df = None
            if hasattr(market_data, 'close') and hasattr(market_data, 'timestamp'):
                # MarketData object
                new_df = pd.DataFrame({
                    'timestamp': market_data.timestamp,
                    'open': market_data.open,
                    'high': market_data.high,
                    'low': market_data.low,
                    'close': market_data.close,
                    'volume': market_data.volume
                })
                if not new_df.empty:
                    new_df.set_index('timestamp', inplace=True)
                
            elif isinstance(market_data, dict) and 'close' in market_data:
                # Dictionary with OHLCV data
                new_df = pd.DataFrame(market_data)
                if not new_df.empty and 'timestamp' in new_df.columns:
                    new_df.set_index('timestamp', inplace=True)
            
            elif isinstance(market_data, list) and market_data and isinstance(market_data[0], dict):
                # List of candle dictionaries - convert to DataFrame
                self.logger.debug(f"Converting list of {len(market_data)} candle dicts to DataFrame for IMBA")
                try:
                    # Convert list of dicts to DataFrame
                    rows = []
                    for candle in market_data:
                        if isinstance(candle, dict) and 'close' in candle:
                            # Extract timestamp
                            ts = candle.get('timestamp', candle.get('time', pd.Timestamp.now()))
                            if isinstance(ts, (int, float)):
                                # Convert timestamp to datetime
                                ts = pd.to_datetime(ts, unit='ms' if ts > 1e10 else 's')
                            elif isinstance(ts, str):
                                ts = pd.to_datetime(ts)
                            
                            rows.append({
                                'timestamp': ts,
                                'open': float(candle.get('open', candle.get('close', 0))),
                                'high': float(candle.get('high', candle.get('close', 0))),
                                'low': float(candle.get('low', candle.get('close', 0))),
                                'close': float(candle.get('close', 0)),
                                'volume': float(candle.get('volume', 0))
                            })
                    
                    if rows:
                        new_df = pd.DataFrame(rows)
                        new_df.set_index('timestamp', inplace=True)
                        new_df.sort_index(inplace=True)
                        self.logger.debug(f"✅ Converted to DataFrame: {len(new_df)} candles, range: {new_df.index[0]} → {new_df.index[-1]}")
                    else:
                        self.logger.warning(f"No valid candles found in list")
                        
                except Exception as e:
                    self.logger.error(f"Error converting candle list to DataFrame: {e}")
                    return None
            
            elif isinstance(market_data, pd.DataFrame):
                # Already a DataFrame
                new_df = market_data
            
            # CRITICAL: Merge with historical data cache
            if symbol in self._historical_data and len(self._historical_data[symbol]) > 0:
                # We have preloaded historical data - merge with new data
                cached_df = self._historical_data[symbol]
                
                if isinstance(cached_df, pd.DataFrame):
                    if new_df is not None and not new_df.empty:
                        # Combine cached + new data, remove duplicates
                        df = pd.concat([cached_df, new_df])
                        df = df[~df.index.duplicated(keep='last')]  # Keep newest data
                        df = df.sort_index()
                        
                        # Keep only last 1200 candles to prevent memory issues (FVG needs 1200)
                        if len(df) > 1200:
                            df = df.iloc[-1200:]
                        
                        # Update cache with merged data
                        self._historical_data[symbol] = df
                        
                        self.logger.debug(f"IMBA using cached+new data: {len(df)} total candles for {symbol}")
                    else:
                        # Use cached data only
                        df = cached_df
                        self.logger.debug(f"IMBA using cached data: {len(df)} candles for {symbol}")
                else:
                    # Cache exists but wrong format, use new data
                    df = new_df
            else:
                # No cache yet, use new data and store it
                df = new_df
                if df is not None and not df.empty:
                    self._historical_data[symbol] = df
                    self.logger.debug(f"IMBA initialized cache with {len(df)} candles for {symbol}")
            
            if df is None or len(df) < 250:
                self.logger.debug(f"Insufficient data for IMBA: {len(df) if df is not None else 0} candles (need 250+)")
                return None
            
            # Generate IMBA signal
            imba_result = self.imba_integration.generate_signal_from_df(
                df=df,
                symbol=symbol,
                lstm_prediction=None,  # Could integrate LSTM here
                funding_rate=None,
                alt_prices=None
            )
            
            # Convert IMBA result to TradingSignal
            direction = imba_result.get('direction', 'wait')
            confidence = imba_result.get('confidence', 0.0)
            
            # CRITICAL: Check filters_passed and direction FIRST
            filters_passed = imba_result.get('filters_passed', True)
            
            if direction == 'wait' or confidence < self.config.bt_conf_min or not filters_passed:
                self.logger.debug(f"IMBA signal rejected: direction={direction}, confidence={confidence:.3f}, filters_passed={filters_passed}")
                return None
            
            # Check cooldown per symbol (WITH HIGH CONFIDENCE BYPASS!)
            now = datetime.now(timezone.utc)
            high_confidence_threshold = getattr(self.config, "high_confidence_threshold", 1.2)  # Default 1.2
            
            if symbol in self._last_signal_time:
                seconds_since_last = (now - self._last_signal_time[symbol]).total_seconds()
                
                # 🔥 COOLDOWN BYPASS: High confidence signals skip cooldown!
                if confidence >= high_confidence_threshold:
                    self.logger.info(f"⚡ COOLDOWN BYPASSED for {symbol}: High confidence {confidence:.2f} >= {high_confidence_threshold:.2f}")
                elif seconds_since_last < self._signal_cooldown_seconds:
                    self.logger.debug(f"IMBA signal in cooldown for {symbol}: {seconds_since_last:.0f}s < {self._signal_cooldown_seconds}s (confidence={confidence:.2f} < bypass threshold={high_confidence_threshold:.2f})")
                    return None
            
            # Map IMBA direction to SignalType
            if direction == 'buy':
                signal_type = SignalType.BUY
            elif direction == 'sell':
                signal_type = SignalType.SELL
            else:
                return None
            
            # Create TradingSignal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=confidence,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    'imba': True,
                    'regime': imba_result.get('regime', {}),
                    'signals': imba_result.get('signals', []),
                    'filters_passed': imba_result.get('filters_passed', True),
                    'strategy': 'IMBA_RESEARCH',
                    'signal_count': len(imba_result.get('signals', []))
                }
            )
            
            # Update last signal time for this symbol
            self._last_signal_time[symbol] = now
            
            self.logger.info(f"IMBA {signal_type.value} signal for {symbol} "
                           f"(confidence: {confidence:.2f}, regime: {imba_result.get('regime', {}).get('current', 'unknown')})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"IMBA signal generation error: {e}", exc_info=True)
            return None


# Compatibility class if needed
class SimpleScalper:
    """Minimal scalper for compatibility."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> None:
        """Async initialize for compatibility."""
        self.logger.info("SimpleScalper initialized")
        
    def generate_signal(self, market_data) -> Optional[TradingSignal]:
        """Simple scalping signal generation."""
        return None  # Not implemented in compatibility mode
