"""
IMBA Signal Integration Wrapper.

Integrates IMBA research signals into the existing SignalGenerator.
When use_imba_signals=True, replaces default signal generation with IMBA aggregation.
"""

import pandas as pd
import logging
from typing import Optional, Dict, Any

from core.config import Config
from strategy.imba_signals import IMBASignalAggregator
from strategy.filters import FilterManager
from strategy.spread_filter import SpotFuturesSpreadFilter
from strategy.fear_greed_index import FearGreedIndex
from strategy.btc_dominance import BTCDominanceIndicator

logger = logging.getLogger(__name__)


class IMBASignalIntegration:
    """
    Wrapper to integrate IMBA signals into existing trading engine.
    
    Usage in SignalGenerator:
        if config.use_imba_signals:
            imba = IMBASignalIntegration(config)
            signal = imba.generate_signal_from_df(df, symbol)
    """
    
    def __init__(self, config: Config):
        """
        Initialize IMBA integration with enhanced error handling.
        
        Args:
            config: Trading configuration with IMBA parameters
        """
        self.config = config
        
        # Initialize with safe fallbacks
        try:
            # Get alt_influence safely with fallback
            alt_influence = getattr(config, 'alt_influence', 0.3)
            
            self.aggregator = IMBASignalAggregator(
                min_confidence=config.bt_conf_min,
                lstm_weight=0.35,
                alt_influence=alt_influence
            )
            logger.debug("✅ IMBASignalAggregator initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize IMBASignalAggregator: {e}")
            # Create a minimal fallback aggregator
            self.aggregator = None
        
        try:
            self.filter_manager = FilterManager(config)
            logger.debug("✅ FilterManager initialized")
        except Exception as e:
            logger.warning(f"⚠️ FilterManager init failed: {e}")
            self.filter_manager = None
        
        try:
            self.spread_filter = SpotFuturesSpreadFilter(
                min_spread_pct=3.0,      # Detect spreads >= 3%
                strong_spread_pct=5.0,   # Strong signal at 5%
                extreme_spread_pct=7.0   # Extreme signal at 7% (like ETH case)
            )
            logger.debug("✅ SpotFuturesSpreadFilter initialized")
        except Exception as e:
            logger.warning(f"⚠️ SpotFuturesSpreadFilter init failed: {e}")
            self.spread_filter = None
        
        try:
            self.fear_greed = FearGreedIndex()  # 🔥 Fear & Greed sentiment analysis
            logger.debug("✅ FearGreedIndex initialized")
        except Exception as e:
            logger.warning(f"⚠️ FearGreedIndex init failed: {e}")
            self.fear_greed = None
            
        try:
            self.btc_dominance = BTCDominanceIndicator()  # 🔥 BTC Dominance for altcoin timing
            logger.debug("✅ BTCDominanceIndicator initialized")
        except Exception as e:
            logger.warning(f"⚠️ BTCDominanceIndicator init failed: {e}")
            self.btc_dominance = None
        
        # Log initialization status
        logger.info("IMBA Signal Integration initialized")
        logger.info(f"  Min confidence: {config.bt_conf_min}")
        
        # Safe attribute access with fallbacks
        trend_adx = getattr(config, 'trend_adx_threshold', 25.0)
        flat_adx = getattr(config, 'flat_adx_threshold', 20.0)
        logger.info(f"  Trend ADX threshold: {trend_adx}")
        logger.info(f"  Flat ADX threshold: {flat_adx}")
        
        if self.spread_filter:
            logger.info(f"  Spot-Futures spread detection: ENABLED (min=3%, strong=5%, extreme=7%)")
        else:
            logger.warning("  Spot-Futures spread detection: DISABLED (init failed)")
        
        # Count successful initializations
        components = [
            self.aggregator, self.filter_manager, self.spread_filter,
            self.fear_greed, self.btc_dominance
        ]
        success_count = sum(1 for c in components if c is not None)
        logger.info(f"  Successfully initialized {success_count}/5 IMBA components")
    
    def generate_signal_from_df(
        self,
        df: pd.DataFrame,
        symbol: str,
        lstm_prediction: Optional[float] = None,
        funding_rate: Optional[float] = None,
        alt_prices: Optional[Dict[str, list]] = None
    ) -> Dict[str, Any]:
        """
        Generate trading signal using IMBA aggregation.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            lstm_prediction: Optional LSTM prediction value
            funding_rate: Optional funding rate for filtering
            alt_prices: Optional altcoin prices for market bias
            
        Returns:
            Dictionary with signal information:
            {
                'direction': 'buy' | 'sell' | 'wait',
                'strength': float (0.0-1.0),
                'confidence': float,
                'regime': dict,
                'signals': list,
                'filters_passed': bool,
                'metadata': dict
            }
        """
        # Clear cached adjustments from previous symbol
        self._fng_adjustment = None
        self._btcd_adjustment = None
        
        if len(df) < 250:
            logger.warning(f"Insufficient data for IMBA signals: {len(df)} candles (need >= 250)")
            return {
                'direction': 'wait',
                'strength': 0.0,
                'confidence': 0.0,
                'regime': {'kind': 'unknown'},
                'signals': [],
                'filters_passed': False,
                'metadata': {'error': 'Insufficient data'}
            }
        
        try:
            # 🔍 DIAGNOSTIC: Log candle data quality
            latest_candle = df.iloc[-1]
            earliest_candle = df.iloc[0]
            
            logger.debug(f"[IMBA_DATA] {symbol}: Using {len(df)} candles")
            logger.debug(f"[IMBA_DATA] {symbol}: Range {earliest_candle.name} → {latest_candle.name}")
            logger.debug(f"[IMBA_DATA] {symbol}: Latest OHLC: {latest_candle['open']:.2f}/{latest_candle['high']:.2f}/{latest_candle['low']:.2f}/{latest_candle['close']:.2f}")
            
            # Aggregate all IMBA signals with diagnostic info
            result = self.aggregator.aggregate(
                df=df,
                lstm_rel=lstm_prediction if lstm_prediction else 0.0,
                alt_bias=0.0,  # Will be calculated by filters
                lstm_threshold=self.config.lstm_signal_threshold
            )
            
            # 🔍 DIAGNOSTIC: Log voting results for debugging  
            votes = result.get('votes', {})
            signal_details = result.get('signal_details', [])
            logger.debug(f"[IMBA_VOTES] {symbol}: BUY={votes.get('buy', 0):.2f}, SELL={votes.get('sell', 0):.2f}")
            logger.debug(f"[IMBA_SIGNALS] {symbol}: {', '.join(signal_details[:5])}...")  # Show first 5 signals
            
            # 🔥 Apply Fear & Greed Index adjustment
            fng_multipliers = self.fear_greed.get_signal_multiplier()
            original_confidence = result['confidence']
            
            if result['direction'] == 'buy':
                result['confidence'] *= fng_multipliers['buy']
            elif result['direction'] == 'sell':
                result['confidence'] *= fng_multipliers['sell']
            
            # Log Fear & Greed adjustment if applied
            if original_confidence != result['confidence']:
                logger.info(f"[FEAR_GREED] {symbol}: {result['direction'].upper()} confidence adjusted: "
                          f"{original_confidence:.2f} → {result['confidence']:.2f} "
                          f"(Fear & Greed: {fng_multipliers.get('value', 'N/A')}/100, {fng_multipliers.get('reason', 'unknown')})")
            
            # Cache FNG for display (avoid duplicate API calls later)
            self._cached_fng_display = self.fear_greed.get_display_string()
            self._fng_adjustment = {
                'before': original_confidence,
                'after': result['confidence'],
                'multiplier': fng_multipliers.get('buy' if result['direction'] == 'buy' else 'sell', 1.0)
            }
            
            # 🔥 Apply BTC Dominance adjustment (altcoin timing!)
            btc_d_multiplier_data = self.btc_dominance.get_altcoin_multiplier(symbol)
            btc_d_multiplier = btc_d_multiplier_data['multiplier']
            original_after_fng = result['confidence']
            
            if btc_d_multiplier != 1.0:
                result['confidence'] *= btc_d_multiplier
                logger.info(f"[BTC_DOMINANCE] {symbol}: confidence adjusted: "
                          f"{original_after_fng:.2f} → {result['confidence']:.2f} "
                          f"(BTC.D: {btc_d_multiplier_data.get('btc_d', 'N/A')}%, {btc_d_multiplier_data.get('reason', 'unknown')}, "
                          f"trend: {btc_d_multiplier_data.get('trend', 'unknown')})")
            
            # Cache BTC.D for display (avoid duplicate API calls later)
            self._cached_btcd_display = self.btc_dominance.get_display_string(symbol)
            self._btcd_adjustment = {
                'before': original_after_fng,
                'after': result['confidence'],
                'multiplier': btc_d_multiplier
            }
            
            # Apply filters if signal is not wait
            filters_passed = True
            filter_results = None
            
            if result['direction'] != 'wait':
                filter_results = self.filter_manager.apply_all_filters(
                    symbol=symbol,
                    side=result['direction'],
                    funding_rate=funding_rate,
                    alt_prices=alt_prices
                )
                filters_passed = filter_results['filters_passed']
                
                # Update alt_bias in result
                if 'alt_bias' in filter_results:
                    result['alt_bias'] = filter_results['alt_bias']
                
                # If filters failed, override to wait
                if not filters_passed:
                    logger.info(
                        f"IMBA signal filtered out for {symbol}: "
                        f"{', '.join(filter_results.get('reasons', []))}"
                    )
                    result['direction'] = 'wait'
                    result['confidence'] = 0.0
            
            # Format output
            output = {
                'direction': result['direction'],
                'strength': result['confidence'],  # Use confidence as strength
                'confidence': result['confidence'],
                'regime': result['regime'],
                'signals': result['signals'],
                'filters_passed': filters_passed,
                'metadata': {
                    'imba_enabled': True,
                    'votes': result['votes'],
                    'lstm_contribution': result.get('lstm_contribution', 0.0),
                    'alt_bias': result.get('alt_bias', 0.0),
                    'filter_results': filter_results
                }
            }
            
            # Log beautiful extended debug information
            self._log_extended_debug(symbol, result, output, df)
            
            return output
            
        except Exception as e:
            logger.error(f"IMBA signal generation failed for {symbol}: {e}", exc_info=True)
            return {
                'direction': 'wait',
                'strength': 0.0,
                'confidence': 0.0,
                'regime': {'kind': 'unknown'},
                'signals': [],
                'filters_passed': False,
                'metadata': {'error': str(e)}
            }
    
    def _log_extended_debug(self, symbol: str, result: Dict[str, Any], output: Dict[str, Any], df: pd.DataFrame) -> None:
        """
        Beautiful extended debug logging with emojis and structured output.
        Shows all 9 IMBA signals with their status, votes, and regime multipliers.
        """
        # Get regime info
        regime = result['regime']
        regime_kind = regime.get('kind', 'unknown')
        regime_emoji = {
            'trend': '📈',
            'flat': '📊', 
            'volatile': '⚡',
            'unknown': '❓'
        }.get(regime_kind, '❓')
        
        # Direction emoji
        direction = output['direction']
        direction_emoji = {
            'buy': '🟢',
            'sell': '🔴',
            'wait': '⏸️'
        }.get(direction, '⏸️')
        
        # Build header
        header = (
            f"\n{'='*80}\n"
            f"{direction_emoji} IMBA SIGNAL ANALYSIS: {symbol} {direction_emoji}\n"
            f"{'='*80}\n"
        )
        
        # Regime info
        regime_info = (
            f"{regime_emoji} REGIME: {regime_kind.upper()}\n"
            f"  ├─ ADX: {regime.get('adx', 0):.2f}\n"
            f"  ├─ BBW: {regime.get('bbw', 0):.4f}\n"
            f"  └─ Confidence: {regime.get('confidence', 0):.2f}\n"
        )
        
        # Voting results with adjustment breakdown
        votes = result['votes']
        base_confidence = votes['buy'] - votes['sell'] if votes['buy'] > votes['sell'] else votes['sell'] - votes['buy']
        final_confidence = output['confidence']
        
        voting_info = f"\n🗳️  VOTING RESULTS:\n"
        voting_info += f"  ├─ BUY votes:  {votes['buy']:.3f} {'🟢' if votes['buy'] > votes['sell'] else '⚪'}\n"
        voting_info += f"  ├─ SELL votes: {votes['sell']:.3f} {'🔴' if votes['sell'] > votes['buy'] else '⚪'}\n"
        voting_info += f"  ├─ Base confidence: {base_confidence:.3f}\n"
        
        # Show adjustments if they occurred (check for None!)
        if (hasattr(self, '_fng_adjustment') and self._fng_adjustment is not None and
            hasattr(self, '_btcd_adjustment') and self._btcd_adjustment is not None):
            fng_adj = self._fng_adjustment
            btcd_adj = self._btcd_adjustment
            
            # Only show if adjustment actually happened
            if fng_adj['multiplier'] != 1.0 or btcd_adj['multiplier'] != 1.0:
                voting_info += f"  ├─ After Fear & Greed (×{fng_adj['multiplier']:.2f}): {fng_adj['after']:.3f}\n"
                voting_info += f"  ├─ After BTC.D (×{btcd_adj['multiplier']:.2f}): {btcd_adj['after']:.3f}\n"
        
        voting_info += f"  └─ Final confidence: {final_confidence:.3f} {'✅' if final_confidence >= self.config.bt_conf_min else '❌'}\n"
        
        # All 12 signals breakdown (CVD, FVG, and Volume Profile added!)
        signals_info = "\n🔍 SIGNAL BREAKDOWN (12 indicators):\n"
        
        signal_names_map = {
            'bb_squeeze': '1️⃣  BB Squeeze',
            'vwap_pullback': '2️⃣  VWAP Pullback',
            'vwap_bands_mr': '3️⃣  VWAP Mean Rev',
            'breakout_retest': '4️⃣  Breakout Retest',
            'atr_momentum': '5️⃣  ATR Momentum',
            'rsi_mr': '6️⃣  RSI Mean Rev',
            'sfp': '7️⃣  Swing Failure',
            'ema_pinch': '8️⃣  EMA Pinch',
            'cvd': '9️⃣  CVD',                    # 🔥 NEW! Cumulative Volume Delta
            'fvg': '🔟 FVG',                    # 🔥 NEW! Fair Value Gaps
            'volume_profile': '1️⃣1️⃣ Volume Profile',  # 🔥 NEW! Volume Profile POC
            'obi': '1️⃣2️⃣ Order Imbalance'
        }
        
        # Use signal_details if available (contains weighted voting info)
        if 'signal_details' in result and result['signal_details']:
            # Parse signal_details which has format: "name(D:weight×mult×mult=final)"
            import re
            for i, detail_str in enumerate(result['signal_details'], 1):
                # Extract signal name and details
                # Format examples:
                # "BB Squeeze(wait)"
                # "VWAP Mean Rev(B:0.45×0.7×1.0=0.31)"
                # "EMA Pinch(FILTERED)"
                
                if '(' not in detail_str:
                    continue
                
                name_part, rest = detail_str.split('(', 1)
                rest = rest.rstrip(')')
                
                # Get display name
                display_name = f"{i}️⃣  {name_part:15s}"
                
                if rest == 'wait':
                    status_emoji = '⏸️  WAIT'
                    vote_display = '0.00'
                elif rest == 'FILTERED':
                    status_emoji = '🚫 FILTERED'
                    vote_display = '0.00'
                elif rest.startswith('B:') or rest.startswith('S:'):
                    direction_char = rest[0]
                    status_emoji = '🟢 BUY' if direction_char == 'B' else '🔴 SELL'
                    vote_display = rest[2:]  # Everything after "B:" or "S:"
                else:
                    status_emoji = '⏸️  WAIT'
                    vote_display = '0.00'
                
                signals_info += (
                    f"  ├─ {display_name:20s} │ {status_emoji:10s} │ "
                    f"Vote: {vote_display}\n"
                )
        else:
            # Fallback to old format (if signal_details not available)
            for signal_data in result['signals']:
                name = signal_data['name']
                direction = signal_data['direction']
                strength = signal_data['strength']
                
                # Get display name
                display_name = signal_names_map.get(name, name)
                
                # Status emoji
                if direction == 'buy':
                    status_emoji = '🟢 BUY'
                    vote_display = f"+{strength:.2f}"
                elif direction == 'sell':
                    status_emoji = '🔴 SELL'
                    vote_display = f"+{strength:.2f}"
                else:
                    status_emoji = '⏸️  WAIT'
                    vote_display = '0.00'
                
                # Regime multiplier (from aggregator logic)
                regime_mult = self.aggregator.regime_detector.get_regime_multiplier(
                    type('Regime', (), regime)(), name
                )
                
                signals_info += (
                    f"  ├─ {display_name:20s} │ {status_emoji:10s} │ "
                    f"Vote: {vote_display:6s} │ Regime×: {regime_mult:.2f}\n"
                )
        
        # Additional contributions
        contrib_info = ""
        lstm_contrib = result.get('lstm_contribution', 0.0)
        alt_bias = result.get('alt_bias', 0.0)
        
        if lstm_contrib != 0.0 or alt_bias != 0.0:
            contrib_info = "\n🤖 ADDITIONAL CONTRIBUTIONS:\n"
            if lstm_contrib != 0.0:
                contrib_info += f"  ├─ LSTM: {lstm_contrib:+.4f}\n"
            if alt_bias != 0.0:
                contrib_info += f"  └─ Alt Bias: {alt_bias:+.4f}\n"
        
        # Multi-source price display (exchanges + aggregators)
        exchange_info = ""
        try:
            from utils.exchange_prices import get_price_fetcher, format_exchange_prices, format_price
            import os
            
            # Get current Binance price
            current_price = float(df.iloc[-1]['close'])
            
            # Fetch prices from other sources (with caching, 60s TTL)
            fetcher = get_price_fetcher()
            cmc_api_key = os.getenv('CMC_API_KEY')  # Optional: set in .env if you have one
            prices = fetcher.fetch_all_sync(symbol, cmc_api_key=cmc_api_key)
            
            # Format prices
            exchanges_str, aggregators_str = format_exchange_prices(prices, current_price)
            
            # Build display
            exchange_info = f"\n📊 CURRENT PRICE: {format_price(current_price)} (Binance Futures)\n"
            
            if exchanges_str:
                exchange_info += f"💱 SPOT EXCHANGES:\n"
                exchange_info += f"  └─ {exchanges_str}\n"
            
            if aggregators_str:
                exchange_info += f"💎 PRICE AGGREGATORS:\n"
                exchange_info += f"  └─ {aggregators_str}\n"
            
            # Check spot-futures spread
            # Note: prices is a flat dict like {'bybit': 123.45, 'okx': 123.50, ...}
            if prices:
                # Extract only exchange prices (bybit, okx, kraken) - skip aggregators
                spot_exchanges = ['bybit', 'okx', 'kraken']
                spot_prices = {}
                for exch_name in spot_exchanges:
                    if exch_name in prices and prices[exch_name] and prices[exch_name] > 0:
                        spot_prices[exch_name] = prices[exch_name]
                
                logger.debug(f"[SPREAD_DEBUG] {symbol}: Collected {len(spot_prices)} spot prices: {spot_prices}")
                
                if len(spot_prices) >= 2:  # Need at least 2 spot prices
                    logger.info(f"[SPREAD_CHECK_START] {symbol}: Checking spread with {len(spot_prices)} spot exchanges")
                    spread_signal = self.spread_filter.analyze_spread(
                        futures_price=current_price,
                        spot_prices=spot_prices,
                        symbol=symbol
                    )
                    logger.debug(f"[SPREAD_RESULT] {symbol}: spread_signal={spread_signal}")
                    
                    if spread_signal:
                        # Add spread signal as an additional vote
                        if spread_signal.direction in result['votes']:
                            original_vote = result['votes'][spread_signal.direction]
                            result['votes'][spread_signal.direction] += spread_signal.strength
                            
                            # Log the boost
                            logger.info(
                                f"[SPREAD_BOOST] {symbol}: {spread_signal.direction.upper()} "
                                f"vote boosted by +{spread_signal.strength:.2f} "
                                f"({original_vote:.2f} → {result['votes'][spread_signal.direction]:.2f}) - "
                                f"{spread_signal.reason}"
                            )
                            
                            # Recalculate confidence with spread boost
                            if result['votes']['buy'] > result['votes']['sell']:
                                result['direction'] = 'buy'
                                result['confidence'] = result['votes']['buy'] - result['votes']['sell']
                            elif result['votes']['sell'] > result['votes']['buy']:
                                result['direction'] = 'sell'
                                result['confidence'] = result['votes']['sell'] - result['votes']['buy']
                            else:
                                result['direction'] = 'wait'
                                result['confidence'] = 0.0
                            
                            # Add spread info to exchange display
                            exchange_info += f"\n⚠️  SPREAD ALERT: {spread_signal.reason}\n"
                            exchange_info += f"  └─ Signal boost: {spread_signal.direction.upper()} +{spread_signal.strength:.2f}\n"
            
        except Exception as e:
            logger.debug(f"Failed to fetch multi-source prices: {e}")
            # Fallback to just current price
            try:
                current_price = float(df.iloc[-1]['close'])
                if current_price < 0.01:
                    formatted = f"{current_price:.6f}".rstrip('0').rstrip('.')
                    price_str = f"${formatted}"
                elif current_price < 1:
                    price_str = f"${current_price:.4f}"
                elif current_price < 10:
                    price_str = f"${current_price:.3f}"
                else:
                    price_str = f"${current_price:,.2f}"
                exchange_info = f"\n📊 CURRENT PRICE: {price_str}\n"
            except:
                pass
        
        # Fear & Greed Index display (use cached value to avoid duplicate API calls)
        fear_greed_info = ""
        try:
            if hasattr(self, '_cached_fng_display'):
                fear_greed_info = f"\n😱 {self._cached_fng_display}\n"
            else:
                fear_greed_info = f"\n😱 {self.fear_greed.get_display_string()}\n"
        except Exception:
            pass
        
        # BTC Dominance display (use cached value to avoid duplicate API calls)
        btc_d_info = ""
        try:
            if hasattr(self, '_cached_btcd_display'):
                btc_d_info = f"📊 {self._cached_btcd_display}\n"
            else:
                btc_d_info = f"📊 {self.btc_dominance.get_display_string(symbol)}\n"
        except Exception:
            pass
        
        # Decision summary with CORRECT logic
        # Trade Action should be EXECUTE only if:
        # 1. Direction is buy/sell (not wait)
        # 2. Filters passed
        # 3. Confidence >= bt_conf_min (CRITICAL CHECK!)
        will_execute = (
            direction in ('buy', 'sell') and 
            output['filters_passed'] and 
            output['strength'] >= self.config.bt_conf_min
        )
        
        # Calculate progress to threshold
        confidence_pct = (output['strength'] / self.config.bt_conf_min) * 100 if self.config.bt_conf_min > 0 else 0
        threshold_status = f"{output['strength']:.3f} / {self.config.bt_conf_min:.1f} ({confidence_pct:.0f}%)"
        
        # Build threshold line (avoid nested f-strings)
        threshold_met = output['strength'] >= self.config.bt_conf_min
        if threshold_met:
            threshold_line = '✅ MET'
        else:
            threshold_diff = self.config.bt_conf_min - output['strength']
            threshold_line = f'❌ BELOW (need {threshold_diff:.3f} more)'
        
        decision = (
            f"\n🎯 FINAL DECISION: {direction.upper()} {direction_emoji}\n"
            f"  ├─ Signal Strength: {threshold_status}\n"
            f"  ├─ Confidence Threshold: {threshold_line}\n"
            f"  ├─ Filters Passed: {'✅ YES' if output['filters_passed'] else '❌ NO'}\n"
            f"  └─ Trade Action: {'🚀 EXECUTE' if will_execute else '⏸️  WAIT'}\n"
        )
        
        footer = f"{'='*80}\n"
        
        # Combine all sections
        full_log = header + regime_info + voting_info + signals_info + contrib_info + exchange_info + fear_greed_info + btc_d_info + decision + footer
        
        # Log it with safe encoding for Windows
        try:
            logger.info(full_log)
        except UnicodeEncodeError:
            # Fallback for Windows console (cp1251)
            # Remove emojis and special characters
            safe_log = full_log.encode('ascii', 'ignore').decode('ascii')
            logger.info(safe_log)
    
    def convert_to_trading_signal(self, imba_result: Dict[str, Any], symbol: str):
        """
        Convert IMBA result to TradingSignal format (if needed).
        
        Args:
            imba_result: Result from generate_signal_from_df
            symbol: Trading symbol
            
        Returns:
            TradingSignal object (if TradingSignal class is available)
        """
        from datetime import datetime, timezone
        
        # Try to import TradingSignal
        try:
            from core.types import TradingSignal, SignalType
            
            # Map direction to SignalType
            if imba_result['direction'] == 'buy':
                signal_type = SignalType.BUY
            elif imba_result['direction'] == 'sell':
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=imba_result['strength'],
                timestamp=datetime.now(timezone.utc),
                metadata=imba_result['metadata']
            )
        except ImportError:
            # If TradingSignal not available, return dict
            return imba_result


def should_use_imba_signals(config: Config) -> bool:
    """
    Check if IMBA signals should be used.
    
    Args:
        config: Trading configuration
        
    Returns:
        True if IMBA signals are enabled
    """
    return getattr(config, 'use_imba_signals', False)


def integrate_imba_into_signal_generator(signal_generator, config: Config):
    """
    Monkey-patch existing SignalGenerator to use IMBA when enabled.
    
    Args:
        signal_generator: Existing SignalGenerator instance
        config: Trading configuration
    """
    if not should_use_imba_signals(config):
        return
    
    # Store original generate_signal method
    original_generate_signal = signal_generator.generate_signal
    
    # Create IMBA integration
    imba = IMBASignalIntegration(config)
    
    def imba_generate_signal(market_data, *args, **kwargs):
        """Wrapped generate_signal using IMBA."""
        # Convert market_data to DataFrame if needed
        if isinstance(market_data, pd.DataFrame):
            df = market_data
        elif hasattr(market_data, 'close') and hasattr(market_data, 'timestamp'):
            # MarketData object
            df = pd.DataFrame({
                'open': market_data.open,
                'high': market_data.high,
                'low': market_data.low,
                'close': market_data.close,
                'volume': market_data.volume
            })
        else:
            # Fallback to original method
            logger.warning("Cannot convert market_data to DataFrame, using original signal generator")
            return original_generate_signal(market_data, *args, **kwargs)
        
        # Get symbol from market_data or args
        symbol = getattr(market_data, 'symbol', config.symbol)
        
        # Get LSTM prediction if available
        lstm_pred = None
        if hasattr(signal_generator, 'lstm_predictor') and signal_generator.lstm_predictor:
            try:
                lstm_pred = signal_generator._get_lstm_prediction()
            except Exception:
                pass
        
        # Generate IMBA signal
        imba_result = imba.generate_signal_from_df(
            df=df,
            symbol=symbol,
            lstm_prediction=lstm_pred
        )
        
        # Convert to TradingSignal format
        return imba.convert_to_trading_signal(imba_result, symbol)
    
    # Replace method
    signal_generator.generate_signal = imba_generate_signal
    logger.info("SignalGenerator patched to use IMBA signals")
