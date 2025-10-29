"""
IMBA Research Trading Signals.

Implements 9 advanced trading signals from imba_research_bot:
1. BB Squeeze Breakout
2. VWAP Pullback
3. VWAP Bands Mean Reversion
4. Breakout Retest (Donchian)
5. ATR Momentum
6. RSI Mean Reversion
7. Swing Failure Pattern (SFP)
8. EMA Pinch (Convergence)
9. Orderbook Imbalance (requires WebSocket)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import logging

from data.indicators import TechnicalIndicators
from strategy.regime import Regime, RegimeDetector

logger = logging.getLogger(__name__)


# Signal weights based on quality and frequency
# Rare, accurate signals get higher weight
# Frequent, noisy signals get lower weight
# IMPORTANT: Use internal signal names (not display names!)
SIGNAL_WEIGHTS = {
    "ema_pinch": 1.5,           # Rare, very accurate convergence signal
    "cvd": 1.4,                 # 🔥 NEW! Cumulative Volume Delta - divergences very accurate (70-80%)
    "volume_profile": 1.3,      # 🔥 NEW! Volume Profile POC - mean reversion 65-70% accurate
    "sfp": 1.3,                 # Swing Failure Pattern - rare, strong reversal
    "fvg": 1.3,                 # 🔥 NEW! Fair Value Gaps - gap fills 65-70% accurate
    "breakout_retest": 1.2,     # Medium frequency, good quality
    "atr_momentum": 1.1,        # Momentum confirmation
    "bb_squeeze": 1.0,          # Standard signal
    "rsi_mr": 0.9,              # RSI Mean Rev - can be noisy in trends
    "vwap_pullback": 0.8,       # Moderate frequency
    "vwap_bands_mr": 0.7,       # ⚠️ VWAP Mean Rev - VERY FREQUENT, NOISY - reduced weight
    "obi": 1.0,                 # Orderbook imbalance (when available)
}


@dataclass
class SignalOut:
    """Signal output structure."""
    name: str
    direction: str  # "buy" | "sell" | "wait"
    strength: float  # 0.0 to 1.0
    info: Dict[str, Any]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "direction": self.direction,
            "strength": self.strength,
            "info": self.info
        }


class IMBASignals:
    """
    Collection of IMBA trading signals.
    """
    
    @staticmethod
    def bb_squeeze(df: pd.DataFrame, lookback: int = 200) -> SignalOut:
        """
        BB Squeeze Breakout Signal.
        
        Detects Bollinger Bands squeeze and signals on breakout.
        - Squeeze: BB Width in lowest 20% of last 200 candles
        - BUY: Price breaks above upper band during squeeze
        - SELL: Price breaks below lower band during squeeze
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Period for squeeze detection
            
        Returns:
            SignalOut with direction and strength
        """
        if len(df) < 40:
            return SignalOut("bb_squeeze", "wait", 0.0, {})
        
        try:
            # Calculate BB Width
            if 'bb_upper' not in df.columns:
                bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
                    df['close'], period=20, std_dev=2.0
                )
                df['bb_upper'] = bb_upper
                df['bb_middle'] = bb_middle
                df['bb_lower'] = bb_lower
                df['bb_width'] = (bb_upper - bb_lower) / df['close']
            
            # Detect squeeze (MORE AGGRESSIVE)
            recent_widths = df['bb_width'].tail(lookback).dropna()
            if len(recent_widths) < 40:
                return SignalOut("bb_squeeze", "wait", 0.0, {})
            
            # Further relaxed threshold: 35% quantile for more signals (was 30%, originally 20%)
            threshold = float(np.quantile(recent_widths, 0.35))
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            direction = "wait"
            strength = 0.0
            
            # Check if in squeeze OR near squeeze
            if last['bb_width'] <= threshold * 1.2:  # Give 20% buffer
                # Breakout above OR touching upper band
                if last['close'] >= last['bb_upper'] * 0.998:  # 99.8% of upper (looser)
                    direction = "buy"
                    # Stronger signal if actually broke out
                    if last['close'] > last['bb_upper']:
                        strength = 0.70  # Increased from 0.6
                    else:
                        strength = 0.55  # Near upper band
                
                # Breakout below OR touching lower band
                elif last['close'] <= last['bb_lower'] * 1.002:  # 100.2% of lower (looser)
                    direction = "sell"
                    if last['close'] < last['bb_lower']:
                        strength = 0.70
                    else:
                        strength = 0.55
            
            return SignalOut(
                name="bb_squeeze",
                direction=direction,
                strength=strength,
                info={
                    "squeeze_threshold": threshold,
                    "bb_width": float(last['bb_width']),
                    "in_squeeze": last['bb_width'] <= threshold
                }
            )
            
        except Exception as e:
            logger.error(f"BB Squeeze signal error: {e}")
            return SignalOut("bb_squeeze", "wait", 0.0, {"error": str(e)})
    
    @staticmethod
    def vwap_pullback(df: pd.DataFrame, vwap_band_pct: float = 0.005) -> SignalOut:
        """
        VWAP Pullback Signal (MORE AGGRESSIVE).
        
        Detects pullbacks to VWAP in trending markets.
        - Trend detection: EMA50 (relaxed threshold)
        - Pullback: Price within vwap_band_pct of VWAP (0.5% instead of 0.3%)
        - Confirmation: EMA20 alignment (loosened)
        
        Args:
            df: DataFrame with OHLCV data
            vwap_band_pct: VWAP band percentage (default 0.5%, was 0.3%)
            
        Returns:
            SignalOut with direction and strength
        """
        if len(df) < 60:
            return SignalOut("vwap_pullback", "wait", 0.0, {})
        
        try:
            # Calculate VWAP if not present
            if 'vwap' not in df.columns:
                df['vwap'] = TechnicalIndicators.vwap(
                    df['high'], df['low'], df['close'], df['volume']
                )
            
            # Calculate EMAs
            if 'ema_20' not in df.columns:
                df['ema_20'] = TechnicalIndicators.ema(df['close'], 20)
            if 'ema_50' not in df.columns:
                df['ema_50'] = TechnicalIndicators.ema(df['close'], 50)
            
            last = df.iloc[-1]
            
            # Check for trend
            uptrend = last['ema_50'] < last['close']
            downtrend = last['ema_50'] > last['close']
            
            # Check distance to VWAP
            vwap_dist = abs(last['close'] - last['vwap']) / last['vwap']
            near_vwap = vwap_dist <= vwap_band_pct
            
            direction = "wait"
            strength = 0.0
            
            # Bullish pullback
            if uptrend and near_vwap and last['close'] > last['ema_20']:
                direction = "buy"
                strength = 0.55
            
            # Bearish pullback
            elif downtrend and near_vwap and last['close'] < last['ema_20']:
                direction = "sell"
                strength = 0.55
            
            return SignalOut(
                name="vwap_pullback",
                direction=direction,
                strength=strength,
                info={
                    "vwap_distance": float(vwap_dist),
                    "near_vwap": near_vwap,
                    "trend": "up" if uptrend else ("down" if downtrend else "none")
                }
            )
            
        except Exception as e:
            logger.error(f"VWAP Pullback signal error: {e}")
            return SignalOut("vwap_pullback", "wait", 0.0, {"error": str(e)})
    
    @staticmethod
    def vwap_bands_mr(df: pd.DataFrame, vwap_band_pct: float = 0.008) -> SignalOut:
        """
        VWAP Bands Mean Reversion Signal - ANTI-SPAM optimized.
        
        Mean reversion from VWAP band boundaries with strict entry conditions.
        - Only signals on FRESH breakouts of bands
        - Requires momentum + volume confirmation  
        - Anti-spam: blocks repetitive signals after initial trigger
        
        Args:
            df: DataFrame with OHLCV data
            vwap_band_pct: VWAP band percentage (default 0.8% - increased from 0.3%)
            
        Returns:
            SignalOut with direction and strength
        """
        if len(df) < 30:
            return SignalOut("vwap_bands_mr", "wait", 0.0, {})
        
        try:
            # Calculate VWAP if not present
            if 'vwap' not in df.columns:
                df['vwap'] = TechnicalIndicators.vwap(
                    df['high'], df['low'], df['close'], df['volume']
                )
            
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            
            # Calculate bands - wider bands for less noise
            upper_band = last['vwap'] * (1 + vwap_band_pct)
            lower_band = last['vwap'] * (1 - vwap_band_pct)
            
            # ANTI-SPAM: Only signal on FRESH penetration (previous candle was inside bands)
            prev_in_bands = lower_band <= prev['close'] <= upper_band
            
            direction = "wait"
            strength = 0.0
            
            # STRICT CONDITIONS: Must have volume surge + momentum + fresh breakout
            volume_sma = df['volume'].rolling(20).mean().iloc[-1]
            volume_surge = last['volume'] > volume_sma * 1.5  # Require 50%+ volume increase
            
            # Price momentum requirement
            price_momentum = (last['close'] - prev['close']) / prev['close']
            
            # BUY: Fresh penetration below lower band with strong momentum DOWN
            if (last['close'] < lower_band and                    # Below band
                prev_in_bands and                               # Was inside bands previously  
                price_momentum < -0.003 and                     # Strong -0.3% momentum
                volume_surge):                                  # Volume confirmation
                
                direction = "buy"
                # Strength based on momentum and penetration
                penetration_depth = (lower_band - last['close']) / last['vwap']
                momentum_strength = abs(price_momentum) * 20  # Scale momentum 
                strength = min(0.45 + penetration_depth * 30 + momentum_strength, 0.70)
            
            # SELL: Fresh penetration above upper band with strong momentum UP
            elif (last['close'] > upper_band and                 # Above band  
                  prev_in_bands and                             # Was inside bands previously
                  price_momentum > 0.003 and                    # Strong +0.3% momentum
                  volume_surge):                                # Volume confirmation
                
                direction = "sell"
                # Strength based on momentum and penetration
                penetration_depth = (last['close'] - upper_band) / last['vwap']
                momentum_strength = abs(price_momentum) * 20
                strength = min(0.45 + penetration_depth * 30 + momentum_strength, 0.70)
            
            # ELIMINATED: No "touch" signals - only fresh breakouts with momentum
            
            return SignalOut(
                name="vwap_bands_mr",
                direction=direction,
                strength=strength,
                info={
                    "vwap": float(last['vwap']),
                    "upper_band": float(upper_band),
                    "lower_band": float(lower_band),
                    "price": float(last['close']),
                    "momentum": float(price_momentum),
                    "volume_surge": volume_surge,
                    "prev_in_bands": prev_in_bands,
                    "band_width_pct": vwap_band_pct * 100
                }
            )
            
        except Exception as e:
            logger.error(f"VWAP Bands MR signal error: {e}")
            return SignalOut("vwap_bands_mr", "wait", 0.0, {"error": str(e)})
    
    @staticmethod
    def breakout_retest(df: pd.DataFrame, period: int = 20) -> SignalOut:
        """
        Breakout Retest Signal (Donchian Channels).
        
        Detects breakouts of Donchian channels followed by successful retest.
        - Breakout: Price exceeds highest high / lowest low
        - Retest: Price pulls back but holds above/below breakout level
        
        Args:
            df: DataFrame with OHLCV data
            period: Donchian channel period (default 20)
            
        Returns:
            SignalOut with direction and strength
        """
        if len(df) < period + 5:
            return SignalOut("breakout_retest", "wait", 0.0, {})
        
        try:
            # Calculate Donchian Channels
            if 'don_upper' not in df.columns:
                don_upper, don_middle, don_lower = TechnicalIndicators.donchian_channels(
                    df['high'], df['low'], period=period
                )
                df['don_upper'] = don_upper
                df['don_middle'] = don_middle
                df['don_lower'] = don_lower
            
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else last
            
            direction = "wait"
            strength = 0.0
            
            # Bullish breakout + retest
            # Previous: broke above, Current: pulled back but held
            if prev['high'] > prev['don_upper'] and last['low'] > prev['don_upper']:
                direction = "buy"
                strength = 0.6
            
            # Bearish breakout + retest
            elif prev['low'] < prev['don_lower'] and last['high'] < prev['don_lower']:
                direction = "sell"
                strength = 0.6
            
            return SignalOut(
                name="breakout_retest",
                direction=direction,
                strength=strength,
                info={
                    "don_upper": float(last['don_upper']),
                    "don_lower": float(last['don_lower']),
                    "price": float(last['close'])
                }
            )
            
        except Exception as e:
            logger.error(f"Breakout Retest signal error: {e}")
            return SignalOut("breakout_retest", "wait", 0.0, {"error": str(e)})
    
    @staticmethod
    def atr_momentum(df: pd.DataFrame, atr_mult: float = 0.6, adx_threshold: float = 18.0) -> SignalOut:
        """
        ATR Momentum Signal.
        
        Detects strong directional moves based on candle size vs ATR.
        - Candle size > atr_mult * ATR
        - ADX >= adx_threshold (trend confirmation)
        - Direction: Bull/bear candle
        
        Args:
            df: DataFrame with OHLCV data
            atr_mult: ATR multiplier for candle size threshold
            adx_threshold: Minimum ADX for trend confirmation
            
        Returns:
            SignalOut with direction and strength
        """
        if len(df) < 30:
            return SignalOut("atr_momentum", "wait", 0.0, {})
        
        try:
            # Calculate ATR
            if 'atr' not in df.columns:
                df['atr'] = TechnicalIndicators.atr(
                    df['high'], df['low'], df['close'], period=14
                )
            
            # Calculate ADX
            if 'adx' not in df.columns:
                df['adx'] = TechnicalIndicators.adx(
                    df['high'], df['low'], df['close'], period=14
                )
            
            last = df.iloc[-1]
            
            candle_size = abs(last['close'] - last['open'])
            threshold_size = atr_mult * last['atr']
            
            # Convert ADX to scalar to avoid Series ambiguity
            adx_value = float(last['adx']) if not pd.isna(last['adx']) else 0.0
            
            direction = "wait"
            strength = 0.0
            
            # Large bullish candle with trend confirmation
            if (candle_size > threshold_size and 
                last['close'] > last['open'] and 
                adx_value >= adx_threshold):
                direction = "buy"
                # Dynamic strength based on candle size
                strength = min(0.85, 0.5 + (candle_size / last['atr']) * 0.15)
            
            # Large bearish candle with trend confirmation
            elif (candle_size > threshold_size and 
                  last['close'] < last['open'] and 
                  adx_value >= adx_threshold):
                direction = "sell"
                strength = min(0.85, 0.5 + (candle_size / last['atr']) * 0.15)
            
            return SignalOut(
                name="atr_momentum",
                direction=direction,
                strength=strength,
                info={
                    "candle_size": float(candle_size),
                    "atr": float(last['atr']),
                    "atr_ratio": float(candle_size / last['atr']),
                    "adx": float(last['adx'])
                }
            )
            
        except Exception as e:
            logger.error(f"ATR Momentum signal error: {e}")
            return SignalOut("atr_momentum", "wait", 0.0, {"error": str(e)})
    
    @staticmethod
    def rsi_mr(df: pd.DataFrame, rsi_buy: float = 30.0, rsi_sell: float = 70.0) -> SignalOut:
        """
        Adaptive RSI Mean Reversion Signal.
        
        Smart RSI with dynamic thresholds based on market conditions:
        - Adjusts levels based on trend strength (ADX)
        - Uses price position relative to BB for context
        - Detects divergences for stronger signals
        - Considers volume confirmation
        - Adapts to volatility regime
        
        TREND MARKET (ADX > 40):
          - More extreme levels: 25/75 → wait for real exhaustion
        FLAT MARKET (ADX < 25):
          - Tighter levels: 35/65 → catch small moves
        VOLATILE MARKET:
          - Standard levels: 30/70
          
        Args:
            df: DataFrame with OHLCV data
            rsi_buy: Base RSI oversold threshold (default 30)
            rsi_sell: Base RSI overbought threshold (default 70)
            
        Returns:
            SignalOut with direction and strength
        """
        if len(df) < 50:
            return SignalOut("rsi_mr", "wait", 0.0, {})
        
        try:
            # Calculate RSI
            if 'rsi' not in df.columns:
                df['rsi'] = TechnicalIndicators.rsi(df['close'], period=14)
            
            # Calculate ADX for trend detection
            if 'adx' not in df.columns:
                _, _, df['adx'] = TechnicalIndicators.adx(
                    df['high'], df['low'], df['close'], period=14
                )
            
            # Calculate BB for context
            if 'bb_lower' not in df.columns:
                bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
                    df['close'], period=20, std_dev=2.0
                )
                df['bb_upper'] = bb_upper
                df['bb_middle'] = bb_middle
                df['bb_lower'] = bb_lower
            
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Adaptive thresholds based on ADX (trend strength)
            # Convert to scalar to avoid Series ambiguity
            adx = float(last['adx']) if not pd.isna(last['adx']) else 30.0
            
            if adx > 40:
                # Strong trend - wait for extreme levels
                rsi_buy_adaptive = 25.0
                rsi_sell_adaptive = 75.0
                regime = "trend"
            elif adx < 25:
                # Flat market - catch earlier moves
                rsi_buy_adaptive = 35.0
                rsi_sell_adaptive = 65.0
                regime = "flat"
            else:
                # Volatile/normal market
                rsi_buy_adaptive = rsi_buy
                rsi_sell_adaptive = rsi_sell
                regime = "volatile"
            
            # Convert to scalars FIRST to avoid Series ambiguity
            rsi_current = float(last['rsi']) if not pd.isna(last['rsi']) else 50.0
            close_current = float(last['close']) if not pd.isna(last['close']) else 0.0
            bb_lower_val = float(last['bb_lower']) if not pd.isna(last['bb_lower']) else close_current * 0.95
            bb_upper_val = float(last['bb_upper']) if not pd.isna(last['bb_upper']) else close_current * 1.05
            
            # Price position in BB (0=lower, 0.5=middle, 1=upper) - using scalars
            bb_width = bb_upper_val - bb_lower_val
            if bb_width > 0:
                bb_position = (close_current - bb_lower_val) / bb_width
            else:
                bb_position = 0.5
            
            # Detect RSI divergence (price makes new low/high but RSI doesn't)
            divergence = None
            if len(df) >= 20:
                # Bullish divergence: price lower but RSI higher
                last_20 = df.tail(20).reset_index(drop=True)  # Reset index to avoid issues
                price_low_idx = last_20['close'].idxmin()
                rsi_low_idx = last_20['rsi'].idxmin()
                
                # Use iloc to ensure scalar access
                price_at_low = float(last_20.iloc[price_low_idx]['close'])
                rsi_at_low = float(last_20.iloc[price_low_idx]['rsi'])
                
                if price_low_idx != rsi_low_idx and price_at_low < close_current:
                    if rsi_at_low < rsi_current:
                        divergence = "bullish"
                
                # Bearish divergence: price higher but RSI lower
                price_high_idx = last_20['close'].idxmax()
                rsi_high_idx = last_20['rsi'].idxmax()
                
                # Use iloc to ensure scalar access
                price_at_high = float(last_20.iloc[price_high_idx]['close'])
                rsi_at_high = float(last_20.iloc[price_high_idx]['rsi'])
                
                if price_high_idx != rsi_high_idx and price_at_high > close_current:
                    if rsi_at_high > rsi_current:
                        divergence = "bearish"
            
            # Volume confirmation (current volume > average) - scalar conversion
            volume_ratio = 1.0
            if 'volume' in df.columns:
                avg_volume = float(df['volume'].tail(20).mean())  # Explicit scalar conversion
                current_volume = float(last['volume']) if not pd.isna(last['volume']) else 0.0
                if avg_volume > 0 and current_volume > 0:
                    volume_ratio = current_volume / avg_volume
            
            volume_confirmed = volume_ratio > 0.6  # RELAXED from 0.8 to 0.6 for more signals
            
            direction = "wait"
            strength = 0.0
            
            # === BUY CONDITIONS (MORE AGGRESSIVE) ===
            # Lower threshold: catch moves earlier (was <=, now <)
            if rsi_current <= rsi_buy_adaptive + 5:  # Give 5-point buffer for more signals
                # Base strength from RSI level
                rsi_strength = (rsi_buy_adaptive - rsi_current) / rsi_buy_adaptive if rsi_buy_adaptive > 0 else 0.5
                base_strength = 0.30 + abs(rsi_strength) * 0.25  # Lowered base, increased multiplier
                
                # Bonus for being near BB lower (real support)
                if bb_position < 0.4:  # Relaxed from 0.3 to 0.4
                    base_strength += 0.12
                
                # Bonus for bullish divergence
                if divergence == "bullish":
                    base_strength += 0.18  # Increased from 0.15
                
                # Bonus for volume
                if volume_confirmed:
                    base_strength += 0.08  # Increased from 0.05
                
                # MORE AGGRESSIVE: reduced safety margin
                if close_current > bb_lower_val * 0.99:  # Was 0.995, now 0.99 (looser)
                    direction = "buy"
                    strength = min(0.80, base_strength)  # Increased max from 0.75 to 0.80
            
            # === SELL CONDITIONS (MORE AGGRESSIVE) ===
            elif rsi_current >= rsi_sell_adaptive - 5:  # Give 5-point buffer for more signals
                # Base strength from RSI level
                rsi_strength = (rsi_current - rsi_sell_adaptive) / (100 - rsi_sell_adaptive) if rsi_sell_adaptive < 100 else 0.5
                base_strength = 0.30 + abs(rsi_strength) * 0.25
                
                # Bonus for being near BB upper (real resistance)
                if bb_position > 0.6:  # Relaxed from 0.7 to 0.6
                    base_strength += 0.12
                
                # Bonus for bearish divergence
                if divergence == "bearish":
                    base_strength += 0.18  # Increased from 0.15
                
                # Bonus for volume
                if volume_confirmed:
                    base_strength += 0.08  # Increased from 0.05
                
                # MORE AGGRESSIVE: reduced safety margin
                if close_current < bb_upper_val * 1.01:  # Was 1.005, now 1.01 (looser)
                    direction = "sell"
                    strength = min(0.80, base_strength)  # Increased max from 0.75 to 0.80
            
            return SignalOut(
                name="rsi_mr",
                direction=direction,
                strength=strength,
                info={
                    "rsi": float(last['rsi']),
                    "rsi_buy_level": float(rsi_buy_adaptive),
                    "rsi_sell_level": float(rsi_sell_adaptive),
                    "adx": float(adx),
                    "regime": regime,
                    "bb_position": float(bb_position),
                    "divergence": divergence if divergence else "none",
                    "volume_ratio": float(volume_ratio),
                    "volume_confirmed": volume_confirmed
                }
            )
            
        except Exception as e:
            logger.error(f"RSI MR signal error: {e}")
            return SignalOut("rsi_mr", "wait", 0.0, {"error": str(e)})
    
    @staticmethod
    def sfp(df: pd.DataFrame, lookback: int = 8, wick_ratio: float = 0.5) -> SignalOut:
        """
        Swing Failure Pattern (SFP) Signal.
        
        Detects false breakouts / stop hunts.
        - Bearish SFP: High > prev_high but close < prev_high (rejection)
        - Bullish SFP: Low < prev_low but close > prev_low (rejection)
        - Long wicks confirm the rejection
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Periods to look for swing high/low
            wick_ratio: Minimum wick size ratio for confirmation
            
        Returns:
            SignalOut with direction and strength
        """
        if len(df) < lookback + 2:
            return SignalOut("sfp", "wait", 0.0, {})
        
        try:
            last = df.iloc[-1]
            recent = df.iloc[-lookback-1:-1]
            
            prev_high = recent['high'].max()
            prev_low = recent['low'].min()
            
            body_size = abs(last['close'] - last['open'])
            upper_wick = last['high'] - max(last['close'], last['open'])
            lower_wick = min(last['close'], last['open']) - last['low']
            
            direction = "wait"
            strength = 0.0
            
            # Bullish SFP: Fakeout below then rejection up
            if (last['low'] < prev_low and 
                last['close'] > prev_low and
                lower_wick > body_size * wick_ratio):
                direction = "buy"
                strength = 0.55
            
            # Bearish SFP: Fakeout above then rejection down
            elif (last['high'] > prev_high and 
                  last['close'] < prev_high and
                  upper_wick > body_size * wick_ratio):
                direction = "sell"
                strength = 0.55
            
            return SignalOut(
                name="sfp",
                direction=direction,
                strength=strength,
                info={
                    "prev_high": float(prev_high),
                    "prev_low": float(prev_low),
                    "upper_wick": float(upper_wick),
                    "lower_wick": float(lower_wick),
                    "body_size": float(body_size)
                }
            )
            
        except Exception as e:
            logger.error(f"SFP signal error: {e}")
            return SignalOut("sfp", "wait", 0.0, {"error": str(e)})
    
    @staticmethod
    def ema_pinch(df: pd.DataFrame, lookback: int = 200, quantile: float = 0.15) -> SignalOut:
        """
        EMA Pinch (Convergence) Signal.
        
        Detects EMA convergence followed by alignment.
        - Tracks distance between EMA20, EMA50, SMA200
        - Squeeze: Distance in lowest quantile% over lookback
        - BUY: EMAs aligned upward (EMA20 > EMA50 > SMA200)
        - SELL: EMAs aligned downward
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Period for squeeze detection
            quantile: Percentile for squeeze threshold (default 15%)
            
        Returns:
            SignalOut with direction and strength
        """
        if len(df) < 210:
            return SignalOut("ema_pinch", "wait", 0.0, {})
        
        try:
            # Calculate EMAs
            if 'ema_20' not in df.columns:
                df['ema_20'] = TechnicalIndicators.ema(df['close'], 20)
            if 'ema_50' not in df.columns:
                df['ema_50'] = TechnicalIndicators.ema(df['close'], 50)
            if 'sma_200' not in df.columns:
                df['sma_200'] = TechnicalIndicators.sma(df['close'], 200)
            
            # Calculate EMA distance
            df['ema_distance'] = (
                abs(df['ema_20'] - df['ema_50']) + 
                abs(df['ema_50'] - df['sma_200']) + 
                abs(df['ema_20'] - df['sma_200'])
            ) / df['close']
            
            # Detect pinch
            recent_distances = df['ema_distance'].tail(lookback).dropna()
            if len(recent_distances) < 50:
                return SignalOut("ema_pinch", "wait", 0.0, {})
            
            threshold = float(np.quantile(recent_distances, quantile))
            last = df.iloc[-1]
            
            direction = "wait"
            strength = 0.0
            
            in_pinch = last['ema_distance'] <= threshold
            
            if in_pinch:
                # Bullish alignment
                if last['ema_20'] > last['ema_50'] > last['sma_200']:
                    direction = "buy"
                    strength = 0.5
                
                # Bearish alignment
                elif last['ema_20'] < last['ema_50'] < last['sma_200']:
                    direction = "sell"
                    strength = 0.5
            
            return SignalOut(
                name="ema_pinch",
                direction=direction,
                strength=strength,
                info={
                    "ema_distance": float(last['ema_distance']),
                    "pinch_threshold": threshold,
                    "in_pinch": in_pinch,
                    "ema_20": float(last['ema_20']),
                    "ema_50": float(last['ema_50']),
                    "sma_200": float(last['sma_200'])
                }
            )
            
        except Exception as e:
            logger.error(f"EMA Pinch signal error: {e}")
            return SignalOut("ema_pinch", "wait", 0.0, {"error": str(e)})
    
    @staticmethod
    def cvd(df: pd.DataFrame) -> SignalOut:
        """
        Cumulative Volume Delta Signal.
        
        Tracks buy vs sell volume to detect accumulation/distribution.
        Primary signal: Price-CVD divergences (70-80% win rate).
        Secondary: CVD vs MA crossovers (weaker).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            SignalOut with direction and strength
        """
        try:
            from strategy.cvd_signal import cvd_signal
            
            result = cvd_signal(df)
            
            return SignalOut(
                name="cvd",
                direction=result["direction"],
                strength=result["strength"],
                info=result["info"]
            )
            
        except ImportError:
            logger.warning("CVD signal module not available")
            return SignalOut("cvd", "wait", 0.0, {"error": "Module not available"})
        except Exception as e:
            logger.error(f"CVD signal error: {e}")
            return SignalOut("cvd", "wait", 0.0, {"error": str(e)})
    
    @staticmethod
    def fvg(df: pd.DataFrame) -> SignalOut:
        """
        Fair Value Gap Signal.
        
        Detects liquidity voids (gaps) in price action and signals when
        price approaches to fill them (65-70% win rate).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            SignalOut with direction and strength
        """
        try:
            from strategy.fvg_signal import fvg_signal
            
            result = fvg_signal(df)
            
            return SignalOut(
                name="fvg",
                direction=result["direction"],
                strength=result["strength"],
                info=result["info"]
            )
            
        except ImportError:
            logger.warning("FVG signal module not available")
            return SignalOut("fvg", "wait", 0.0, {"error": "Module not available"})
        except Exception as e:
            logger.error(f"FVG signal error: {e}")
            return SignalOut("fvg", "wait", 0.0, {"error": str(e)})
    
    @staticmethod
    def volume_profile(df: pd.DataFrame) -> SignalOut:
        """
        Volume Profile POC (Point of Control) Signal.
        
        Analyzes where most volume traded to find key levels.
        POC acts as price magnet - mean reversion signal (65-70% win rate).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            SignalOut with direction and strength
        """
        try:
            from strategy.volume_profile import volume_profile_signal
            
            result = volume_profile_signal(df)
            
            return SignalOut(
                name="volume_profile",
                direction=result["direction"],
                strength=result["strength"],
                info=result["info"]
            )
            
        except ImportError:
            logger.warning("Volume Profile signal module not available")
            return SignalOut("volume_profile", "wait", 0.0, {"error": "Module not available"})
        except Exception as e:
            logger.error(f"Volume Profile signal error: {e}")
            return SignalOut("volume_profile", "wait", 0.0, {"error": str(e)})


class IMBASignalAggregator:
    """
    Aggregates IMBA signals with regime-based weighting.
    """
    
    def __init__(self, 
                 min_confidence: float = 0.80,
                 lstm_weight: float = 0.35,
                 alt_influence: float = 0.35):
        """
        Initialize aggregator.
        
        Args:
            min_confidence: Minimum confidence threshold for trade
            lstm_weight: Weight for LSTM signal
            alt_influence: Weight for altcoin bias
        """
        self.min_confidence = min_confidence
        self.lstm_weight = lstm_weight
        self.alt_influence = alt_influence
        self.regime_detector = RegimeDetector()
    
    def aggregate(self,
                  df: pd.DataFrame,
                  lstm_rel: float = 0.0,
                  alt_bias: float = 0.0,
                  lstm_threshold: float = 0.0015) -> Dict[str, Any]:
        """
        Aggregate all IMBA signals.
        
        Args:
            df: DataFrame with OHLCV data
            lstm_rel: LSTM prediction relative value
            alt_bias: Altcoin market bias (-1 to 1)
            lstm_threshold: Minimum LSTM value to consider
            
        Returns:
            Dictionary with aggregated signal
        """
        # Detect regime
        regime = self.regime_detector.detect_regime(df)
        
        # Generate all signals (12 indicators now!)
        signals = [
            IMBASignals.bb_squeeze(df),
            IMBASignals.vwap_pullback(df),
            IMBASignals.vwap_bands_mr(df),
            IMBASignals.breakout_retest(df),
            IMBASignals.atr_momentum(df),
            IMBASignals.rsi_mr(df),
            IMBASignals.sfp(df),
            IMBASignals.ema_pinch(df),
            IMBASignals.cvd(df),             # 🔥 NEW! Cumulative Volume Delta
            IMBASignals.fvg(df),             # 🔥 NEW! Fair Value Gaps
            IMBASignals.volume_profile(df),  # 🔥 NEW! Volume Profile POC
        ]
        
        # Aggregate votes with weighted voting
        votes = {"buy": 0.0, "sell": 0.0}
        signal_details = []  # For debugging
        
        for signal in signals:
            if signal.direction in ("buy", "sell"):
                # Get base weight from signal strength
                weight = signal.strength
                original_weight = weight
                
                # Apply signal quality weight (rare signals > frequent signals)
                signal_weight_mult = SIGNAL_WEIGHTS.get(signal.name, 1.0)
                weight *= signal_weight_mult
                
                # Apply regime multiplier
                regime_mult = self.regime_detector.get_regime_multiplier(
                    regime, signal.name
                )
                weight *= regime_mult
                
                # Filter if needed
                filter_passed = self.regime_detector.regime_filter(regime, signal.name, signal.direction)
                if filter_passed:
                    votes[signal.direction] += weight
                    # Log with all multipliers for debugging
                    signal_details.append(
                        f"{signal.name}({signal.direction[0].upper()}:{original_weight:.2f}"
                        f"×{signal_weight_mult:.1f}×{regime_mult:.1f}={weight:.2f})"
                    )
                else:
                    signal_details.append(f"{signal.name}(FILTERED)")
            else:
                signal_details.append(f"{signal.name}(wait)")
        
        # Add LSTM contribution
        if abs(lstm_rel) >= lstm_threshold:
            lstm_vote = self.lstm_weight
            if lstm_rel > 0:
                votes["buy"] += lstm_vote
            else:
                votes["sell"] += lstm_vote
        
        # Add alt bias
        if alt_bias > 0:
            votes["buy"] += abs(alt_bias) * self.alt_influence * 0.2
        elif alt_bias < 0:
            votes["sell"] += abs(alt_bias) * self.alt_influence * 0.2
        
        # Determine final signal
        direction = "wait"
        confidence = 0.0
        
        if votes["buy"] > votes["sell"] and votes["buy"] >= self.min_confidence:
            direction = "buy"
            confidence = votes["buy"] - votes["sell"]
        elif votes["sell"] > votes["buy"] and votes["sell"] >= self.min_confidence:
            direction = "sell"
            confidence = votes["sell"] - votes["buy"]
        
        return {
            "direction": direction,
            "confidence": confidence,
            "votes": votes,
            "regime": regime.to_dict(),
            "signals": [s.to_dict() for s in signals],
            "signal_details": signal_details,  # Add signal breakdown
            "lstm_contribution": lstm_rel if abs(lstm_rel) >= lstm_threshold else 0.0,
            "alt_bias": alt_bias
        }
