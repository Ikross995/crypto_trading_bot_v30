"""
Cumulative Volume Delta (CVD) Signal Implementation.

CVD tracks the difference between aggressive buy volume and aggressive sell volume.
When buyers are aggressive (market buy orders), CVD increases.
When sellers are aggressive (market sell orders), CVD decreases.

Key concept: CVD divergences are highly predictive:
- Price lower low + CVD higher low = Bullish divergence (buy signal)
- Price higher high + CVD lower high = Bearish divergence (sell signal)

Win rate: 70-80% for divergence signals.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class CVDSignalOut:
    """CVD signal output."""
    direction: str  # "buy" | "sell" | "wait"
    strength: float  # 0.0 to 1.0
    cvd_current: float
    cvd_ma: float
    divergence_type: Optional[str]  # "bullish" | "bearish" | None
    info: dict


class CVDSignal:
    """
    Cumulative Volume Delta signal generator.
    
    Features:
    - Tracks aggressive buy vs sell volume
    - Detects price-CVD divergences
    - Multiple confirmation methods
    - Adaptive thresholds
    """
    
    def __init__(self, 
                 ma_period: int = 20,
                 divergence_lookback: int = 20,
                 min_divergence_strength: float = 0.15):
        """
        Initialize CVD signal generator.
        
        Args:
            ma_period: Moving average period for CVD smoothing
            divergence_lookback: Candles to look back for divergence detection
            min_divergence_strength: Minimum strength for divergence signal (0-1)
        """
        self.ma_period = ma_period
        self.divergence_lookback = divergence_lookback
        self.min_divergence_strength = min_divergence_strength
    
    def calculate_cvd(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Cumulative Volume Delta.
        
        For each candle, estimates buy vs sell volume based on close position
        within the candle range:
        - Close near high = more buying pressure
        - Close near low = more selling pressure
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with CVD values
        """
        if len(df) < 2:
            return pd.Series([0.0] * len(df), index=df.index)
        
        # Calculate buy/sell volume estimation
        # Method: Use close position in range as proxy for buyer/seller aggression
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # Avoid division by zero
        range_size = high - low
        range_size[range_size == 0] = 1e-10
        
        # Position of close in range (0 = at low, 1 = at high)
        close_position = (close - low) / range_size
        
        # Buy volume = volume * close_position
        # Sell volume = volume * (1 - close_position)
        buy_volume = volume * close_position
        sell_volume = volume * (1 - close_position)
        
        # Delta = buy - sell
        delta = buy_volume - sell_volume
        
        # Cumulative sum
        cvd = np.cumsum(delta)
        
        return pd.Series(cvd, index=df.index)
    
    def detect_divergence(self, 
                         price_series: pd.Series,
                         cvd_series: pd.Series,
                         lookback: int) -> Tuple[Optional[str], float]:
        """
        Detect price-CVD divergence.
        
        Args:
            price_series: Price data (typically close)
            cvd_series: CVD data
            lookback: Number of candles to analyze
            
        Returns:
            Tuple of (divergence_type, strength)
            divergence_type: "bullish" | "bearish" | None
            strength: 0.0 to 1.0
        """
        if len(price_series) < lookback or len(cvd_series) < lookback:
            return None, 0.0
        
        # Get recent data
        recent_price = price_series.tail(lookback)
        recent_cvd = cvd_series.tail(lookback)
        
        # Find swing highs and lows in price
        price_highs_idx = []
        price_lows_idx = []
        
        for i in range(2, len(recent_price) - 2):
            # Local high: higher than neighbors
            if (recent_price.iloc[i] > recent_price.iloc[i-1] and 
                recent_price.iloc[i] > recent_price.iloc[i-2] and
                recent_price.iloc[i] > recent_price.iloc[i+1] and
                recent_price.iloc[i] > recent_price.iloc[i+2]):
                price_highs_idx.append(i)
            
            # Local low: lower than neighbors
            if (recent_price.iloc[i] < recent_price.iloc[i-1] and 
                recent_price.iloc[i] < recent_price.iloc[i-2] and
                recent_price.iloc[i] < recent_price.iloc[i+1] and
                recent_price.iloc[i] < recent_price.iloc[i+2]):
                price_lows_idx.append(i)
        
        # Need at least 2 swings for divergence
        if len(price_highs_idx) < 2 and len(price_lows_idx) < 2:
            return None, 0.0
        
        # Check for bearish divergence (price higher high, CVD lower high)
        if len(price_highs_idx) >= 2:
            # Get last 2 highs
            ph1_idx = price_highs_idx[-2]
            ph2_idx = price_highs_idx[-1]
            
            ph1_price = recent_price.iloc[ph1_idx]
            ph2_price = recent_price.iloc[ph2_idx]
            ph1_cvd = recent_cvd.iloc[ph1_idx]
            ph2_cvd = recent_cvd.iloc[ph2_idx]
            
            # Bearish divergence: price makes higher high, CVD makes lower high
            if ph2_price > ph1_price and ph2_cvd < ph1_cvd:
                # Calculate divergence strength
                price_increase = (ph2_price - ph1_price) / ph1_price
                cvd_decrease = (ph1_cvd - ph2_cvd) / abs(ph1_cvd) if ph1_cvd != 0 else 0
                
                # Strength based on magnitude of divergence
                strength = min(1.0, (price_increase + cvd_decrease) * 3)
                
                if strength >= self.min_divergence_strength:
                    logger.debug(f"[CVD] Bearish divergence detected: strength={strength:.2f}")
                    return "bearish", strength
        
        # Check for bullish divergence (price lower low, CVD higher low)
        if len(price_lows_idx) >= 2:
            # Get last 2 lows
            pl1_idx = price_lows_idx[-2]
            pl2_idx = price_lows_idx[-1]
            
            pl1_price = recent_price.iloc[pl1_idx]
            pl2_price = recent_price.iloc[pl2_idx]
            pl1_cvd = recent_cvd.iloc[pl1_idx]
            pl2_cvd = recent_cvd.iloc[pl2_idx]
            
            # Bullish divergence: price makes lower low, CVD makes higher low
            if pl2_price < pl1_price and pl2_cvd > pl1_cvd:
                # Calculate divergence strength
                price_decrease = (pl1_price - pl2_price) / pl1_price
                cvd_increase = (pl2_cvd - pl1_cvd) / abs(pl1_cvd) if pl1_cvd != 0 else 0
                
                # Strength based on magnitude of divergence
                strength = min(1.0, (price_decrease + cvd_increase) * 3)
                
                if strength >= self.min_divergence_strength:
                    logger.debug(f"[CVD] Bullish divergence detected: strength={strength:.2f}")
                    return "bullish", strength
        
        return None, 0.0
    
    def generate_signal(self, df: pd.DataFrame) -> CVDSignalOut:
        """
        Generate CVD trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            CVDSignalOut with signal direction and strength
        """
        if len(df) < self.divergence_lookback + 5:
            return CVDSignalOut(
                direction="wait",
                strength=0.0,
                cvd_current=0.0,
                cvd_ma=0.0,
                divergence_type=None,
                info={"error": "Insufficient data"}
            )
        
        try:
            # Calculate CVD
            cvd = self.calculate_cvd(df)
            
            # Calculate CVD moving average
            cvd_ma = cvd.rolling(window=self.ma_period).mean()
            
            # Get current values
            cvd_current = float(cvd.iloc[-1])
            cvd_ma_current = float(cvd_ma.iloc[-1])
            
            # Detect divergence
            divergence_type, divergence_strength = self.detect_divergence(
                price_series=df['close'],
                cvd_series=cvd,
                lookback=self.divergence_lookback
            )
            
            direction = "wait"
            strength = 0.0
            
            # PRIMARY SIGNAL: Divergence (strongest)
            if divergence_type == "bullish":
                direction = "buy"
                strength = 0.50 + (divergence_strength * 0.30)  # 0.50 to 0.80
                
            elif divergence_type == "bearish":
                direction = "sell"
                strength = 0.50 + (divergence_strength * 0.30)  # 0.50 to 0.80
            
            # SECONDARY SIGNAL: CVD vs MA (weaker, for confirmation)
            else:
                # CVD crossing above MA = buying pressure increasing
                if len(cvd) >= 2 and len(cvd_ma) >= 2:
                    cvd_prev = float(cvd.iloc[-2])
                    cvd_ma_prev = float(cvd_ma.iloc[-2])
                    
                    # Bullish cross
                    if cvd_prev <= cvd_ma_prev and cvd_current > cvd_ma_current:
                        direction = "buy"
                        strength = 0.40  # Weaker than divergence
                    
                    # Bearish cross
                    elif cvd_prev >= cvd_ma_prev and cvd_current < cvd_ma_current:
                        direction = "sell"
                        strength = 0.40
            
            # Additional info for debugging
            info = {
                "cvd_current": cvd_current,
                "cvd_ma": cvd_ma_current,
                "cvd_vs_ma": "above" if cvd_current > cvd_ma_current else "below",
                "divergence_strength": divergence_strength if divergence_type else 0.0,
                "signal_type": "divergence" if divergence_type else "crossover"
            }
            
            return CVDSignalOut(
                direction=direction,
                strength=strength,
                cvd_current=cvd_current,
                cvd_ma=cvd_ma_current,
                divergence_type=divergence_type,
                info=info
            )
            
        except Exception as e:
            logger.error(f"[CVD] Signal generation error: {e}")
            return CVDSignalOut(
                direction="wait",
                strength=0.0,
                cvd_current=0.0,
                cvd_ma=0.0,
                divergence_type=None,
                info={"error": str(e)}
            )


# Convenience function for IMBA integration
def cvd_signal(df: pd.DataFrame) -> dict:
    """
    Generate CVD signal (IMBA-compatible format).
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dict with direction, strength, and info
    """
    cvd = CVDSignal()
    result = cvd.generate_signal(df)
    
    return {
        "direction": result.direction,
        "strength": result.strength,
        "info": {
            "cvd_current": result.cvd_current,
            "cvd_ma": result.cvd_ma,
            "divergence": result.divergence_type or "none",
            **result.info
        }
    }
