"""
Spot-Futures Spread Filter

Detects significant price differences between spot exchanges and futures,
which often indicates overpricing or underpricing and provides strong
directional signals for trading.

When futures are significantly higher than spot (overpriced):
  → Strong SELL signal (expect reversion to spot)

When futures are significantly lower than spot (underpriced):
  → Strong BUY signal (expect reversion upward)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from statistics import mean, median

logger = logging.getLogger(__name__)


@dataclass
class SpreadSignal:
    """Signal generated from spot-futures spread analysis."""
    direction: str  # "buy", "sell", or "neutral"
    strength: float  # 0.0 to 2.0+ (can be very strong for large spreads)
    spread_pct: float  # Percentage difference
    avg_spot: float  # Average spot price
    futures_price: float  # Futures price
    reason: str  # Human-readable explanation


class SpotFuturesSpreadFilter:
    """
    Analyzes price differences between spot and futures markets.
    
    Large spreads indicate market inefficiencies that typically revert,
    providing high-confidence trading signals.
    """
    
    def __init__(
        self,
        min_spread_pct: float = 3.0,  # Minimum spread to generate signal
        strong_spread_pct: float = 5.0,  # Strong signal threshold
        extreme_spread_pct: float = 7.0,  # Extreme signal threshold
        base_strength: float = 0.6,  # Base signal strength
        strength_multiplier: float = 0.15,  # Strength per % spread
    ):
        """
        Initialize spread filter.
        
        Args:
            min_spread_pct: Minimum spread % to generate signal (default 3%)
            strong_spread_pct: Threshold for strong signal (default 5%)
            extreme_spread_pct: Threshold for extreme signal (default 7%)
            base_strength: Base signal strength (default 0.6)
            strength_multiplier: Additional strength per % spread (default 0.15)
        """
        self.min_spread_pct = min_spread_pct
        self.strong_spread_pct = strong_spread_pct
        self.extreme_spread_pct = extreme_spread_pct
        self.base_strength = base_strength
        self.strength_multiplier = strength_multiplier
        
        logger.info(
            f"SpotFuturesSpreadFilter initialized: "
            f"min={min_spread_pct}%, strong={strong_spread_pct}%, extreme={extreme_spread_pct}%"
        )
    
    def analyze_spread(
        self,
        futures_price: float,
        spot_prices: Dict[str, float],
        symbol: str = "UNKNOWN"
    ) -> Optional[SpreadSignal]:
        """
        Analyze spread between futures and spot prices.
        
        Args:
            futures_price: Current futures price
            spot_prices: Dict of exchange -> price (e.g., {"bybit": 4019.0, "okx": 4020.0})
            symbol: Trading symbol for logging
            
        Returns:
            SpreadSignal if significant spread detected, None otherwise
        """
        if not spot_prices or futures_price <= 0:
            return None
        
        # Calculate average spot price (use median to avoid outlier influence)
        valid_spots = [p for p in spot_prices.values() if p > 0]
        if not valid_spots:
            return None
        
        # Use median for robustness against outliers
        if len(valid_spots) >= 3:
            avg_spot = median(valid_spots)
        else:
            avg_spot = mean(valid_spots)
        
        # Calculate spread percentage
        spread_pct = ((futures_price / avg_spot) - 1.0) * 100.0
        
        # Determine direction and check if spread is significant
        if abs(spread_pct) < self.min_spread_pct:
            # Log at INFO level so users can see spread is being checked
            logger.info(
                f"[SPREAD_CHECK] {symbol}: {spread_pct:+.2f}% "
                f"(futures=${futures_price:.2f}, spot=${avg_spot:.2f}) - Below {self.min_spread_pct}% threshold, no signal"
            )
            return None
        
        # Calculate signal strength based on spread magnitude
        strength = self._calculate_strength(spread_pct)
        
        # Determine direction
        if spread_pct > 0:
            # Futures overpriced → SELL
            direction = "sell"
            reason = (
                f"Futures overpriced by {spread_pct:.1f}% vs spot "
                f"(${futures_price:.2f} vs ${avg_spot:.2f})"
            )
        else:
            # Futures underpriced → BUY
            direction = "buy"
            reason = (
                f"Futures underpriced by {abs(spread_pct):.1f}% vs spot "
                f"(${futures_price:.2f} vs ${avg_spot:.2f})"
            )
        
        signal = SpreadSignal(
            direction=direction,
            strength=strength,
            spread_pct=spread_pct,
            avg_spot=avg_spot,
            futures_price=futures_price,
            reason=reason
        )
        
        # Log significant spreads
        severity = self._get_spread_severity(abs(spread_pct))
        logger.info(
            f"[SPREAD] {symbol}: {severity.upper()} {direction.upper()} signal "
            f"(spread={spread_pct:+.2f}%, strength={strength:.2f}) - {reason}"
        )
        
        return signal
    
    def _calculate_strength(self, spread_pct: float) -> float:
        """
        Calculate signal strength based on spread percentage.
        
        Strength increases with spread magnitude:
        - 3%: 0.60 (base)
        - 5%: 0.90 (strong)
        - 7%: 1.20 (extreme)
        - 10%: 1.65 (very extreme)
        """
        abs_spread = abs(spread_pct)
        
        # Base strength + additional strength per %
        strength = self.base_strength + (abs_spread * self.strength_multiplier)
        
        # Apply multipliers for different severity levels
        if abs_spread >= self.extreme_spread_pct:
            # Extreme spread: boost strength significantly
            strength *= 1.2
        elif abs_spread >= self.strong_spread_pct:
            # Strong spread: moderate boost
            strength *= 1.1
        
        # Cap maximum strength at 2.0
        return min(strength, 2.0)
    
    def _get_spread_severity(self, abs_spread_pct: float) -> str:
        """Get human-readable severity level."""
        if abs_spread_pct >= self.extreme_spread_pct:
            return "EXTREME"
        elif abs_spread_pct >= self.strong_spread_pct:
            return "STRONG"
        else:
            return "MODERATE"
    
    def get_detailed_analysis(
        self,
        futures_price: float,
        spot_prices: Dict[str, float],
        symbol: str = "UNKNOWN"
    ) -> Dict:
        """
        Get detailed analysis including individual exchange comparisons.
        
        Returns dict with:
        - signal: SpreadSignal or None
        - exchange_diffs: Dict of exchange -> diff %
        - avg_spot: Average spot price
        - spread_pct: Overall spread %
        """
        signal = self.analyze_spread(futures_price, spot_prices, symbol)
        
        # Calculate per-exchange differences
        exchange_diffs = {}
        for exchange, spot_price in spot_prices.items():
            if spot_price > 0:
                diff_pct = ((futures_price / spot_price) - 1.0) * 100.0
                exchange_diffs[exchange] = diff_pct
        
        valid_spots = [p for p in spot_prices.values() if p > 0]
        avg_spot = median(valid_spots) if len(valid_spots) >= 3 else (mean(valid_spots) if valid_spots else 0)
        
        spread_pct = ((futures_price / avg_spot) - 1.0) * 100.0 if avg_spot > 0 else 0
        
        return {
            "signal": signal,
            "exchange_diffs": exchange_diffs,
            "avg_spot": avg_spot,
            "spread_pct": spread_pct,
            "futures_price": futures_price,
        }


# Convenience function for quick checks
def check_spread(
    futures_price: float,
    spot_prices: Dict[str, float],
    min_spread: float = 3.0
) -> Optional[Tuple[str, float]]:
    """
    Quick spread check without creating SpreadSignal object.
    
    Returns:
        Tuple of (direction, strength) if significant, None otherwise
    """
    filter = SpotFuturesSpreadFilter(min_spread_pct=min_spread)
    signal = filter.analyze_spread(futures_price, spot_prices)
    
    if signal:
        return (signal.direction, signal.strength)
    return None
