"""
Signal Filters for IMBA Trading System.

Includes:
- Funding rate filter (avoid high funding costs)
- Altcoin market bias (overall market sentiment)
- Liquidation heatmap integration (optional)
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class SignalFilters:
    """
    Collection of signal filtering mechanisms.
    """
    
    @staticmethod
    def funding_rate_filter(
        symbol: str,
        side: str,  # "buy" or "sell"
        funding_rate: Optional[float] = None,
        threshold: float = 0.1
    ) -> bool:
        """
        Filter trades based on funding rate.
        
        Avoids opening positions when funding is unfavorable:
        - Don't open LONG if funding rate > threshold (paying too much)
        - Don't open SHORT if funding rate < -threshold (paying too much)
        
        Args:
            symbol: Trading pair
            side: Trade side ("buy" for LONG, "sell" for SHORT)
            funding_rate: Current funding rate (% per 8h)
            threshold: Funding rate threshold (default 0.1%)
            
        Returns:
            True if trade should be allowed, False to filter out
        """
        if funding_rate is None:
            # No funding data available - allow trade
            return True
        
        try:
            # Convert to percentage if needed
            funding_pct = float(funding_rate) * 100 if abs(funding_rate) < 1 else float(funding_rate)
            
            # Filter LONG if funding too high (longs pay shorts)
            if side.lower() == "buy" and funding_pct > threshold:
                logger.info(
                    f"Filtering LONG signal for {symbol}: "
                    f"funding rate {funding_pct:.3f}% exceeds threshold {threshold}%"
                )
                return False
            
            # Filter SHORT if funding too negative (shorts pay longs)
            if side.lower() == "sell" and funding_pct < -threshold:
                logger.info(
                    f"Filtering SHORT signal for {symbol}: "
                    f"funding rate {funding_pct:.3f}% below -threshold -{threshold}%"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Funding rate filter error: {e}")
            # On error, allow the trade (fail-open)
            return True
    
    @staticmethod
    def alt_context_bias(
        alt_symbols: List[str] = None,
        prices_data: Dict[str, List[float]] = None,
        influence: float = 0.35
    ) -> float:
        """
        Calculate altcoin market bias for signal enhancement.
        
        Uses price changes of major altcoins to gauge overall market sentiment.
        Positive bias = alts pumping (bullish)
        Negative bias = alts dumping (bearish)
        
        Args:
            alt_symbols: List of altcoin symbols to track
            prices_data: Dictionary of {symbol: [prices]} with recent prices
            influence: Maximum influence weight (default 0.35)
            
        Returns:
            Bias value between -influence and +influence
        """
        if alt_symbols is None:
            alt_symbols = ["ETHUSDT", "BNBUSDT", "SOLUSDT"]
        
        if prices_data is None:
            logger.warning("No price data provided for alt bias calculation")
            return 0.0
        
        try:
            alt_returns = []
            
            for symbol in alt_symbols:
                if symbol not in prices_data:
                    continue
                
                prices = prices_data[symbol]
                if len(prices) < 2:
                    continue
                
                # Calculate recent return
                pct_change = (prices[-1] - prices[-2]) / prices[-2]
                alt_returns.append(pct_change)
            
            if not alt_returns:
                return 0.0
            
            # Average return across altcoins
            mean_return = np.mean(alt_returns)
            
            # Tanh scaling to keep in reasonable range
            # Multiply by 500 to make percentage changes meaningful
            bias = np.tanh(mean_return * 500) * influence
            
            logger.debug(
                f"Alt market bias: {bias:.3f} "
                f"(mean return: {mean_return:.4f}, n={len(alt_returns)})"
            )
            
            return float(bias)
            
        except Exception as e:
            logger.error(f"Alt bias calculation error: {e}")
            return 0.0
    
    @staticmethod
    def liquidation_heatmap_filter(
        symbol: str,
        side: str,
        liquidation_data: Optional[Dict[str, Any]] = None,
        notional_threshold: float = 5_000_000
    ) -> Dict[str, Any]:
        """
        Analyze liquidation heatmap for trade opportunities.
        
        Identifies areas with high liquidation concentration that could
        act as magnets for price movement.
        
        Args:
            symbol: Trading pair
            side: Trade side ("buy" or "sell")
            liquidation_data: Real-time liquidation data from WebSocket
            notional_threshold: Minimum notional value for significant level
            
        Returns:
            Dictionary with liquidation analysis
        """
        if liquidation_data is None:
            return {
                "significant_levels": [],
                "total_notional": 0.0,
                "recommendation": "neutral"
            }
        
        try:
            # Extract liquidation levels
            liq_levels = liquidation_data.get("levels", [])
            
            # Filter significant levels
            significant = [
                level for level in liq_levels
                if level.get("notional", 0) >= notional_threshold
            ]
            
            # Calculate total liquidation pressure
            total_long_liq = sum(
                level["notional"] for level in significant
                if level.get("side") == "long"
            )
            total_short_liq = sum(
                level["notional"] for level in significant
                if level.get("side") == "short"
            )
            
            # Recommendation based on liquidation pressure
            recommendation = "neutral"
            if total_long_liq > total_short_liq * 1.5:
                # Many long liquidations ahead - bearish
                recommendation = "bearish"
            elif total_short_liq > total_long_liq * 1.5:
                # Many short liquidations ahead - bullish
                recommendation = "bullish"
            
            logger.info(
                f"Liquidation heatmap for {symbol}: "
                f"Long liq ${total_long_liq/1e6:.2f}M, "
                f"Short liq ${total_short_liq/1e6:.2f}M, "
                f"Recommendation: {recommendation}"
            )
            
            return {
                "significant_levels": significant,
                "total_long_liquidation": total_long_liq,
                "total_short_liquidation": total_short_liq,
                "total_notional": total_long_liq + total_short_liq,
                "recommendation": recommendation
            }
            
        except Exception as e:
            logger.error(f"Liquidation heatmap filter error: {e}")
            return {
                "significant_levels": [],
                "total_notional": 0.0,
                "recommendation": "neutral",
                "error": str(e)
            }
    
    @staticmethod
    def volume_filter(
        symbol: str,
        current_volume: float,
        avg_volume: float,
        min_ratio: float = 0.5
    ) -> bool:
        """
        Filter trades based on volume.
        
        Avoids trading during low liquidity periods.
        
        Args:
            symbol: Trading pair
            current_volume: Current period volume
            avg_volume: Average volume (e.g., 24h average)
            min_ratio: Minimum volume ratio (default 0.5 = 50% of average)
            
        Returns:
            True if volume is sufficient, False to filter out
        """
        if avg_volume == 0:
            return True
        
        try:
            volume_ratio = current_volume / avg_volume
            
            if volume_ratio < min_ratio:
                logger.info(
                    f"Filtering {symbol} due to low volume: "
                    f"{volume_ratio:.2%} of average (threshold: {min_ratio:.2%})"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Volume filter error: {e}")
            return True
    
    @staticmethod
    def volatility_filter(
        symbol: str,
        current_atr: float,
        avg_atr: float,
        min_ratio: float = 0.3,
        max_ratio: float = 3.0
    ) -> bool:
        """
        Filter trades based on volatility (ATR).
        
        Avoids trading in extreme volatility conditions.
        
        Args:
            symbol: Trading pair
            current_atr: Current ATR value
            avg_atr: Average ATR
            min_ratio: Minimum ATR ratio (too quiet)
            max_ratio: Maximum ATR ratio (too volatile)
            
        Returns:
            True if volatility is acceptable, False to filter out
        """
        if avg_atr == 0:
            return True
        
        try:
            atr_ratio = current_atr / avg_atr
            
            # Too quiet - avoid choppy conditions
            if atr_ratio < min_ratio:
                logger.info(
                    f"Filtering {symbol} due to low volatility: "
                    f"{atr_ratio:.2f}x average (min: {min_ratio}x)"
                )
                return False
            
            # Too volatile - avoid extreme risk
            if atr_ratio > max_ratio:
                logger.info(
                    f"Filtering {symbol} due to high volatility: "
                    f"{atr_ratio:.2f}x average (max: {max_ratio}x)"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Volatility filter error: {e}")
            return True


class FilterManager:
    """
    Manages all signal filters in one place.
    """
    
    def __init__(self, config):
        """
        Initialize filter manager.
        
        Args:
            config: Trading configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Filter parameters from config
        self.funding_threshold = getattr(config, "funding_filter_threshold", 0.1)
        self.alt_symbols = getattr(config, "alt_symbols", ["ETHUSDT", "BNBUSDT", "SOLUSDT"])
        self.alt_influence = getattr(config, "alt_influence", 0.35)
        self.liquidation_threshold = getattr(config, "liquidation_notional_threshold", 5_000_000)
        self.volume_min_ratio = getattr(config, "volume_min_ratio", 0.5)
        self.volatility_min_ratio = getattr(config, "volatility_min_ratio", 0.3)
        self.volatility_max_ratio = getattr(config, "volatility_max_ratio", 3.0)
    
    def apply_all_filters(
        self,
        symbol: str,
        side: str,
        funding_rate: Optional[float] = None,
        alt_prices: Optional[Dict[str, List[float]]] = None,
        current_volume: Optional[float] = None,
        avg_volume: Optional[float] = None,
        current_atr: Optional[float] = None,
        avg_atr: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Apply all filters and return results.
        
        Args:
            symbol: Trading pair
            side: Trade side ("buy" or "sell")
            funding_rate: Current funding rate
            alt_prices: Altcoin price data
            current_volume: Current volume
            avg_volume: Average volume
            current_atr: Current ATR
            avg_atr: Average ATR
            
        Returns:
            Dictionary with filter results and final decision
        """
        results = {
            "funding_pass": True,
            "volume_pass": True,
            "volatility_pass": True,
            "alt_bias": 0.0,
            "filters_passed": True,
            "reasons": []
        }
        
        # Funding rate filter
        if funding_rate is not None:
            results["funding_pass"] = SignalFilters.funding_rate_filter(
                symbol, side, funding_rate, self.funding_threshold
            )
            if not results["funding_pass"]:
                results["reasons"].append("High funding rate")
        
        # Volume filter
        if current_volume is not None and avg_volume is not None:
            results["volume_pass"] = SignalFilters.volume_filter(
                symbol, current_volume, avg_volume, self.volume_min_ratio
            )
            if not results["volume_pass"]:
                results["reasons"].append("Low volume")
        
        # Volatility filter
        if current_atr is not None and avg_atr is not None:
            results["volatility_pass"] = SignalFilters.volatility_filter(
                symbol, current_atr, avg_atr,
                self.volatility_min_ratio, self.volatility_max_ratio
            )
            if not results["volatility_pass"]:
                results["reasons"].append("Extreme volatility")
        
        # Alt bias (not a pass/fail, just context)
        if alt_prices is not None:
            results["alt_bias"] = SignalFilters.alt_context_bias(
                self.alt_symbols, alt_prices, self.alt_influence
            )
        
        # Final decision
        results["filters_passed"] = (
            results["funding_pass"] and
            results["volume_pass"] and
            results["volatility_pass"]
        )
        
        if not results["filters_passed"]:
            self.logger.info(
                f"Signal filtered out for {symbol} {side}: {', '.join(results['reasons'])}"
            )
        
        return results
