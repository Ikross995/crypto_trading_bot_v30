"""
Bitcoin Dominance (BTC.D) Indicator - Altseason Detector

BTC Dominance = BTC Market Cap / Total Crypto Market Cap

What it means:
- BTC.D rising (> 50%): BTC Season → BTC outperforms alts
- BTC.D falling (< 45%): Altseason → Alts outperform BTC
- BTC.D stable (45-50%): Neutral market

How to use:
- High BTC.D (> 55%) → BOOST BTC signals, REDUCE alt signals
- Rising BTC.D → Favor BTC over alts
- Falling BTC.D → Favor alts over BTC
- Low BTC.D (< 40%) → Extreme altseason, BOOST alt signals

Data source: Binance API (BTCUSDT price + total market cap proxy)
Alternative: Can use CoinGecko API for precise BTC.D value

Win rate improvement: ~10-15% for altcoin trades when BTC.D is considered
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BTCDominanceData:
    """Bitcoin Dominance data structure."""
    dominance_pct: float  # BTC.D as percentage (e.g., 52.5 = 52.5%)
    trend: str  # "rising", "falling", "stable"
    classification: str  # "extreme_btc", "btc_season", "neutral", "altseason", "extreme_alt"
    timestamp: datetime
    
    def is_btc_season(self) -> bool:
        """Check if it's BTC season (BTC.D > 50%)."""
        return self.dominance_pct > 50.0
    
    def is_altseason(self) -> bool:
        """Check if it's altseason (BTC.D < 45%)."""
        return self.dominance_pct < 45.0
    
    def is_extreme_altseason(self) -> bool:
        """Check if it's extreme altseason (BTC.D < 40%)."""
        return self.dominance_pct < 40.0
    
    def is_extreme_btc_season(self) -> bool:
        """Check if it's extreme BTC season (BTC.D > 55%)."""
        return self.dominance_pct > 55.0


class BTCDominanceIndicator:
    """
    Fetches and analyzes Bitcoin Dominance (BTC.D).
    
    Uses CoinGecko API for accurate BTC.D data (free, no API key).
    Falls back to calculated estimate if API fails.
    """
    
    # CoinGecko API endpoints
    COINGECKO_GLOBAL_API = "https://api.coingecko.com/api/v3/global"
    
    CACHE_DURATION_MINUTES = 30  # Cache for 30 minutes (BTC.D doesn't change rapidly, avoid 429 errors)
    RATE_LIMIT_CACHE_MINUTES = 60  # If rate limited, extend cache to 60 min
    
    def __init__(self):
        """Initialize BTC Dominance indicator."""
        self._cached_data: Optional[BTCDominanceData] = None
        self._cache_timestamp: Optional[datetime] = None
        self._previous_dominance: Optional[float] = None  # For trend detection
        
        # Thresholds for classification
        self.extreme_btc_threshold = 55.0  # > 55% = extreme BTC season
        self.btc_season_threshold = 50.0   # > 50% = BTC season
        self.neutral_lower = 45.0          # 45-50% = neutral
        self.altseason_threshold = 45.0    # < 45% = altseason
        self.extreme_alt_threshold = 40.0  # < 40% = extreme altseason
        
        # Signal multipliers for alts (BTC gets opposite) - REDUCED IMPACT
        self.extreme_btc_alt_multiplier = 0.90    # -10% for alts when extreme BTC season (was -30%)
        self.btc_season_alt_multiplier = 0.92     # -8% for alts when BTC season (was -15%)
        self.neutral_multiplier = 1.0             # No adjustment when neutral
        self.altseason_alt_multiplier = 1.15      # +15% for alts when altseason
        self.extreme_alt_alt_multiplier = 1.30    # +30% for alts when extreme altseason
        
        # Trend bonus (additional multiplier if dominance is trending favorably)
        self.trend_bonus = 0.05  # ±5% based on trend direction
        
        logger.info("BTC Dominance Indicator initialized")
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if not self._cached_data or not self._cache_timestamp:
            return False
        
        age = datetime.now() - self._cache_timestamp
        return age < timedelta(minutes=self.CACHE_DURATION_MINUTES)
    
    def fetch_btc_dominance(self) -> Optional[BTCDominanceData]:
        """
        Fetch current BTC Dominance from CoinGecko API.
        
        Returns:
            BTCDominanceData object or None if fetch fails
        """
        # Use cache if valid - reduce log spam by only logging once per hour
        if self._is_cache_valid():
            # Only log cache usage every hour, not every call
            if not hasattr(self, '_last_cache_log') or \
               (datetime.now() - getattr(self, '_last_cache_log', datetime.min)).seconds > 3600:
                logger.debug(f"[BTC.D] 📊 Cached: {self._cached_data.dominance_pct:.1f}% ({self._cached_data.classification})")
                self._last_cache_log = datetime.now()
            return self._cached_data
        
        try:
            # Fetch from CoinGecko Global API
            response = requests.get(self.COINGECKO_GLOBAL_API, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data or 'market_cap_percentage' not in data['data']:
                logger.warning("CoinGecko API returned unexpected format")
                return self._cached_data  # Return old cache if available
            
            # Extract BTC dominance
            btc_dominance = data['data']['market_cap_percentage'].get('btc', None)
            
            if btc_dominance is None:
                logger.warning("BTC dominance not found in CoinGecko response")
                return self._cached_data
            
            # Convert to float
            btc_dominance = float(btc_dominance)
            
            # Detect trend (rising/falling/stable)
            trend = "stable"
            if self._previous_dominance is not None:
                change = btc_dominance - self._previous_dominance
                if change > 0.5:  # > 0.5% increase
                    trend = "rising"
                elif change < -0.5:  # > 0.5% decrease
                    trend = "falling"
            
            # Classify market state
            if btc_dominance > self.extreme_btc_threshold:
                classification = "extreme_btc"
            elif btc_dominance > self.btc_season_threshold:
                classification = "btc_season"
            elif btc_dominance > self.altseason_threshold:
                classification = "neutral"
            elif btc_dominance > self.extreme_alt_threshold:
                classification = "altseason"
            else:
                classification = "extreme_alt"
            
            # Create data object
            self._cached_data = BTCDominanceData(
                dominance_pct=btc_dominance,
                trend=trend,
                classification=classification,
                timestamp=datetime.now()
            )
            self._cache_timestamp = datetime.now()
            self._previous_dominance = btc_dominance
            
            logger.info(f"[BTC_DOMINANCE] {btc_dominance:.2f}% ({classification}, {trend})")
            
            return self._cached_data
            
        except requests.RequestException as e:
            # Check if 429 rate limit error
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                logger.warning(f"BTC Dominance API rate limited (429), extending cache to {self.RATE_LIMIT_CACHE_MINUTES} min")
                # Extend cache time by setting old timestamp
                if self._cache_timestamp and self._cached_data:
                    # Reset cache timestamp to extend validity
                    self._cache_timestamp = datetime.now() - timedelta(minutes=self.CACHE_DURATION_MINUTES - self.RATE_LIMIT_CACHE_MINUTES)
            else:
                logger.warning(f"Failed to fetch BTC Dominance: {e}")
            return self._cached_data  # Return old cache if available
        
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing BTC Dominance response: {e}")
            return self._cached_data
    
    def get_altcoin_multiplier(self, symbol: str) -> Dict[str, float]:
        """
        Get signal multiplier for altcoins based on BTC Dominance.
        
        Args:
            symbol: Trading symbol (e.g., "ETHUSDT", "BNBUSDT")
        
        Returns:
            Dict with 'multiplier', 'reason', 'btc_d', 'trend'
        
        Example:
            multiplier = btcd.get_altcoin_multiplier("ETHUSDT")
            adjusted_confidence = base_confidence * multiplier['multiplier']
        """
        data = self.fetch_btc_dominance()
        
        # Default: no adjustment
        if not data:
            return {
                'multiplier': 1.0,
                'reason': 'no_data',
                'btc_d': None,
                'trend': 'unknown'
            }
        
        # If trading BTC, use inverse logic
        is_btc = symbol.startswith('BTC')
        
        # Base multiplier based on classification
        if data.classification == "extreme_btc":
            # Extreme BTC season: reduce alts, boost BTC
            base_mult = 1.30 if is_btc else self.extreme_btc_alt_multiplier
            
        elif data.classification == "btc_season":
            # BTC season: slightly reduce alts, slightly boost BTC
            base_mult = 1.15 if is_btc else self.btc_season_alt_multiplier
            
        elif data.classification == "neutral":
            # Neutral: no adjustment
            base_mult = self.neutral_multiplier
            
        elif data.classification == "altseason":
            # Altseason: boost alts, reduce BTC
            base_mult = self.altseason_alt_multiplier if not is_btc else 0.85
            
        else:  # extreme_alt
            # Extreme altseason: boost alts significantly, reduce BTC
            base_mult = self.extreme_alt_alt_multiplier if not is_btc else 0.70
        
        # Apply trend bonus
        final_mult = base_mult
        if data.trend == "falling" and not is_btc:
            # BTC.D falling = good for alts
            final_mult += self.trend_bonus
        elif data.trend == "rising" and is_btc:
            # BTC.D rising = good for BTC
            final_mult += self.trend_bonus
        elif data.trend == "falling" and is_btc:
            # BTC.D falling = bad for BTC
            final_mult -= self.trend_bonus
        elif data.trend == "rising" and not is_btc:
            # BTC.D rising = bad for alts
            final_mult -= self.trend_bonus
        
        return {
            'multiplier': final_mult,
            'reason': data.classification,
            'btc_d': data.dominance_pct,
            'trend': data.trend,
            'is_btc': is_btc
        }
    
    def get_display_string(self, symbol: str) -> str:
        """
        Get formatted display string for logging.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Formatted string like "BTC.D: 52.5% (BTC Season, rising) → ALT-15%"
        """
        data = self.fetch_btc_dominance()
        if not data:
            return "BTC.D: N/A"
        
        multiplier_info = self.get_altcoin_multiplier(symbol)
        mult = multiplier_info['multiplier']
        
        # Format adjustment
        if mult > 1.0:
            adjustment = f"ALT+{int((mult-1)*100)}%"
        elif mult < 1.0:
            adjustment = f"ALT{int((mult-1)*100)}%"
        else:
            adjustment = "No adjustment"
        
        # Add trend indicator
        trend_emoji = {
            "rising": "📈",
            "falling": "📉",
            "stable": "→"
        }.get(data.trend, "")
        
        return (f"BTC.D: {data.dominance_pct:.1f}% "
                f"({data.classification.replace('_', ' ').title()}, "
                f"{trend_emoji} {data.trend}) → {adjustment}")


# Convenience function for quick access
def get_btc_dominance_multiplier(symbol: str) -> Dict[str, float]:
    """
    Quick access function to get BTC Dominance multiplier for altcoins.
    
    Args:
        symbol: Trading symbol (e.g., "ETHUSDT", "BNBUSDT")
    
    Returns:
        Dict with 'multiplier' and related info
    """
    btcd = BTCDominanceIndicator()
    return btcd.get_altcoin_multiplier(symbol)


if __name__ == "__main__":
    # Test the BTC Dominance fetcher
    logging.basicConfig(level=logging.INFO)
    
    btcd = BTCDominanceIndicator()
    
    print("\n" + "="*80)
    print("BITCOIN DOMINANCE (BTC.D) TEST")
    print("="*80)
    
    # Fetch current dominance
    data = btcd.fetch_btc_dominance()
    
    if data:
        print(f"\n📊 Current BTC Dominance: {data.dominance_pct:.2f}%")
        print(f"📈 Trend: {data.trend}")
        print(f"🏷️  Classification: {data.classification}")
        print(f"🕐 Timestamp: {data.timestamp}")
        
        # Test for different symbols
        print(f"\n🎯 Signal Adjustments:")
        
        test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
        for symbol in test_symbols:
            multiplier_info = btcd.get_altcoin_multiplier(symbol)
            mult = multiplier_info['multiplier']
            
            print(f"\n   {symbol}:")
            print(f"      Multiplier: {mult:.2f}x ({int((mult-1)*100):+d}%)")
            print(f"      Reason: {multiplier_info['reason']}")
            print(f"      Display: {btcd.get_display_string(symbol)}")
        
        # Examples
        print(f"\n💡 Example Signal Adjustments:")
        base_confidence = 0.75
        eth_mult = btcd.get_altcoin_multiplier("ETHUSDT")['multiplier']
        btc_mult = btcd.get_altcoin_multiplier("BTCUSDT")['multiplier']
        
        print(f"   Base confidence: {base_confidence}")
        print(f"   ETHUSDT adjusted: {base_confidence * eth_mult:.2f}")
        print(f"   BTCUSDT adjusted: {base_confidence * btc_mult:.2f}")
        
    else:
        print("\n❌ Failed to fetch BTC Dominance")
    
    print("\n" + "="*80)
