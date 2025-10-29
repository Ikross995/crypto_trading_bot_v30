"""
Fear & Greed Index Integration

Uses the Crypto Fear & Greed Index from Alternative.me to gauge market sentiment
and adjust trading signals accordingly.

The Fear & Greed Index is a contrarian indicator:
- Extreme Fear (0-25): Market oversold, good time to BUY
- Fear (25-45): Cautious sentiment, slight buy bias
- Neutral (45-55): Normal market conditions
- Greed (55-75): Cautious sentiment, slight sell bias
- Extreme Greed (75-100): Market overbought, good time to SELL

API: https://api.alternative.me/fng/
- Free, no API key required
- Updates once per day
- Returns index value 0-100 and classification

Usage:
    fng = FearGreedIndex()
    multiplier = fng.get_signal_multiplier()
    adjusted_confidence = base_confidence * multiplier['buy']
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FearGreedData:
    """Fear & Greed Index data structure."""
    value: int  # 0-100
    classification: str  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    timestamp: datetime
    
    def is_extreme_fear(self) -> bool:
        return self.value < 25
    
    def is_fear(self) -> bool:
        return 25 <= self.value < 45
    
    def is_neutral(self) -> bool:
        return 45 <= self.value < 55
    
    def is_greed(self) -> bool:
        return 55 <= self.value < 75
    
    def is_extreme_greed(self) -> bool:
        return self.value >= 75


class FearGreedIndex:
    """
    Fetches and caches Crypto Fear & Greed Index data.
    
    The index is updated once per day, so we cache it for 1 hour
    to avoid unnecessary API calls.
    """
    
    API_URL = "https://api.alternative.me/fng/"
    CACHE_DURATION_HOURS = 1  # Cache for 1 hour
    
    def __init__(self):
        """Initialize Fear & Greed Index fetcher."""
        self._cached_data: Optional[FearGreedData] = None
        self._cache_timestamp: Optional[datetime] = None
        
        # Configuration for signal adjustment
        self.extreme_fear_buy_boost = 1.20  # +20% confidence on BUY in extreme fear
        self.extreme_fear_sell_reduce = 0.80  # -20% confidence on SELL in extreme fear
        
        self.fear_buy_boost = 1.10  # +10% on BUY in fear
        self.fear_sell_reduce = 0.90  # -10% on SELL in fear
        
        self.greed_buy_reduce = 0.90  # -10% on BUY in greed
        self.greed_sell_boost = 1.10  # +10% on SELL in greed
        
        self.extreme_greed_buy_reduce = 0.80  # -20% on BUY in extreme greed
        self.extreme_greed_sell_boost = 1.20  # +20% on SELL in extreme greed
        
        logger.info("Fear & Greed Index initialized")
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if not self._cached_data or not self._cache_timestamp:
            return False
        
        age = datetime.now() - self._cache_timestamp
        return age < timedelta(hours=self.CACHE_DURATION_HOURS)
    
    def fetch_index(self) -> Optional[FearGreedData]:
        """
        Fetch current Fear & Greed Index from API.
        
        Returns:
            FearGreedData object or None if fetch fails
        """
        # Use cache if valid - reduce log spam by only logging once per hour
        if self._is_cache_valid():
            # Only log cache usage every hour, not every call
            if not hasattr(self, '_last_cache_log') or \
               (datetime.now() - getattr(self, '_last_cache_log', datetime.min)).seconds > 3600:
                logger.debug(f"[F&G] 📊 Cached: {self._cached_data.value}/100 ({self._cached_data.classification})")
                self._last_cache_log = datetime.now()
            return self._cached_data
        
        try:
            # Fetch from API
            response = requests.get(self.API_URL, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data or len(data['data']) == 0:
                logger.warning("Fear & Greed API returned empty data")
                return self._cached_data  # Return old cache if available
            
            # Parse response
            fng_data = data['data'][0]
            value = int(fng_data['value'])
            classification = fng_data['value_classification']
            timestamp = datetime.fromtimestamp(int(fng_data['timestamp']))
            
            # Create data object
            self._cached_data = FearGreedData(
                value=value,
                classification=classification,
                timestamp=timestamp
            )
            self._cache_timestamp = datetime.now()
            
            logger.info(f"[FEAR_GREED] Index: {value}/100 ({classification})")
            
            return self._cached_data
            
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch Fear & Greed Index: {e}")
            return self._cached_data  # Return old cache if available
        
        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"Error parsing Fear & Greed Index response: {e}")
            return self._cached_data
    
    def get_signal_multiplier(self) -> Dict[str, float]:
        """
        Get signal multipliers based on current Fear & Greed Index.
        
        Returns:
            Dict with 'buy' and 'sell' multipliers
            
        Example:
            multiplier = fng.get_signal_multiplier()
            adjusted_buy = base_confidence * multiplier['buy']
            adjusted_sell = base_confidence * multiplier['sell']
        """
        data = self.fetch_index()
        
        # Default: no adjustment
        if not data:
            return {'buy': 1.0, 'sell': 1.0, 'reason': 'no_data'}
        
        # Extreme Fear: BOOST BUY, REDUCE SELL
        if data.is_extreme_fear():
            return {
                'buy': self.extreme_fear_buy_boost,
                'sell': self.extreme_fear_sell_reduce,
                'reason': 'extreme_fear',
                'value': data.value
            }
        
        # Fear: Slight buy bias
        elif data.is_fear():
            return {
                'buy': self.fear_buy_boost,
                'sell': self.fear_sell_reduce,
                'reason': 'fear',
                'value': data.value
            }
        
        # Neutral: No adjustment
        elif data.is_neutral():
            return {
                'buy': 1.0,
                'sell': 1.0,
                'reason': 'neutral',
                'value': data.value
            }
        
        # Greed: Slight sell bias
        elif data.is_greed():
            return {
                'buy': self.greed_buy_reduce,
                'sell': self.greed_sell_boost,
                'reason': 'greed',
                'value': data.value
            }
        
        # Extreme Greed: REDUCE BUY, BOOST SELL
        else:  # extreme_greed
            return {
                'buy': self.extreme_greed_buy_reduce,
                'sell': self.extreme_greed_sell_boost,
                'reason': 'extreme_greed',
                'value': data.value
            }
    
    def should_boost_buy(self) -> Tuple[bool, float]:
        """
        Check if BUY signals should be boosted.
        
        Returns:
            (should_boost, multiplier)
        """
        multiplier_data = self.get_signal_multiplier()
        buy_mult = multiplier_data['buy']
        return buy_mult > 1.0, buy_mult
    
    def should_boost_sell(self) -> Tuple[bool, float]:
        """
        Check if SELL signals should be boosted.
        
        Returns:
            (should_boost, multiplier)
        """
        multiplier_data = self.get_signal_multiplier()
        sell_mult = multiplier_data['sell']
        return sell_mult > 1.0, sell_mult
    
    def get_display_string(self) -> str:
        """
        Get formatted display string for logging.
        
        Returns:
            Formatted string like "Fear & Greed: 25 (Extreme Fear) → BUY+20%"
        """
        data = self.fetch_index()
        if not data:
            return "Fear & Greed: N/A"
        
        multiplier_data = self.get_signal_multiplier()
        reason = multiplier_data['reason']
        
        if reason == 'extreme_fear':
            adjustment = f"BUY+{int((multiplier_data['buy']-1)*100)}%"
        elif reason == 'fear':
            adjustment = f"BUY+{int((multiplier_data['buy']-1)*100)}%"
        elif reason == 'neutral':
            adjustment = "No adjustment"
        elif reason == 'greed':
            adjustment = f"SELL+{int((multiplier_data['sell']-1)*100)}%"
        else:  # extreme_greed
            adjustment = f"SELL+{int((multiplier_data['sell']-1)*100)}%"
        
        return f"Fear & Greed: {data.value}/100 ({data.classification}) → {adjustment}"


# Convenience function for quick access
def get_fear_greed_multiplier() -> Dict[str, float]:
    """
    Quick access function to get Fear & Greed multipliers.
    
    Returns:
        Dict with 'buy' and 'sell' multipliers
    """
    fng = FearGreedIndex()
    return fng.get_signal_multiplier()


if __name__ == "__main__":
    # Test the Fear & Greed Index fetcher
    logging.basicConfig(level=logging.INFO)
    
    fng = FearGreedIndex()
    
    print("\n" + "="*80)
    print("CRYPTO FEAR & GREED INDEX TEST")
    print("="*80)
    
    # Fetch current index
    data = fng.fetch_index()
    
    if data:
        print(f"\n📊 Current Index: {data.value}/100")
        print(f"📈 Classification: {data.classification}")
        print(f"🕐 Timestamp: {data.timestamp}")
        print(f"\n{fng.get_display_string()}")
        
        # Get multipliers
        multipliers = fng.get_signal_multiplier()
        print(f"\n🎯 Signal Adjustments:")
        print(f"   BUY multiplier:  {multipliers['buy']:.2f}x ({int((multipliers['buy']-1)*100):+d}%)")
        print(f"   SELL multiplier: {multipliers['sell']:.2f}x ({int((multipliers['sell']-1)*100):+d}%)")
        print(f"   Reason: {multipliers['reason']}")
        
        # Examples
        print(f"\n💡 Example Signal Adjustments:")
        base_confidence = 0.75
        print(f"   Base confidence: {base_confidence}")
        print(f"   Adjusted BUY:  {base_confidence * multipliers['buy']:.2f}")
        print(f"   Adjusted SELL: {base_confidence * multipliers['sell']:.2f}")
        
    else:
        print("\n❌ Failed to fetch Fear & Greed Index")
    
    print("\n" + "="*80)
