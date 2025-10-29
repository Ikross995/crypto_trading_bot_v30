"""
Multi-source price fetcher for informational display only.
Fetches prices from:
- Exchanges: Bybit, OKX, Kraken (spot)
- Aggregators: CoinGecko, CoinMarketCap

Does NOT integrate with trading signals - just shows prices.
"""

import requests
import time
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Simple in-memory cache (symbol -> (prices, timestamp))
_price_cache: Dict[str, Tuple[Dict[str, Optional[float]], float]] = {}
CACHE_TTL = 60  # Cache for 60 seconds


class ExchangePriceFetcher:
    """Fetch current prices from multiple sources (exchanges + aggregators)."""
    
    def __init__(self):
        self.timeout = 2  # 2 seconds timeout for requests
    
    def _map_symbol_binance(self, symbol: str) -> str:
        """Binance: BTCUSDT"""
        return symbol
    
    def _map_symbol_bybit(self, symbol: str) -> str:
        """Bybit: BTCUSDT"""
        return symbol
    
    def _map_symbol_okx(self, symbol: str) -> str:
        """OKX: BTC-USDT"""
        base = symbol.replace('USDT', '')
        return f"{base}-USDT"
    
    def _map_symbol_kraken(self, symbol: str) -> str:
        """Kraken: XXBTZUSD"""
        mapping = {
            'BTCUSDT': 'XXBTZUSD',
            'ETHUSDT': 'XETHZUSD',
            'BNBUSDT': 'BNBUSD',
            'SOLUSDT': 'SOLUSD',
            'XRPUSDT': 'XXRPZUSD',
            'ADAUSDT': 'ADAUSD',
            'DOGEUSDT': 'XDGUSD',
            'AVAXUSDT': 'AVAXUSD',
            'LINKUSDT': 'LINKUSD',
            'MATICUSDT': 'MATICUSD',
        }
        return mapping.get(symbol, symbol)
    
    def _map_symbol_coingecko(self, symbol: str) -> str:
        """CoinGecko: bitcoin, ethereum, etc."""
        mapping = {
            'BTCUSDT': 'bitcoin',
            'ETHUSDT': 'ethereum',
            'BNBUSDT': 'binancecoin',
            'SOLUSDT': 'solana',
            'XRPUSDT': 'ripple',
            'ADAUSDT': 'cardano',
            'DOGEUSDT': 'dogecoin',
            'AVAXUSDT': 'avalanche-2',
            'LINKUSDT': 'chainlink',
            'MATICUSDT': 'matic-network',
        }
        return mapping.get(symbol, 'bitcoin')
    
    def _map_symbol_cmc(self, symbol: str) -> str:
        """CoinMarketCap: BTC, ETH, etc."""
        return symbol.replace('USDT', '')
    
    def _fetch_bybit(self, symbol: str) -> Optional[float]:
        """Fetch from Bybit spot."""
        try:
            mapped = self._map_symbol_bybit(symbol)
            url = f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={mapped}"
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('result', {}).get('list'):
                    return float(data['result']['list'][0]['lastPrice'])
        except Exception as e:
            logger.debug(f"Bybit fetch failed: {e}")
        return None
    
    def _fetch_okx(self, symbol: str) -> Optional[float]:
        """Fetch from OKX spot."""
        try:
            mapped = self._map_symbol_okx(symbol)
            url = f"https://www.okx.com/api/v5/market/ticker?instId={mapped}"
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('data'):
                    return float(data['data'][0]['last'])
        except Exception as e:
            logger.debug(f"OKX fetch failed: {e}")
        return None
    
    def _fetch_kraken(self, symbol: str) -> Optional[float]:
        """Fetch from Kraken spot."""
        try:
            mapped = self._map_symbol_kraken(symbol)
            url = f"https://api.kraken.com/0/public/Ticker?pair={mapped}"
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('result'):
                    key = list(data['result'].keys())[0]
                    return float(data['result'][key]['c'][0])
        except Exception as e:
            logger.debug(f"Kraken fetch failed: {e}")
        return None
    
    def _fetch_coingecko(self, symbol: str) -> Optional[float]:
        """Fetch from CoinGecko (FREE API, no key needed)."""
        try:
            coin_id = self._map_symbol_coingecko(symbol)
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if coin_id in data and 'usd' in data[coin_id]:
                    return float(data[coin_id]['usd'])
        except Exception as e:
            logger.debug(f"CoinGecko fetch failed: {e}")
        return None
    
    def _fetch_coinmarketcap(self, symbol: str, api_key: Optional[str] = None) -> Optional[float]:
        """Fetch from CoinMarketCap (requires API key - optional)."""
        if not api_key:
            logger.debug("CoinMarketCap API key not provided, skipping")
            return None
        try:
            symbol_cmc = self._map_symbol_cmc(symbol)
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            headers = {
                'X-CMC_PRO_API_KEY': api_key,
                'Accept': 'application/json'
            }
            params = {'symbol': symbol_cmc, 'convert': 'USD'}
            resp = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if 'data' in data and symbol_cmc in data['data']:
                    return float(data['data'][symbol_cmc]['quote']['USD']['price'])
        except Exception as e:
            logger.debug(f"CoinMarketCap fetch failed: {e}")
        return None
    
    def fetch_all_sync(self, symbol: str, cmc_api_key: Optional[str] = None) -> Dict[str, Optional[float]]:
        """
        Fetch prices from all sources (exchanges + aggregators) synchronously.
        Uses caching to avoid API spam (60s TTL).
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            cmc_api_key: Optional CoinMarketCap API key
        
        Returns:
            Dict with source names and prices (None if failed):
            {
                'bybit': float or None,
                'okx': float or None,
                'kraken': float or None,
                'coingecko': float or None,
                'coinmarketcap': float or None
            }
        """
        global _price_cache
        
        # Check cache first
        now = time.time()
        if symbol in _price_cache:
            cached_prices, cached_time = _price_cache[symbol]
            if now - cached_time < CACHE_TTL:
                logger.debug(f"Using cached prices for {symbol} (age: {now - cached_time:.1f}s)")
                return cached_prices
        
        # Fetch fresh prices
        logger.debug(f"Fetching prices for {symbol} from multiple sources...")
        prices = {
            'bybit': self._fetch_bybit(symbol),
            'okx': self._fetch_okx(symbol),
            'kraken': self._fetch_kraken(symbol),
            'coingecko': self._fetch_coingecko(symbol),
            'coinmarketcap': self._fetch_coinmarketcap(symbol, cmc_api_key)
        }
        
        # Cache result
        _price_cache[symbol] = (prices, now)
        
        return prices


def format_price(price: float) -> str:
    """
    Smart price formatting based on value.
    
    Args:
        price: Price value
    
    Returns:
        Formatted price string like "$0.000123" or "$43,250.50"
    """
    if price < 0.01:
        # Very small prices: show up to 6 decimals, strip trailing zeros
        formatted = f"{price:.6f}".rstrip('0').rstrip('.')
        return f"${formatted}"
    elif price < 1:
        # Small prices (0.01-1.00): show 4 decimals
        return f"${price:.4f}"
    elif price < 10:
        # Medium prices (1-10): show 3 decimals
        return f"${price:.3f}"
    else:
        # Large prices (>10): show 2 decimals with comma separator
        return f"${price:,.2f}"


def format_exchange_prices(prices: Dict[str, Optional[float]], base_price: float) -> Tuple[str, str]:
    """
    Format exchange and aggregator prices for display.
    
    Args:
        prices: Dict from fetch_all_sync()
        base_price: Current Binance price for comparison
    
    Returns:
        Tuple of (exchanges_str, aggregators_str):
        - exchanges_str: "Bybit $95,100 (+0.21%) | OKX $95,050 (+0.05%)" or ""
        - aggregators_str: "CoinGecko $95,120 (+0.02%) | CMC $95,080 (+0.01%)" or ""
    """
    # Format exchanges (spot)
    exchanges_parts = []
    for exchange in ['bybit', 'okx', 'kraken']:
        price = prices.get(exchange)
        if price is not None:
            diff_pct = ((price - base_price) / base_price) * 100
            sign = '+' if diff_pct >= 0 else ''
            exchanges_parts.append(f"{exchange.capitalize()} {format_price(price)} ({sign}{diff_pct:.2f}%)")
    
    # Format aggregators
    aggregators_parts = []
    agg_mapping = {
        'coingecko': 'CoinGecko',
        'coinmarketcap': 'CMC'
    }
    for key, display_name in agg_mapping.items():
        price = prices.get(key)
        if price is not None:
            diff_pct = ((price - base_price) / base_price) * 100
            sign = '+' if diff_pct >= 0 else ''
            aggregators_parts.append(f"{display_name} {format_price(price)} ({sign}{diff_pct:.2f}%)")
    
    exchanges_str = " | ".join(exchanges_parts) if exchanges_parts else ""
    aggregators_str = " | ".join(aggregators_parts) if aggregators_parts else ""
    
    return exchanges_str, aggregators_str


# Singleton instance
_fetcher = None

def get_price_fetcher() -> ExchangePriceFetcher:
    """Get singleton price fetcher."""
    global _fetcher
    if _fetcher is None:
        _fetcher = ExchangePriceFetcher()
    return _fetcher
