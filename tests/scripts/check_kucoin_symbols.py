"""Quick script to check available KuCoin Futures symbols."""
import ccxt

try:
    exchange = ccxt.kucoinfutures({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    print("Loading markets...")
    markets = exchange.load_markets()
    
    print(f"\nTotal markets: {len(markets)}")
    
    # Find all active futures
    print("\n=== Searching for perpetual futures ===")
    perpetual = []
    btc_related = []
    
    for symbol, market in markets.items():
        if market.get('active', False):
            market_type = market.get('type')
            contract_type = market.get('contract', 'unknown')
            linear = market.get('linear', False)
            
            # Look for perpetual or swap
            if 'perpetual' in str(contract_type).lower() or 'swap' in str(contract_type).lower():
                perpetual.append((symbol, market.get('quote'), contract_type))
                if 'BTC' in symbol.upper():
                    btc_related.append(symbol)
            
            # Also check BTC in any type
            if 'BTC' in symbol.upper():
                btc_related.append(symbol)
                print(f"BTC market: {symbol} | type: {market_type} | contract: {contract_type} | quote: {market.get('quote')} | linear: {linear}")
    
    print(f"\n=== Perpetual futures (first 20) ===")
    for sym, quote, ctype in sorted(perpetual)[:20]:
        print(f"  {sym} | quote: {quote} | contract: {ctype}")
    
    print(f"\n=== All BTC-related symbols ===")
    for s in sorted(set(btc_related)):
        m = markets[s]
        print(f"  {s} | type: {m.get('type')} | quote: {m.get('quote')} | contract: {m.get('contract')}")
    
except Exception as e:
    print(f"Error: {e}")
