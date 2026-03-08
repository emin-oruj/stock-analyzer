from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
import os
import requests

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB = "https://finnhub.io/api/v1"
COINGECKO = "https://api.coingecko.com/api/v3"
COINGECKO_KEY = os.getenv("COINGECKO_API_KEY")

# Common crypto symbol -> CoinCap ID mapping
CRYPTO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "BNB": "binancecoin",
    "XRP": "ripple", "ADA": "cardano", "DOGE": "dogecoin", "AVAX": "avalanche-2",
    "DOT": "polkadot", "MATIC": "matic-network", "LINK": "chainlink",
    "UNI": "uniswap", "LTC": "litecoin", "BCH": "bitcoin-cash", "ATOM": "cosmos",
    "XLM": "stellar", "ALGO": "algorand", "VET": "vechain", "FIL": "filecoin",
    "TRX": "tron", "SHIB": "shiba-inu", "SUI": "sui", "APT": "aptos",
    "ARB": "arbitrum", "NEAR": "near", "ICP": "internet-computer",
    "HBAR": "hedera-hashgraph", "INJ": "injective-protocol", "OP": "optimism",
    "RENDER": "render-token", "RNDR": "render-token", "FET": "fetch-ai",
    "WLD": "worldcoin-wld", "GRT": "the-graph", "SAND": "the-sandbox",
    "MANA": "decentraland", "AAVE": "aave", "MKR": "maker", "SNX": "havven",
    "CRV": "curve-dao-token", "LDO": "lido-dao", "RUNE": "thorchain",
    "PENDLE": "pendle", "JUP": "jupiter-exchange-solana", "WIF": "dogwifcoin",
    "BONK": "bonk", "PEPE": "pepe", "FLOKI": "floki",
}

app = FastAPI()


class AnalyzeRequest(BaseModel):
    ticker: str


def is_crypto(ticker: str) -> bool:
    return ticker.upper() in CRYPTO_IDS


def fetch_crypto_data(ticker: str) -> dict:
    coin_id = CRYPTO_IDS.get(ticker.upper())
    if not coin_id:
        raise ValueError(f"Crypto '{ticker}' not supported. Try BTC, ETH, SOL, etc.")

    r = requests.get(
        f"{COINGECKO}/coins/{coin_id}",
        params={"localization": "false", "tickers": "false", "community_data": "false", "developer_data": "false"},
        headers={"x-cg-demo-api-key": COINGECKO_KEY},
        timeout=15
    )
    if not r.ok:
        raise ValueError(f"Could not fetch data for {ticker}. Status: {r.status_code}")

    d = r.json()
    market = d.get("market_data", {})

    def g(obj, *keys, default="N/A"):
        for k in keys:
            obj = obj.get(k) if isinstance(obj, dict) else None
        if obj is None:
            return default
        return round(obj, 6) if isinstance(obj, float) and obj < 1 else (round(obj, 2) if isinstance(obj, float) else obj)

    return {
        "type": "crypto",
        "name": d.get("name", ticker),
        "symbol": ticker.upper(),
        "description": (d.get("description", {}).get("en") or "")[:500],
        "categories": ", ".join(d.get("categories", [])[:3]) or "Cryptocurrency",
        "price": g(market, "current_price", "usd"),
        "market_cap": g(market, "market_cap", "usd"),
        "market_cap_rank": d.get("market_cap_rank", "N/A"),
        "total_volume_24h": g(market, "total_volume", "usd"),
        "high_24h": g(market, "high_24h", "usd"),
        "low_24h": g(market, "low_24h", "usd"),
        "price_change_24h": g(market, "price_change_percentage_24h"),
        "price_change_7d": g(market, "price_change_percentage_7d"),
        "price_change_30d": g(market, "price_change_percentage_30d"),
        "ath": g(market, "ath", "usd"),
        "ath_change_pct": g(market, "ath_change_percentage", "usd"),
        "atl": g(market, "atl", "usd"),
        "circulating_supply": g(market, "circulating_supply"),
        "total_supply": g(market, "total_supply"),
        "max_supply": g(market, "max_supply"),
        "fully_diluted_valuation": g(market, "fully_diluted_valuation", "usd"),
    }


def fetch_stock_data(ticker: str) -> dict:
    h = {"X-Finnhub-Token": FINNHUB_KEY}

    profile = requests.get(f"{FINNHUB}/stock/profile2", params={"symbol": ticker}, headers=h, timeout=10).json()
    quote   = requests.get(f"{FINNHUB}/quote",          params={"symbol": ticker}, headers=h, timeout=10).json()
    metrics = requests.get(f"{FINNHUB}/stock/metric",   params={"symbol": ticker, "metric": "all"}, headers=h, timeout=10).json().get("metric", {})
    target  = requests.get(f"{FINNHUB}/stock/price-target", params={"symbol": ticker}, headers=h, timeout=10).json()

    if not profile.get("name"):
        raise ValueError(f"Ticker '{ticker}' not found. Check the symbol and try again.")

    def g(d, key, default="N/A"):
        val = d.get(key)
        if val is None or val == "" or val == 0:
            return default
        return round(val, 4) if isinstance(val, float) else val

    market_cap_raw = g(profile, "marketCapitalization")
    market_cap = int(market_cap_raw * 1_000_000) if market_cap_raw != "N/A" else "N/A"

    return {
        "type": "stock",
        "name":             g(profile, "name", ticker),
        "sector":           g(profile, "finnhubIndustry", "N/A"),
        "industry":         g(profile, "finnhubIndustry", "N/A"),
        "country":          g(profile, "country", "N/A"),
        "exchange":         g(profile, "exchange", "N/A"),
        "currency":         g(profile, "currency", "USD"),
        "employees":        g(profile, "employeeTotal", "N/A"),
        "summary":          f"{g(profile,'name',ticker)} operates in the {g(profile,'finnhubIndustry','')} industry, listed on {g(profile,'exchange','')} ({g(profile,'country','')}).",
        "price":            g(quote, "c"),
        "market_cap":       market_cap,
        "week52_high":      g(metrics, "52WeekHigh"),
        "week52_low":       g(metrics, "52WeekLow"),
        "analyst_target":   g(target, "targetMean"),
        "beta":             g(metrics, "beta"),
        "pe_ratio":         g(metrics, "peNormalizedAnnual"),
        "forward_pe":       g(metrics, "peTTM"),
        "peg_ratio":        g(metrics, "pegAnnual"),
        "ps_ratio":         g(metrics, "psAnnual"),
        "pb_ratio":         g(metrics, "pbAnnual"),
        "ev_ebitda":        g(metrics, "currentEv/freeCashFlowAnnual"),
        "revenue":          g(metrics, "revenuePerShareTTM"),
        "revenue_growth":   g(metrics, "revenueGrowthQuarterlyYoy"),
        "gross_margins":    g(metrics, "grossMarginAnnual"),
        "operating_margins":g(metrics, "operatingMarginAnnual"),
        "profit_margins":   g(metrics, "netProfitMarginAnnual"),
        "roe":              g(metrics, "roeRfy"),
        "roa":              g(metrics, "roaRfy"),
        "debt_to_equity":   g(metrics, "totalDebt/totalEquityAnnual"),
        "current_ratio":    g(metrics, "currentRatioAnnual"),
        "free_cashflow":    g(metrics, "freeCashFlowAnnual"),
        "dividend_yield":   g(metrics, "dividendYieldIndicatedAnnual"),
    }


def build_crypto_prompt(ticker: str, d: dict) -> str:
    def fmt(val):
        if isinstance(val, (int, float)) and val != "N/A":
            if val >= 1_000_000_000:
                return f"${val/1_000_000_000:.2f}B"
            elif val >= 1_000_000:
                return f"${val/1_000_000:.2f}M"
            return f"${val:,.4f}" if val < 1 else f"${val:,.2f}"
        return str(val)

    return f"""You are a senior professional crypto market analyst. Analyze the following cryptocurrency using the real-time market data below. Be precise, honest, and data-driven.

CRYPTO: {ticker} — {d['name']}
Categories: {d['categories']}
Market Cap Rank: #{d['market_cap_rank']}

PRICE & MARKET DATA:
- Current Price: {fmt(d['price'])}
- Market Cap: {fmt(d['market_cap'])}
- 24h Volume: {fmt(d['total_volume_24h'])}
- 24h Range: {fmt(d['low_24h'])} – {fmt(d['high_24h'])}
- Fully Diluted Valuation: {fmt(d['fully_diluted_valuation'])}

PRICE PERFORMANCE:
- 24h Change: {d['price_change_24h']}%
- 7d Change: {d['price_change_7d']}%
- 30d Change: {d['price_change_30d']}%
- All-Time High: {fmt(d['ath'])} ({d['ath_change_pct']}% from ATH)
- All-Time Low: {fmt(d['atl'])}

SUPPLY:
- Circulating Supply: {d['circulating_supply']}
- Total Supply: {d['total_supply']}
- Max Supply: {d['max_supply']}

DESCRIPTION:
{d['description'] or 'N/A'}

---
Provide a comprehensive crypto analysis in the following markdown structure:

## 1. What Is It?
Explain the project, its purpose, technology, and use case.

## 2. Market Position & Tokenomics
Assess market cap rank, supply dynamics, inflation/deflation mechanics.

## 3. Price Analysis
Current price vs ATH, recent momentum, support/resistance levels.

## 4. Competitive Edge
What makes this crypto unique vs competitors?

## 5. Key Risks
Top 3–5 risks: regulatory, technical, competition, liquidity, etc.

## 6. Long-Term Potential
Adoption trends, ecosystem growth, 1–3 year outlook.

## 7. Verdict & Recommendation
Give a clear rating: **Strong Buy / Buy / Hold / Avoid / Sell**
Include a price target range, a 2–3 sentence investment thesis, and the potential return multiplier from current price (e.g. "At a target of $X, this represents a **2.5x** from the current price of $Y").
"""


def build_stock_prompt(ticker: str, d: dict) -> str:
    def fmt_num(val):
        if isinstance(val, (int, float)) and val != "N/A":
            if val > 1_000_000_000:
                return f"${val/1_000_000_000:.2f}B"
            elif val > 1_000_000:
                return f"${val/1_000_000:.2f}M"
            return str(val)
        return str(val)

    return f"""You are a senior professional stock market analyst. Analyze the following stock using the real-time financial data below. Be precise, honest, and data-driven.

STOCK: {ticker} — {d['name']}
Sector: {d['sector']} | Industry: {d['industry']} | Country: {d['country']}
Exchange: {d['exchange']} | Currency: {d['currency']}

PRICE & MARKET DATA:
- Current Price: {d['price']}
- Market Cap: {fmt_num(d['market_cap'])}
- 52-Week Range: {d['week52_low']} – {d['week52_high']}
- Analyst Mean Target: {d['analyst_target']}
- Beta: {d['beta']}
- Employees: {d['employees']}

VALUATION RATIOS:
- P/E (Trailing): {d['pe_ratio']}
- P/E (Forward): {d['forward_pe']}
- PEG Ratio: {d['peg_ratio']}
- P/S Ratio: {d['ps_ratio']}
- P/B Ratio: {d['pb_ratio']}
- EV/EBITDA: {d['ev_ebitda']}

FINANCIALS:
- Revenue Per Share (TTM): {d['revenue']}
- Revenue Growth (YoY): {d['revenue_growth']}
- Gross Margin: {d['gross_margins']}
- Operating Margin: {d['operating_margins']}
- Net Profit Margin: {d['profit_margins']}
- Free Cash Flow (Annual): {fmt_num(d['free_cashflow'])}
- Dividend Yield: {d['dividend_yield']}

FINANCIAL HEALTH:
- ROE: {d['roe']}
- ROA: {d['roa']}
- Debt/Equity: {d['debt_to_equity']}
- Current Ratio: {d['current_ratio']}

BUSINESS SUMMARY:
{d['summary']}

---
Provide a detailed analysis in the following markdown structure:

## 1. Business Model
How the company makes money, revenue streams, and customer base.

## 2. Fundamentals Analysis
Assess revenue growth, margins, cash flow quality, and balance sheet strength.

## 3. Valuation Analysis
Are current ratios cheap, fair, or expensive relative to peers and history?

## 4. Competitive Edge (Moat)
Key moats: brand, network effects, switching costs, IP, cost advantages, scale.

## 5. Management & Capital Allocation
Infer management quality from ROE, margins, FCF, and capital efficiency.

## 6. Key Risks
Top 3–5 risks the investor must understand.

## 7. Long-Term Potential
Industry tailwinds, 3–5 year growth outlook, secular trends.

## 8. Verdict & Recommendation
Give a clear rating: **Strong Buy / Buy / Hold / Avoid / Sell**
Include a suggested target price range, a 2–3 sentence investment thesis, and the potential return multiplier from current price (e.g. "At a target of $X, this represents a **2.5x** from the current price of $Y").
"""


@app.get("/market-overview")
async def market_overview():
    h = {"X-Finnhub-Token": FINNHUB_KEY}

    try:
        cfg_r = requests.get("https://api.alternative.me/fng/", timeout=10)
        cfg = cfg_r.json()["data"][0] if cfg_r.ok else {}
    except Exception:
        cfg = {}

    try:
        spy_quote   = requests.get(f"{FINNHUB}/quote",        params={"symbol": "SPY"}, headers=h, timeout=10).json()
        spy_metrics = requests.get(f"{FINNHUB}/stock/metric", params={"symbol": "SPY", "metric": "all"}, headers=h, timeout=10).json().get("metric", {})
        ndx_quote   = requests.get(f"{FINNHUB}/quote",        params={"symbol": "QQQ"}, headers=h, timeout=10).json()
        btc_quote   = requests.get(f"{FINNHUB}/quote",        params={"symbol": "BINANCE:BTCUSDT"}, headers=h, timeout=10).json()

        spy_price    = spy_quote.get("c", 0)
        spy_change   = round(spy_quote.get("dp", 0), 2)
        ndx_change   = round(ndx_quote.get("dp", 0), 2)
        btc_price    = round(btc_quote.get("c", 0), 2)
        btc_change   = round(btc_quote.get("dp", 0), 2)
        spy_52high   = spy_metrics.get("52WeekHigh", 0)
        spy_52low    = spy_metrics.get("52WeekLow", 0)

        if spy_52high and spy_52low and spy_price:
            pos = (spy_price - spy_52low) / (spy_52high - spy_52low) * 100
            if pos >= 65:
                stock_sentiment, stock_color = "Bull Market", "green"
            elif pos <= 35:
                stock_sentiment, stock_color = "Bear Market", "red"
            else:
                stock_sentiment, stock_color = "Neutral", "yellow"
        else:
            stock_sentiment, stock_color = "Unknown", "gray"

    except Exception:
        spy_price = spy_change = ndx_change = btc_price = btc_change = 0
        stock_sentiment, stock_color = "Unknown", "gray"

    fg_value = int(cfg.get("value", 0))
    fg_label = cfg.get("value_classification", "N/A")
    if fg_value >= 75:
        crypto_color = "green"
    elif fg_value >= 55:
        crypto_color = "lime"
    elif fg_value >= 45:
        crypto_color = "yellow"
    elif fg_value >= 25:
        crypto_color = "orange"
    else:
        crypto_color = "red"

    return {
        "stocks": {
            "spy_price": spy_price,
            "spy_change": spy_change,
            "ndx_change": ndx_change,
            "sentiment": stock_sentiment,
            "color": stock_color,
        },
        "crypto": {
            "btc_price": btc_price,
            "btc_change": btc_change,
            "fg_value": fg_value,
            "fg_label": fg_label,
            "color": crypto_color,
        }
    }


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    ticker = request.ticker.upper().strip()

    try:
        if is_crypto(ticker):
            data = fetch_crypto_data(ticker)
            prompt = build_crypto_prompt(ticker, data)
        else:
            data = fetch_stock_data(ticker)
            prompt = build_stock_prompt(ticker, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data: {str(e)}")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"temperature": 0}
        )
        analysis = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

    return {
        "ticker": ticker,
        "name": data["name"],
        "price": data["price"],
        "currency": data.get("currency", "USD"),
        "market_cap": data["market_cap"],
        "week52_high": data.get("week52_high", data.get("ath", "N/A")),
        "week52_low": data.get("week52_low", data.get("atl", "N/A")),
        "analyst_target": data.get("analyst_target", "N/A"),
        "sector": data.get("sector", data.get("categories", "Crypto")),
        "analysis": analysis,
        "asset_type": data["type"],
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
