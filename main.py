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

app = FastAPI()


class AnalyzeRequest(BaseModel):
    ticker: str


def fetch_stock_data(ticker: str) -> dict:
    h = {"X-Finnhub-Token": FINNHUB_KEY}

    profile  = requests.get(f"{FINNHUB}/stock/profile2", params={"symbol": ticker}, headers=h, timeout=10).json()
    quote    = requests.get(f"{FINNHUB}/quote",          params={"symbol": ticker}, headers=h, timeout=10).json()
    metrics  = requests.get(f"{FINNHUB}/stock/metric",   params={"symbol": ticker, "metric": "all"}, headers=h, timeout=10).json().get("metric", {})
    target   = requests.get(f"{FINNHUB}/stock/price-target", params={"symbol": ticker}, headers=h, timeout=10).json()

    if not profile.get("name"):
        raise ValueError(f"Ticker '{ticker}' not found.")

    def g(d, key, default="N/A"):
        val = d.get(key)
        if val is None or val == "" or val == 0:
            return default
        return round(val, 4) if isinstance(val, float) else val

    market_cap_raw = g(profile, "marketCapitalization")
    market_cap = int(market_cap_raw * 1_000_000) if market_cap_raw != "N/A" else "N/A"

    return {
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


def build_prompt(ticker: str, d: dict) -> str:
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
Include a suggested target price range and a 2–3 sentence investment thesis.
"""


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    ticker = request.ticker.upper().strip()

    try:
        data = fetch_stock_data(ticker)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data: {str(e)}")

    try:
        prompt = build_prompt(ticker, data)
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        analysis = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

    return {
        "ticker": ticker,
        "name": data["name"],
        "price": data["price"],
        "currency": data["currency"],
        "market_cap": data["market_cap"],
        "week52_high": data["week52_high"],
        "week52_low": data["week52_low"],
        "analyst_target": data["analyst_target"],
        "sector": data["sector"],
        "analysis": analysis,
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
