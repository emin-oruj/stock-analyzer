from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yfinance as yf
from google import genai
from dotenv import load_dotenv
import os
import curl_cffi

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()


class AnalyzeRequest(BaseModel):
    ticker: str


def fetch_stock_data(ticker: str) -> dict:
    session = curl_cffi.requests.Session(impersonate="chrome")
    stock = yf.Ticker(ticker, session=session)
    info = stock.info

    def fmt(val):
        if isinstance(val, float):
            return round(val, 4)
        return val

    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "country": info.get("country", "N/A"),
        "summary": (info.get("longBusinessSummary") or "N/A")[:600],
        "price": fmt(info.get("currentPrice", info.get("regularMarketPrice", "N/A"))),
        "market_cap": info.get("marketCap", "N/A"),
        "pe_ratio": fmt(info.get("trailingPE", "N/A")),
        "forward_pe": fmt(info.get("forwardPE", "N/A")),
        "peg_ratio": fmt(info.get("pegRatio", "N/A")),
        "ps_ratio": fmt(info.get("priceToSalesTrailing12Months", "N/A")),
        "pb_ratio": fmt(info.get("priceToBook", "N/A")),
        "ev_ebitda": fmt(info.get("enterpriseToEbitda", "N/A")),
        "revenue": info.get("totalRevenue", "N/A"),
        "revenue_growth": fmt(info.get("revenueGrowth", "N/A")),
        "gross_margins": fmt(info.get("grossMargins", "N/A")),
        "operating_margins": fmt(info.get("operatingMargins", "N/A")),
        "profit_margins": fmt(info.get("profitMargins", "N/A")),
        "roe": fmt(info.get("returnOnEquity", "N/A")),
        "roa": fmt(info.get("returnOnAssets", "N/A")),
        "debt_to_equity": fmt(info.get("debtToEquity", "N/A")),
        "current_ratio": fmt(info.get("currentRatio", "N/A")),
        "free_cashflow": info.get("freeCashflow", "N/A"),
        "dividend_yield": fmt(info.get("dividendYield", "N/A")),
        "week52_high": fmt(info.get("fiftyTwoWeekHigh", "N/A")),
        "week52_low": fmt(info.get("fiftyTwoWeekLow", "N/A")),
        "analyst_target": fmt(info.get("targetMeanPrice", "N/A")),
        "beta": fmt(info.get("beta", "N/A")),
        "employees": info.get("fullTimeEmployees", "N/A"),
        "exchange": info.get("exchange", "N/A"),
        "currency": info.get("currency", "USD"),
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
- Revenue: {fmt_num(d['revenue'])}
- Revenue Growth (YoY): {d['revenue_growth']}
- Gross Margin: {d['gross_margins']}
- Operating Margin: {d['operating_margins']}
- Net Profit Margin: {d['profit_margins']}
- Free Cash Flow: {fmt_num(d['free_cashflow'])}
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
        if data["price"] == "N/A" and data["market_cap"] == "N/A":
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found.")
    except HTTPException:
        raise
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
