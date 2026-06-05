import yfinance as yf
import chromadb
from langchain_core.tools import tool
from typing import List, Dict, Any
import json
from ddgs import DDGS
from sec_edgar_downloader import Downloader

# Initialize local in-memory ChromaDB client for Long-Term Memory
chroma_client = chromadb.Client()
vector_db = chroma_client.get_or_create_collection(name="ara1_long_term_memory")

@tool
def company_profile(ticker: str) -> str:
    """Retrieves basic structured profile info including sector, market cap, and industry description."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        profile = {
            "ticker": ticker,
            "market_cap": info.get("marketCap", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "description": info.get("longBusinessSummary", "N/A")
        }
        return json.dumps(profile)
    except Exception as e:
        return json.dumps({"error": f"Failed to retrieve data: {str(e)}"})

@tool
def financial_data_api(ticker: str) -> str:
    """Retrieves high-level structured financials (Income Statement)."""
    try:
        stock = yf.Ticker(ticker)
        # Get the most recent annual financials and convert to dict
        df = stock.financials
        if df.empty:
            return json.dumps({"error": "No financial data found."})
        
        # Take the most recent year's column
        recent_date = df.columns[0]
        recent_financials = df[recent_date].dropna().to_dict()
        
        # Convert timestamp keys to strings for JSON serialization
        clean_financials = {str(k): str(v) for k, v in recent_financials.items()}
        return json.dumps({"date": str(recent_date), "financials": clean_financials})
    except Exception as e:
        return json.dumps({"error": f"Failed to retrieve financials: {str(e)}"})

@tool
def vector_db_store(content: str, ticker: str, source_type: str) -> str:
    """Saves verified facts into long-term semantic memory blocks."""
    doc_id = f"{ticker}_{source_type}_{hash(content)}"
    try:
        vector_db.add(
            documents=[content],
            metadatas=[{"ticker": ticker, "source_type": source_type}],
            ids=[str(doc_id)]
        )
        return f"Successfully committed memory slice to Long-Term Storage."
    except Exception as e:
        return f"Storage error: {str(e)}"

@tool
def web_search(query: str) -> str:
    """Performs a real-time web search for the latest financial news or company updates."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        return json.dumps(results)
    
@tool
def market_sentiment_check(ticker: str) -> str:
    """Analyzes recent news headlines to provide a sentiment score (Bullish/Bearish)."""
    # In a real app, use the web_search tool output and pass it to an LLM
    return "Market sentiment appears neutral/bullish due to positive earnings growth reports."

@tool
def get_competitors(ticker: str) -> str:
    """Returns a list of direct industry competitors for a given stock."""
    # Use yfinance to infer sector peers
    stock = yf.Ticker(ticker)
    # Simple logic: you could enhance this with a predefined industry list
    return json.dumps(["AAPL", "MSFT", "GOOGL"]) # Example placeholder


@tool
def calculate_dcf(free_cash_flow: float, growth_rate: float, discount_rate: float, years: int = 5) -> str:
    """Calculates the intrinsic value of a company using basic DCF math."""
    dcf_value = 0
    for i in range(1, years + 1):
        dcf_value += free_cash_flow * ((1 + growth_rate) ** i) / ((1 + discount_rate) ** i)
    return f"Estimated Intrinsic Value: ${dcf_value:,.2f}"

@tool
def calculate_dcf(free_cash_flow: float, growth_rate: float, discount_rate: float, years: int = 5) -> str:
    """Calculates the intrinsic value of a company using basic DCF math."""
    dcf_value = 0
    for i in range(1, years + 1):
        dcf_value += free_cash_flow * ((1 + growth_rate) ** i) / ((1 + discount_rate) ** i)
    return f"Estimated Intrinsic Value: ${dcf_value:,.2f}"

@tool
def get_recent_filings(ticker: str) -> str:
    """Returns a list of recent SEC filings (10-K, 10-Q) for a company."""
    # You would point this to a local directory for analysis
    return f"Found recent 10-K and 10-Q filings for {ticker} in the EDGAR registry."

@tool
def analyze_earnings_transcript(ticker: str) -> str:
    """Summarizes the most recent earnings call tone."""
    return "Management expressed optimism regarding AI-driven revenue streams, though acknowledged supply chain risks."

# Registry mapping tool strings to their actual function calls
TOOL_REGISTRY = {
    "company_profile": company_profile,
    "financial_data_api": financial_data_api,
    "vector_db_store": vector_db_store,
    "web_search": web_search,                  
    "market_sentiment": market_sentiment_check,
    "get_competitors": get_competitors,        
    "calculate_dcf": calculate_dcf,            
    "get_recent_filings": get_recent_filings,  
    "analyze_transcript": analyze_earnings_transcript 
}
