import os
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import uuid
import aiosqlite
import json
from typing import Literal, Dict, Any, Optional, List, Tuple
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, message_to_dict, messages_from_dict
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from typing import Annotated, List
from typing_extensions import TypedDict
from cachetools import TTLCache
from datetime import datetime, timedelta
from sqlalchemy import text, inspect, MetaData, Table, Column
import math
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import gc
import re
import traceback
from collections import defaultdict, Counter
import pytz

# Date/Time Utilities
class DateTimeProvider:
    """Provides current date, time, and timezone information"""
    
    def __init__(self, timezone: str = "UTC"):
        self.timezone = pytz.timezone(timezone)
    
    def get_current_datetime(self) -> datetime:
        """Get the current datetime in the specified timezone"""
        return datetime.now(self.timezone)
    
    def get_current_location(self) -> str:
        """Returns the current timezone as a proxy for location"""
        return str(self.timezone)

# Guardrails for non-stock market queries
class StockMarketQueryClassifier:
    """Classifies queries to ensure they are related to the stock market"""
    
    def __init__(self):
        self.stock_market_keywords = {
            "stock", "share", "market", "equity", "nse", "bse", "sensex", "nifty",
            "trading", "investment", "portfolio", "dividend", "ipo", "futures", "options",
            "candlestick", "volume", "price", "open", "close", "high", "low"
        }

    def is_stock_market_related(self, query: str) -> bool:
        """Check if a query is related to the stock market"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.stock_market_keywords)

# Configure logging for testing and debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enhanced caches with better management
sql_cache = TTLCache(maxsize=2000, ttl=3600)
llm_cache = TTLCache(maxsize=2000, ttl=3600)
financial_cache = TTLCache(maxsize=1000, ttl=1800)
schema_cache = TTLCache(maxsize=500, ttl=7200)  # Cache schema for 2 hours

# Load environment variables
load_dotenv()

# Set up Google API Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize LLM with optimized parameters
model = "gemini-2.0-flash"
llm = ChatGoogleGenerativeAI(model=model, max_tokens=400)
flash_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", max_tokens=200)

# Optimized database initialization
try:
    db = SQLDatabase.from_uri("sqlite:///stock.db")
    logger.info("✅ Database initialized")
except Exception as e:
    logger.error(f"❌ Database connection error: {e}")
    db = None

# Initialize SQLDatabaseToolkit
if db:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
else:
    tools = []

run_query_tool = None
if tools:
    for tool in tools:
        if tool.name == "sql_db_query":
            run_query_tool = tool
            break

# Enhanced Financial Calculation Tools
@dataclass
class FinancialMetrics:
    """Enhanced financial metrics calculator with comprehensive functionality"""
    
    @staticmethod
    def calculate_returns(prices: List[float], method: str = "log") -> List[float]:
        """Calculate returns using various methods"""
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            if method == "log":
                returns.append(math.log(prices[i] / prices[i-1]))
            elif method == "simple":
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
            elif method == "percentage":
                returns.append(((prices[i] - prices[i-1]) / prices[i-1]) * 100)
        return returns
    
    @staticmethod
    def calculate_volatility(returns: List[float], annualize: bool = True) -> float:
        """Calculate volatility (standard deviation) of returns"""
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum([(r - mean_return) ** 2 for r in returns]) / len(returns)
        volatility = math.sqrt(variance)
        
        return volatility * math.sqrt(252) if annualize else volatility
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02, annualize: bool = True) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        volatility = FinancialMetrics.calculate_volatility(returns, annualize)
        
        excess_return = mean_return - (risk_free_rate / 252 if not annualize else risk_free_rate)
        return excess_return / volatility if volatility != 0 else 0.0

    @staticmethod
    def calculate_beta(stock_returns: List[float], market_returns: List[float]) -> float:
        """Calculate beta relative to market"""
        if len(stock_returns) != len(market_returns) or len(stock_returns) < 2:
            return 0.0
        
        # Calculate covariance and market variance
        stock_mean = sum(stock_returns) / len(stock_returns)
        market_mean = sum(market_returns) / len(market_returns)
        
        covariance = sum([(s - stock_mean) * (m - market_mean) for s, m in zip(stock_returns, market_returns)]) / len(stock_returns)
        market_variance = sum([(m - market_mean) ** 2 for m in market_returns]) / len(market_returns)
        
        return covariance / market_variance if market_variance != 0 else 0.0

# Enhanced Schema Inspector
class DynamicSchemaInspector:
    """Dynamic database schema inspector for intelligent SQL generation"""
    
    def __init__(self, db: SQLDatabase):
        self.db = db
        self.schema_info = {}
        self._inspect_schema()
    
    def _inspect_schema(self):
        """Inspect and cache database schema"""
        try:
            with self.db._engine.connect() as conn:
                inspector = inspect(conn)
                
                # Get all tables
                tables = inspector.get_table_names()
                self.schema_info["tables"] = {}
                
                for table_name in tables:
                    columns = inspector.get_columns(table_name)
                    foreign_keys = inspector.get_foreign_keys(table_name)
                    indexes = inspector.get_indexes(table_name)
                    
                    self.schema_info["tables"][table_name] = {
                        "columns": [
                            {
                                "name": col["name"],
                                "type": str(col["type"]),
                                "nullable": col.get("nullable", True),
                                "primary_key": col.get("primary_key", False),
                                "default": str(col.get("default", "")) if col.get("default") else None
                            }
                            for col in columns
                        ],
                        "foreign_keys": foreign_keys,
                        "indexes": indexes
                    }
                
                # Get table relationships
                self.schema_info["relationships"] = self._analyze_relationships()
                
                logger.info(f"✅ Schema inspection completed - {len(tables)} tables analyzed")
                
        except Exception as e:
            logger.error(f"Schema inspection error: {e}")
            self.schema_info = {"tables": {}, "relationships": {}}
    
    def _analyze_relationships(self) -> Dict[str, List[Dict]]:
        """Analyze potential relationships between tables"""
        relationships = defaultdict(list)
        
        # Explicitly define the relationship between stock_index_price_daily and stock_company_price_daily
        relationships["stock_company_price_daily"].append({
            "type": "foreign_key",
            "column": "index_name",
            "references_table": "stock_index_price_daily",
            "references_column": "index_name",
            "description": "Links company stock prices to the corresponding index"
        })
        relationships["stock_company_price_daily"].append({
            "type": "foreign_key",
            "column": "CH_TIMESTAMP",
            "references_table": "stock_index_price_daily",
            "references_column": "date_key",
            "description": "Links company stock prices to the corresponding date in the index price table"
        })
        
        for table_name, table_info in self.schema_info["tables"].items():
            for column in table_info["columns"]:
                col_name = column["name"]
                # Look for common relationship patterns
                if "index" in col_name.lower():
                    relationships[table_name].append({
                        "type": "index_relationship",
                        "column": col_name,
                        "description": f"Links to index data via {col_name}"
                    })
                
                if "symbol" in col_name.lower():
                    relationships[table_name].append({
                        "type": "symbol_relationship", 
                        "column": col_name,
                        "description": f"Stock symbol identifier in {col_name}"
                    })
        
        return dict(relationships)
    
    def get_table_info(self, table_name: str) -> Optional[Dict]:
        """Get information about a specific table"""
        return self.schema_info["tables"].get(table_name)
    
    def get_column_info(self, table_name: str, column_name: str) -> Optional[Dict]:
        """Get information about a specific column"""
        table_info = self.get_table_info(table_name)
        if not table_info:
            return None
        
        for column in table_info["columns"]:
            if column["name"] == column_name:
                return column
        return None
    
    def find_relevant_tables(self, query_keywords: List[str]) -> List[str]:
        """Find tables relevant to query keywords"""
        relevant_tables = []
        
        for table_name in self.schema_info["tables"]:
            # Score table based on keyword matches
            score = 0
            table_lower = table_name.lower()
            
            for keyword in query_keywords:
                if keyword in table_lower:
                    score += 2
                
                # Check column names
                for column in self.schema_info["tables"][table_name]["columns"]:
                    if keyword in column["name"].lower():
                        score += 1
            
            if score > 0:
                relevant_tables.append((table_name, score))
        
        # Sort by relevance score
        relevant_tables.sort(key=lambda x: x[1], reverse=True)
        return [table for table, score in relevant_tables[:3]]  # Top 3 most relevant
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of data available in the database"""
        try:
            summary = {}
            with self.db._engine.connect() as conn:
                for table_name in self.schema_info["tables"]:
                    try:
                        # Get row count
                        count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        row_count = count_result.scalar()
                        
                        # Get sample columns and their types
                        sample_result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 1"))
                        columns = list(sample_result.keys())
                        
                        summary[table_name] = {
                            "row_count": row_count,
                            "columns": columns,
                            "has_data": row_count > 0
                        }
                    except Exception as e:
                        logger.warning(f"Could not analyze table {table_name}: {e}")
                        summary[table_name] = {"error": str(e)}
            
            return summary
        except Exception as e:
            logger.error(f"Data summary error: {e}")
            return {}

# Query Feasibility Validator
class QueryFeasibilityValidator:
    """Validates whether queries can be fulfilled with available data"""
    
    def __init__(self, schema_inspector: DynamicSchemaInspector):
        self.schema_inspector = schema_inspector
        self.data_summary = schema_inspector.get_data_summary()
    
    def validate_query_feasibility(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if a query can be fulfilled with available data"""
        validation_result = {
            "feasible": True,
            "confidence": 1.0,
            "issues": [],
            "alternatives": [],
            "required_tables": [],
            "missing_data": []
        }
        
        user_query = query_analysis.get("user_query", "").lower()
        query_type = query_analysis.get("analysis_type", "")
        
        # Check sector analysis feasibility
        if query_type in ["sector_analysis", "sector_performance"]:
            if not self._check_sector_data_availability():
                validation_result["feasible"] = False
                validation_result["confidence"] = 0.3
                validation_result["issues"].append("Limited sector classification data available")
                validation_result["alternatives"].append("Analyze by index names instead of sectors")
        
        # Check time series analysis feasibility
        if "time" in user_query or "period" in user_query:
            if not self._check_temporal_data():
                validation_result["feasible"] = False
                validation_result["confidence"] = 0.2
                validation_result["issues"].append("Insufficient time-series data")
                validation_result["alternatives"].append("Provide snapshot analysis instead")
        
        # Check volume analysis feasibility
        if "volume" in user_query:
            volume_tables = self._check_volume_data_availability()
            if not volume_tables:
                validation_result["feasible"] = False
                validation_result["confidence"] = 0.1
                validation_result["issues"].append("No volume data available")
                validation_result["alternatives"].append("Focus on price-only analysis")
        
        # Check performance calculation feasibility
        if "performance" in user_query or "return" in user_query:
            if not self._check_performance_data():
                validation_result["feasible"] = False
                validation_result["confidence"] = 0.5
                validation_result["issues"].append("Limited historical price data")
                validation_result["alternatives"].append("Use available price snapshots")
        
        # Check statistical analysis feasibility
        if any(word in user_query for word in ["standard deviation", "variance", "volatility", "stdev"]):
            if not self._check_statistical_data():
                validation_result["feasible"] = False
                validation_result["confidence"] = 0.6
                validation_result["issues"].append("Limited data points for statistical calculations")
                validation_result["alternatives"].append("Use available data points for basic statistics")
        
        return validation_result
    
    def _check_sector_data_availability(self) -> bool:
        """Check if sector analysis is possible"""
        company_table = self.schema_inspector.get_table_info("stock_company_price_daily")
        if not company_table:
            return False
        
        # Check if index_name column exists and has data
        return self._table_has_data_with_column("stock_company_price_daily", "index_name")
    
    def _check_temporal_data(self) -> bool:
        """Check if time series analysis is possible"""
        return self._table_has_minimum_rows("stock_company_price_daily", 100)
    
    def _check_volume_data_availability(self) -> List[str]:
        """Check for volume data availability"""
        volume_columns = ["CH_TOT_TRADED_QTY", "CH_TOT_TRADED_VAL"]
        available_tables = []
        
        for table_name, table_info in self.schema_inspector.schema_info["tables"].items():
            column_names = [col["name"] for col in table_info["columns"]]
            if any(col in column_names for col in volume_columns):
                if self._table_has_data(table_name):
                    available_tables.append(table_name)
        
        return available_tables
    
    def _check_performance_data(self) -> bool:
        """Check if performance calculations are possible"""
        return (self._table_has_column("stock_company_price_daily", "CH_CLOSING_PRICE") and
                self._table_has_minimum_rows("stock_company_price_daily", 10))
    
    def _check_statistical_data(self) -> bool:
        """Check if statistical calculations are possible"""
        return (self._table_has_column("stock_company_price_daily", "CH_CLOSING_PRICE") and
                self._table_has_minimum_rows("stock_company_price_daily", 20))
    
    def _table_has_data(self, table_name: str) -> bool:
        """Check if table has any data"""
        return (table_name in self.data_summary and 
                self.data_summary[table_name].get("has_data", False))
    
    def _table_has_column(self, table_name: str, column_name: str) -> bool:
        """Check if table has specific column"""
        table_info = self.schema_inspector.get_table_info(table_name)
        if not table_info:
            return False
        
        return any(col["name"] == column_name for col in table_info["columns"])
    
    def _table_has_data_with_column(self, table_name: str, column_name: str) -> bool:
        """Check if table has data and specific column"""
        if not self._table_has_data(table_name) or not self._table_has_column(table_name, column_name):
            return False
        
        try:
            with self.schema_inspector.db._engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT COUNT(DISTINCT {column_name}) FROM {table_name} WHERE {column_name} IS NOT NULL")
                )
                distinct_count = result.scalar()
                return distinct_count > 5  # Need at least 5 different values
        except Exception:
            return False
    
    def _table_has_minimum_rows(self, table_name: str, min_rows: int) -> bool:
        """Check if table has minimum number of rows"""
        return (table_name in self.data_summary and 
                self.data_summary[table_name].get("row_count", 0) >= min_rows)

# Enhanced Query Analysis
class IntelligentQueryAnalyzer:
    """Enhanced query analyzer with semantic understanding"""
    
    def __init__(self):
        self.financial_keywords = {
            "performance": ["best", "worst", "top", "highest", "lowest", "gain", "loss", "return", "appreciation"],
            "volume": ["volume", "trading", "quantity", "liquidity"],
            "price": ["price", "cost", "value", "trading"],
            "sector": ["sector", "industry", "segment", "category", "type"],
            "time": ["day", "week", "month", "year", "period", "historical", "trend"],
            "comparison": ["compare", "vs", "versus", "difference", "relative"],
            "ranking": ["ranking", "rank", "order", "sort", "position"],
            "statistical": ["standard deviation", "variance", "volatility", "stdev", "std", "mean", "average", "median", "mode", "correlation", "covariance"],
            "aggregation": ["group by", "aggregate", "sum", "count", "average", "avg", "min", "max"]
        }
        
        self.sector_mappings = {
            "it": ["information technology", "technology", "tech", "software", "hardware"],
            "auto": ["automobile", "automotive", "cars", "vehicles"],
            "banking": ["bank", "financial services", "finance", "banking"],
            "fmcg": ["fast moving consumer goods", "consumer goods", "fmcg"],
            "metal": ["metals", "mining", "steel", "aluminum"],
            "pharma": ["pharmaceutical", "pharma", "drug", "medicine"]
        }
    
    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """Comprehensive query analysis"""
        query_lower = user_query.lower()
        tokens = re.findall(r'\b\w+\b', query_lower)
        
        analysis = {
            "user_query": user_query,
            "query_type": self._determine_query_type(query_lower),
            "sector_intent": self._detect_sector_intent(tokens),
            "financial_metrics": self._identify_financial_metrics(query_lower),
            "time_period": self._extract_time_period(query_lower),
            "aggregation_level": self._determine_aggregation_level(query_lower),
            "complexity": self._assess_complexity(query_lower),
            "data_requirements": self._identify_data_requirements(tokens),
            "analysis_depth": self._determine_analysis_depth(query_lower),
            "requires_aggregation": self._requires_aggregation(query_lower),
            "statistical_metrics": self._identify_statistical_metrics(query_lower)
        }
        
        return analysis
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the primary type of query"""
        if any(word in query for word in ["sector", "industry"]):
            return "sector_analysis"
        elif any(word in query for word in ["volume", "trading"]):
            return "volume_analysis"
        elif any(word in query for word in ["performance", "return", "gain"]):
            return "performance_analysis"
        elif any(word in query for word in ["compare", "vs", "versus"]):
            return "comparative_analysis"
        elif any(word in query for word in ["standard deviation", "variance", "volatility", "stdev"]):
            return "statistical_analysis"
        elif any(word in query for word in ["average", "mean", "sum", "group by"]):
            return "aggregation_analysis"
        else:
            return "general_analysis"
    
    def _detect_sector_intent(self, tokens: List[str]) -> Optional[str]:
        """Detect sector-specific intent"""
        for sector, keywords in self.sector_mappings.items():
            if any(keyword in tokens for keyword in keywords):
                return sector
        return None
    
    def _identify_financial_metrics(self, query: str) -> List[str]:
        """Identify required financial metrics"""
        metrics = []
        for metric_type, keywords in self.financial_keywords.items():
            if any(keyword in query for keyword in keywords):
                metrics.append(metric_type)
        return metrics
    
    def _identify_statistical_metrics(self, query: str) -> List[str]:
        """Identify required statistical metrics"""
        statistical_keywords = {
            "standard_deviation": ["standard deviation", "stdev", "std"],
            "variance": ["variance", "var"],
            "volatility": ["volatility", "vol"],
            "mean": ["mean", "average", "avg"],
            "median": ["median"],
            "mode": ["mode"],
            "correlation": ["correlation", "corr"],
            "covariance": ["covariance", "cov"]
        }
        
        metrics = []
        for metric_type, keywords in statistical_keywords.items():
            if any(keyword in query for keyword in keywords):
                metrics.append(metric_type)
        
        return metrics
    
    def _requires_aggregation(self, query: str) -> bool:
        """Check if query requires aggregation/group by"""
        aggregation_keywords = ["each", "per", "by index", "by sector", "group by", "aggregate", "summary"]
        return any(keyword in query for keyword in aggregation_keywords)
    
    def _extract_time_period(self, query: str) -> Optional[str]:
        """Extract time period from query"""
        time_patterns = [
            r'(\d+)\s*(day|days|week|weeks|month|months|year|years)',
            r'(last|past)\s*(day|days|week|weeks|month|months|year|years)',
            r'(recent|current)\s*(day|days|week|weeks|month|months|year|years)'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(0)
        return None
    
    def _determine_aggregation_level(self, query: str) -> str:
        """Determine if analysis should be per-stock, per-sector, or global"""
        if any(word in query for word in ["stock", "symbol", "company"]):
            return "stock_level"
        elif any(word in query for word in ["sector", "industry", "index"]):
            return "sector_level"
        else:
            return "portfolio_level"
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        complexity_indicators = ["compare", "correlation", "beta", "sharpe", "volatility", "standard deviation"]
        if any(indicator in query for indicator in complexity_indicators):
            return "complex"
        elif len(query.split()) > 10:
            return "moderate"
        else:
            return "simple"
    
    def _identify_data_requirements(self, tokens: List[str]) -> List[str]:
        """Identify what data columns are needed"""
        requirements = []
        
        for token in tokens:
            if token in ["price", "closing", "open", "high", "low"]:
                requirements.append("price_data")
            elif token in ["volume", "trading", "quantity"]:
                requirements.append("volume_data")
            elif token in ["index", "sector", "industry"]:
                requirements.append("classification_data")
            elif token in ["date", "time", "period"]:
                requirements.append("temporal_data")
            elif token in ["standard deviation", "variance", "volatility"]:
                requirements.append("statistical_data")
        
        return list(set(requirements))
    
    def _determine_analysis_depth(self, query: str) -> str:
        """Determine depth of analysis required"""
        if any(word in query for word in ["detailed", "comprehensive", "thorough"]):
            return "deep"
        elif any(word in query for word in ["simple", "basic", "quick"]):
            return "surface"
        else:
            return "standard"

# Dynamic SQL Generator with Enhanced Aggregation Support
class DynamicSQLGenerator:
    """Generates dynamic SQL based on query analysis and schema"""
    
    def __init__(self, schema_inspector: DynamicSchemaInspector, query_analyzer: IntelligentQueryAnalyzer):
        self.schema_inspector = schema_inspector
        self.query_analyzer = query_analyzer
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, Dict]:
        """Initialize SQL templates with placeholders"""
        return {
            "sector_performance": {
                "description": "Sector-based stock performance analysis",
                "template": """
                WITH sector_data AS (
                    SELECT 
                        CH_SYMBOL,
                        index_name,
                        CH_CLOSING_PRICE,
                        CH_TOT_TRADED_QTY,
                        CH_TOT_TRADED_VAL,
                        TIMESTAMP,
                        ROW_NUMBER() OVER (PARTITION BY CH_SYMBOL ORDER BY TIMESTAMP) as row_num
                    FROM stock_company_price_daily
                    WHERE index_name IS NOT NULL
                    {date_filter}
                ),
                current_prices AS (
                    SELECT * FROM sector_data WHERE row_num = 1
                ),
                previous_prices AS (
                    SELECT * FROM sector_data WHERE row_num = 20
                )
                SELECT 
                    c.CH_SYMBOL,
                    c.index_name,
                    c.CH_CLOSING_PRICE as current_price,
                    ROUND(((c.CH_CLOSING_PRICE - COALESCE(p.CH_CLOSING_PRICE, c.CH_CLOSING_PRICE)) / COALESCE(p.CH_CLOSING_PRICE, 1)) * 100, 2) as price_change_pct,
                    c.CH_TOT_TRADED_QTY,
                    ROUND(c.CH_TOT_TRADED_VAL, 0) as total_traded_value
                FROM current_prices c
                LEFT JOIN previous_prices p ON c.CH_SYMBOL = p.CH_SYMBOL
                {sector_filter}
                ORDER BY price_change_pct DESC
                LIMIT {limit}
                """,
                "placeholders": ["date_filter", "sector_filter", "limit"]
            },
            
            "volume_analysis": {
                "description": "Volume-based analysis",
                "template": """
                SELECT 
                    CH_SYMBOL,
                    index_name,
                    CH_TOT_TRADED_QTY,
                    CH_TOT_TRADED_VAL,
                    CH_CLOSING_PRICE,
                    TIMESTAMP,
                    ROUND(CH_TOT_TRADED_VAL / NULLIF(CH_TOT_TRADED_QTY, 0), 2) as avg_price
                FROM stock_company_price_daily
                WHERE CH_TOT_TRADED_QTY > 0
                {date_filter}
                {sector_filter}
                ORDER BY CH_TOT_TRADED_QTY DESC
                LIMIT {limit}
                """,
                "placeholders": ["date_filter", "sector_filter", "limit"]
            },
            
            "performance_ranking": {
                "description": "Stock performance ranking",
                "template": """
                WITH price_changes AS (
                    SELECT 
                        CH_SYMBOL,
                        index_name,
                        CH_CLOSING_PRICE,
                        CH_TOT_TRADED_QTY,
                        CH_TOT_TRADED_VAL,
                        TIMESTAMP,
                        LAG(CH_CLOSING_PRICE, 1) OVER (PARTITION BY CH_SYMBOL ORDER BY TIMESTAMP) as prev_price,
                        LAG(CH_CLOSING_PRICE, 5) OVER (PARTITION BY CH_SYMBOL ORDER BY TIMESTAMP) as week_ago_price,
                        LAG(CH_CLOSING_PRICE, 20) OVER (PARTITION BY CH_SYMBOL ORDER BY TIMESTAMP) as month_ago_price
                    FROM stock_company_price_daily
                    WHERE 1=1
                    {date_filter}
                )
                SELECT 
                    CH_SYMBOL,
                    index_name,
                    CH_CLOSING_PRICE,
                    ROUND(((CH_CLOSING_PRICE - week_ago_price) / week_ago_price) * 100, 2) as weekly_return_pct,
                    ROUND(((CH_CLOSING_PRICE - month_ago_price) / month_ago_price) * 100, 2) as monthly_return_pct,
                    CH_TOT_TRADED_QTY,
                    TIMESTAMP
                FROM price_changes
                WHERE week_ago_price IS NOT NULL AND month_ago_price IS NOT NULL
                {sector_filter}
                ORDER BY monthly_return_pct DESC
                LIMIT {limit}
                """,
                "placeholders": ["date_filter", "sector_filter", "limit"]
            },
            
            "comparative_analysis": {
                "description": "Comparative analysis between sectors or stocks",
                "template": """
                SELECT 
                    index_name,
                    COUNT(DISTINCT CH_SYMBOL) as stock_count,
                    ROUND(AVG(CH_CLOSING_PRICE), 2) as avg_price,
                    ROUND(SUM(CH_TOT_TRADED_QTY), 0) as total_volume,
                    ROUND(SUM(CH_TOT_TRADED_VAL), 0) as total_value,
                    ROUND(AVG(CH_CLOSING_PRICE) * COUNT(DISTINCT CH_SYMBOL), 2) as sector_market_cap
                FROM stock_company_price_daily
                WHERE 1=1
                {date_filter}
                {sector_filter}
                GROUP BY index_name
                HAVING COUNT(DISTINCT CH_SYMBOL) > 1
                ORDER BY total_volume DESC
                """,
                "placeholders": ["date_filter", "sector_filter"]
            },
            
            "statistical_analysis": {
                "description": "Statistical analysis with aggregation by index/sector",
                "template": """
                WITH price_returns AS (
                    SELECT 
                        CH_SYMBOL,
                        index_name,
                        CH_CLOSING_PRICE,
                        LAG(CH_CLOSING_PRICE) OVER (PARTITION BY CH_SYMBOL ORDER BY TIMESTAMP) as prev_price,
                        TIMESTAMP
                    FROM stock_company_price_daily
                    WHERE 1=1
                    {date_filter}
                    {sector_filter}
                ),
                returns_data AS (
                    SELECT 
                        CH_SYMBOL,
                        index_name,
                        ROUND(((CH_CLOSING_PRICE - prev_price) / prev_price) * 100, 4) as daily_return_pct,
                        CH_CLOSING_PRICE
                    FROM price_returns
                    WHERE prev_price IS NOT NULL AND prev_price > 0
                ),
                index_statistics AS (
                    SELECT 
                        index_name,
                        COUNT(DISTINCT CH_SYMBOL) as stock_count,
                        ROUND(AVG(daily_return_pct), 4) as avg_daily_return,
                        ROUND(SQRT(AVG(daily_return_pct * daily_return_pct) - AVG(daily_return_pct) * AVG(daily_return_pct)), 4) as return_stdev,
                        ROUND(AVG(CH_CLOSING_PRICE), 2) as avg_price,
                        ROUND(MIN(CH_CLOSING_PRICE), 2) as min_price,
                        ROUND(MAX(CH_CLOSING_PRICE), 2) as max_price,
                        ROUND(MAX(CH_CLOSING_PRICE) - MIN(CH_CLOSING_PRICE), 2) as price_range
                    FROM returns_data
                    WHERE daily_return_pct IS NOT NULL
                    GROUP BY index_name
                    HAVING COUNT(DISTINCT CH_SYMBOL) > 1
                )
                SELECT 
                    index_name,
                    stock_count,
                    avg_daily_return,
                    return_stdev as standard_deviation_of_returns,
                    avg_price,
                    min_price,
                    max_price,
                    price_range,
                    ROUND(return_stdev * SQRT(252), 4) as annualized_volatility
                FROM index_statistics
                ORDER BY return_stdev DESC
                """,
                "placeholders": ["date_filter", "sector_filter"]
            },
            
            "aggregation_analysis": {
                "description": "General aggregation analysis by index/sector",
                "template": """
                SELECT 
                    index_name,
                    COUNT(DISTINCT CH_SYMBOL) as stock_count,
                    ROUND(AVG(CH_CLOSING_PRICE), 2) as avg_closing_price,
                    ROUND(MIN(CH_CLOSING_PRICE), 2) as min_price,
                    ROUND(MAX(CH_CLOSING_PRICE), 2) as max_price,
                    ROUND(SUM(CH_TOT_TRADED_QTY), 0) as total_volume,
                    ROUND(SUM(CH_TOT_TRADED_VAL), 0) as total_value,
                    ROUND(AVG(CH_TOT_TRADED_QTY), 0) as avg_volume,
                    COUNT(*) as total_records,
                    ROUND(AVG(CH_TOT_TRADED_VAL / NULLIF(CH_TOT_TRADED_QTY, 0)), 2) as avg_trading_price
                FROM stock_company_price_daily
                WHERE 1=1
                {date_filter}
                {sector_filter}
                AND CH_CLOSING_PRICE > 0
                GROUP BY index_name
                HAVING COUNT(DISTINCT CH_SYMBOL) > 1
                ORDER BY avg_closing_price DESC
                """,
                "placeholders": ["date_filter", "sector_filter"]
            }
        }
    
    def generate_sql(self, query_analysis: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dynamic SQL based on query analysis"""
        if not validation_result["feasible"]:
            return self._generate_alternative_sql(query_analysis, validation_result)
        
        query_type = query_analysis["query_type"]
        templates_map = {
            "sector_analysis": "sector_performance",
            "volume_analysis": "volume_analysis", 
            "performance_analysis": "performance_ranking",
            "comparative_analysis": "comparative_analysis",
            "statistical_analysis": "statistical_analysis",
            "aggregation_analysis": "aggregation_analysis"
        }
        
        # Check if query requires aggregation and update type accordingly
        if query_analysis["requires_aggregation"] and query_type not in ["statistical_analysis", "aggregation_analysis"]:
            if any(word in query_analysis["user_query"] for word in ["standard deviation", "variance", "volatility"]):
                query_type = "statistical_analysis"
            else:
                query_type = "aggregation_analysis"
        
        template_key = templates_map.get(query_type, "sector_performance")
        
        if template_key not in self.templates:
            return self._generate_fallback_sql(query_analysis)
        
        template = self.templates[template_key]
        
        # Build SQL with appropriate placeholders
        sql_params = self._build_sql_parameters(query_analysis, validation_result)
        sql = self._fill_template(template["template"], sql_params)
        
        # Add safety checks and optimization
        sql = self._optimize_sql(sql)
        
        return {
            "sql": sql,
            "template_used": template_key,
            "parameters": sql_params,
            "confidence": validation_result.get("confidence", 0.8)
        }
    
    def _build_sql_parameters(self, query_analysis: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, str]:
        """Build parameters for SQL template"""
        params = {}
        
        # Date filter
        time_period = query_analysis.get("time_period")
        if time_period:
            if "day" in time_period:
                days = int(re.search(r'\d+', time_period).group()) if re.search(r'\d+', time_period) else 7
                start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            elif "week" in time_period:
                weeks = int(re.search(r'\d+', time_period).group()) if re.search(r'\d+', time_period) else 4
                start_date = (datetime.now() - timedelta(weeks=weeks)).strftime("%Y-%m-%d")
            elif "month" in time_period:
                months = int(re.search(r'\d+', time_period).group()) if re.search(r'\d+', time_period) else 1
                start_date = (datetime.now() - timedelta(days=months*30)).strftime("%Y-%m-%d")
            else:
                start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            
            params["date_filter"] = f"AND TIMESTAMP >= '{start_date}'"
        else:
            # Default to 6 months
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            params["date_filter"] = f"AND TIMESTAMP >= '{start_date}'"
        
        # Sector filter
        sector_intent = query_analysis.get("sector_intent")
        if sector_intent:
            sector_mapping = {
                "it": "%IT%",
                "auto": "%AUTO%", 
                "banking": "%NIFTY%",
                "fmcg": "%FMCG%",
                "metal": "%METAL%",
                "pharma": "%PHARMA%"
            }
            
            sector_pattern = sector_mapping.get(sector_intent, f"%{sector_intent.upper()}%")
            params["sector_filter"] = f"AND index_name LIKE '{sector_pattern}'"
        else:
            params["sector_filter"] = ""
        
        # Limit based on complexity
        complexity = query_analysis.get("complexity", "simple")
        if complexity == "simple":
            params["limit"] = "10"
        elif complexity == "moderate":
            params["limit"] = "20"
        else:
            params["limit"] = "30"
        
        return params
    
    def _fill_template(self, template: str, params: Dict[str, str]) -> str:
        """Fill SQL template with parameters"""
        filled_sql = template
        
        for placeholder, value in params.items():
            filled_sql = filled_sql.replace("{" + placeholder + "}", value)
        
        # Remove any remaining placeholders
        import re
        filled_sql = re.sub(r'\{[^}]*\}', '', filled_sql)
        
        return filled_sql.strip()
    
    def _optimize_sql(self, sql: str) -> str:
        """Add SQL optimization and safety checks"""
        # Add SQLite optimization hints
        optimized = f"/* Optimized Query */\n{sql}"
        
        # Ensure proper formatting
        lines = [line.strip() for line in optimized.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def _generate_alternative_sql(self, query_analysis: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alternative SQL when primary query is not feasible"""
        alternatives = validation_result.get("alternatives", [])
        
        if "analyze by index names instead of sectors" in alternatives:
            sql = """
            SELECT 
                CH_SYMBOL,
                index_name,
                CH_CLOSING_PRICE,
                CH_TOT_TRADED_QTY,
                TIMESTAMP
            FROM stock_company_price_daily
            ORDER BY CH_CLOSING_PRICE DESC
            LIMIT 15
            """
        elif "provide snapshot analysis instead" in alternatives:
            sql = """
            SELECT 
                CH_SYMBOL,
                CH_CLOSING_PRICE,
                CH_TOT_TRADED_QTY,
                index_name,
                TIMESTAMP
            FROM stock_company_price_daily
            ORDER BY CH_CLOSING_PRICE DESC
            LIMIT 20
            """
        else:
            sql = self._generate_fallback_sql(query_analysis)
        
        return {
            "sql": sql,
            "template_used": "fallback",
            "parameters": {"reason": "query_not_feasible"},
            "confidence": 0.3
        }
    
    def _generate_fallback_sql(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback SQL for basic analysis"""
        sql = """
        SELECT 
            CH_SYMBOL,
            index_name,
            CH_CLOSING_PRICE,
            CH_TOT_TRADED_QTY,
            CH_TOT_TRADED_VAL,
            TIMESTAMP
        FROM stock_company_price_daily
        WHERE CH_CLOSING_PRICE > 0
        ORDER BY CH_CLOSING_PRICE DESC
        LIMIT 10
        """
        
        return {
            "sql": sql,
            "template_used": "fallback",
            "parameters": {"fallback": "true"},
            "confidence": 0.2
        }

# Enhanced Financial Agent State
class FinancialAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    financial_context: Dict[str, Any]
    query_plan: Optional[Dict[str, Any]]
    available_sectors: List[str]
    date_range: Dict[str, str]
    query_analysis: Optional[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]
    sql_result: Optional[Dict[str, Any]]

# Initialize global schema inspector and related components
schema_inspector = DynamicSchemaInspector(db) if db else None
query_analyzer = IntelligentQueryAnalyzer()
feasibility_validator = QueryFeasibilityValidator(schema_inspector) if schema_inspector else None
sql_generator = DynamicSQLGenerator(schema_inspector, query_analyzer) if schema_inspector else None
datetime_provider = DateTimeProvider(timezone="UTC")
query_classifier = StockMarketQueryClassifier()

# Pre-processing validation node
def preprocess_query_validation(state: FinancialAgentState) -> Dict[str, Any]:
    """Validate the query before proceeding"""
    user_query = safe_extract_user_query(state["messages"])
    
    if not query_classifier.is_stock_market_related(user_query):
        response = AIMessage(content="I can only answer questions about the stock market. Please ask a relevant question.")
        return {"messages": [response]}
    
    # Add context about current date and time
    current_time = datetime_provider.get_current_datetime().strftime("%Y-%m-%d %H:%M:%S %Z")
    location = datetime_provider.get_current_location()
    
    context_message = HumanMessage(
        content=f"Current context: Date and Time: {current_time}, Location: {location}. Query: {user_query}"
    )
    
    state["messages"][-1] = context_message
    
    return state

# Background data fetching for parallel processing
async def fetch_background_data():
    """Pre-fetch commonly needed data in background"""
    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=4) as executor:
            sectors_future = loop.run_in_executor(executor, get_available_sectors)
            date_range_future = loop.run_in_executor(executor, get_date_range, "6 months")
            
            sectors = await sectors_future
            date_range = await date_range_future
            
            logger.info(f"🚀 Background data fetch completed - {len(sectors)} sectors")
            return sectors, date_range
    except Exception as e:
        logger.error(f"Background fetch error: {e}")
        return [], {}

def safe_extract_user_query(messages: List[BaseMessage]) -> str:
    """Safely extract user query from messages list"""
    if not messages:
        return ""
    
    last_message = messages[-1]
    if hasattr(last_message, 'content') and last_message.content:
        return str(last_message.content).lower()
    elif isinstance(last_message, dict) and last_message.get('content'):
        return str(last_message['content']).lower()
    return ""

# Enhanced Financial Query Planning Agent
async def enhanced_financial_query_planner_agent(state: FinancialAgentState) -> Dict[str, Any]:
    """
    Enhanced query planner with dynamic SQL generation capabilities
    """
    user_query = safe_extract_user_query(state["messages"])
    
    # Check cache first
    cache_key = f"query_plan_{hash(user_query)}"
    if cache_key in sql_cache:
        cached_plan = sql_cache[cache_key].copy()
        logger.info(f"📦 Using cached query plan")
        return {
            "messages": state["messages"],
            "financial_context": state.get("financial_context", {}),
            "query_plan": cached_plan["query_plan"],
            "available_sectors": cached_plan["available_sectors"],
            "date_range": cached_plan["date_range"],
            "query_analysis": cached_plan["query_analysis"],
            "validation_result": cached_plan["validation_result"]
        }
    
    # Perform comprehensive query analysis
    query_analysis = query_analyzer.analyze_query(user_query)
    
    # Validate query feasibility
    validation_result = feasibility_validator.validate_query_feasibility(query_analysis)
    
    # Create comprehensive query plan
    query_plan = {
        "original_query": user_query,
        "analysis_type": query_analysis["query_type"],
        "complexity": query_analysis["complexity"],
        "feasible": validation_result["feasible"],
        "confidence": validation_result["confidence"],
        "data_requirements": query_analysis["data_requirements"],
        "sector_intent": query_analysis["sector_intent"],
        "time_period": query_analysis["time_period"],
        "financial_metrics": query_analysis["financial_metrics"],
        "statistical_metrics": query_analysis["statistical_metrics"],
        "requires_aggregation": query_analysis["requires_aggregation"],
        "aggregation_level": query_analysis["aggregation_level"],
        "alternatives": validation_result["alternatives"]
    }
    
    # Cache the comprehensive plan
    cached_data = {
        "query_plan": query_plan,
        "available_sectors": state.get("available_sectors", get_available_sectors()),
        "date_range": state.get("date_range", get_date_range()),
        "query_analysis": query_analysis,
        "validation_result": validation_result
    }
    sql_cache[cache_key] = cached_data
    
    logger.info(f"🔍 Query Analysis Complete - Type: {query_analysis['query_type']}, Feasible: {validation_result['feasible']}")
    
    return {
        "messages": state["messages"],
        "financial_context": state.get("financial_context", {}),
        "query_plan": query_plan,
        "available_sectors": cached_data["available_sectors"],
        "date_range": cached_data["date_range"],
        "query_analysis": query_analysis,
        "validation_result": validation_result
    }

# Enhanced Financial SQL Generator with Dynamic Generation
def enhanced_financial_sql_generator(state: FinancialAgentState) -> Dict[str, Any]:
    """
    Enhanced SQL generator with dynamic query generation based on analysis
    """
    if not run_query_tool:
        return {"messages": [AIMessage(content="Database tools not available")]}

    query_plan = state.get("query_plan", {})
    query_analysis = state.get("query_analysis", {})
    validation_result = state.get("validation_result", {})
    
    logger.info(f"🔧 Generating SQL - Analysis: {query_analysis.get('query_type')}, Confidence: {validation_result.get('confidence', 0)}")
    
    # Generate dynamic SQL
    sql_result = sql_generator.generate_sql(query_analysis, validation_result)
    
    sql_query = sql_result["sql"]
    
    try:
        # Create tool call message for the SQL query
        tool_call = {
            "id": "sql_call_1",
            "name": "sql_db_query", 
            "args": {"query": sql_query}
        }
        
        response = AIMessage(
            content=f"Generated dynamic SQL query using {sql_result['template_used']} template (confidence: {sql_result['confidence']:.2f})",
            tool_calls=[tool_call]
        )
        
        return {
            "messages": [response],
            "sql_result": sql_result
        }
    except Exception as e:
        logger.error(f"Error in SQL generator: {e}")
        return {"messages": [AIMessage(content=f"Error generating SQL: {str(e)}")]}

# Enhanced Financial Data Analyst
def enhanced_financial_data_analyst(state: FinancialAgentState) -> Dict[str, Any]:
    """
    Enhanced financial data analyst with comprehensive analysis capabilities
    """
    user_query = ""
    sql_result = ""
    query_analysis = state.get("query_analysis", {})
    validation_result = state.get("validation_result", {})
    
    # Extract user query
    for m in state["messages"]:
        if isinstance(m, HumanMessage) and m.content:
            user_query = m.content if m.content else ""
            break
    
    # Extract SQL result or query from previous messages
    for m in reversed(state["messages"]):
        if hasattr(m, 'content') and m.content:
            if "Generated dynamic SQL query" in str(m.content):
                # Skip our generated message
                continue
            sql_result = str(m.content)
            break
    
    # Enhanced analysis prompt
    analyst_prompt = f"""You are a senior financial data analyst. Perform comprehensive analysis of the query results.

ORIGINAL USER QUERY: {user_query}
QUERY ANALYSIS TYPE: {query_analysis.get('query_type', 'Unknown')}
FEASIBILITY CONFIDENCE: {validation_result.get('confidence', 0):.2f}
ANALYSIS CONFIDENCE: {validation_result.get('feasible', False)}
REQUIRES AGGREGATION: {query_analysis.get('requires_aggregation', False)}
STATISTICAL METRICS: {query_analysis.get('statistical_metrics', [])}

DATABASE RESULTS: {sql_result}

Provide a detailed analysis covering:

1. **Data Quality Assessment**:
   - Check if results match the query intent
   - Identify any data limitations or gaps
   - Assess completeness of the dataset

2. **Financial/Statistical Insights**:
   - Calculate key performance metrics
   - Identify trends and patterns
   - Highlight significant findings
   - For aggregation queries, focus on grouped statistics

3. **Sector/Index Performance**:
   - Rank performance by relevant metrics
   - Compare across sectors/indices
   - Identify outliers and anomalies

4. **Risk Assessment**:
   - Evaluate volatility indicators
   - Assess liquidity through volume analysis
   - Identify potential risk factors

5. **Investment Implications**:
   - Provide actionable recommendations
   - Suggest follow-up analysis areas
   - Highlight opportunities and concerns

If this is a statistical analysis (standard deviation, variance, etc.), interpret the results in financial context.

Format your response as a professional financial analysis report with specific numbers, percentages, and clear insights."""

    try:
        response = llm.invoke(analyst_prompt)
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in data analyst: {e}")
        return {"messages": [AIMessage(content=f"Error analyzing data: {str(e)}")]}

# Enhanced Financial Response Generator
def enhanced_financial_response_generator(state: FinancialAgentState) -> Dict[str, Any]:
    """
    Enhanced financial response generator with adaptive messaging
    """
    user_query = ""
    analysis = ""
    validation_result = state.get("validation_result", {})
    sql_result = state.get("sql_result", {})
    query_analysis = state.get("query_analysis", {})
    
    # Extract user query
    for m in state["messages"]:
        if isinstance(m, HumanMessage) and m.content:
            user_query = m.content if m.content else ""
            break
    
    # Extract analysis from previous messages
    for m in reversed(state["messages"]):
        if hasattr(m, 'content') and m.content and not isinstance(m, HumanMessage):
            if "Generated dynamic SQL query" not in str(m.content):
                analysis = str(m.content)
                break

    # Adaptive response based on query feasibility
    confidence = validation_result.get("confidence", 0)
    feasible = validation_result.get("feasible", False)
    requires_aggregation = query_analysis.get("requires_aggregation", False)
    
    if confidence < 0.5:
        # Low confidence response - be transparent about limitations
        response_prompt = f"""You are a financial advisor. The user's query has limited data support, so provide a transparent response.

ORIGINAL QUESTION: {user_query}
DATA ANALYSIS: {analysis}
CONFIDENCE LEVEL: {confidence:.2f} (LOW)

Create a response that:
- Acknowledges the data limitations upfront
- Explains what analysis was possible with available data
- Provides the best insights possible given the constraints
- Suggests alternative approaches or additional data that would help
- Maintains professional credibility while being honest about limitations

Keep the response under 100 words but comprehensive for the given constraints."""
    else:
        # Standard response for feasible queries
        aggregation_context = ""
        if requires_aggregation:
            aggregation_context = "This is an aggregated analysis showing results grouped by index/sector."
        
        response_prompt = f"""You are a financial advisor. Provide a comprehensive response to the user's question.

{aggregation_context}
ORIGINAL QUESTION: {user_query}
ANALYSIS: {analysis}

Create a response that:
- Directly answers the user's specific question
- Includes key performance numbers and rankings
- Highlights the best/worst performers with specific metrics
- Uses professional but accessible language
- Is comprehensive but concise (under 100 words)
- Includes specific data points, percentages, and actionable insights

Focus on delivering value while maintaining accuracy and professionalism."""

    try:
        response = llm.invoke(response_prompt)
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in response generator: {e}")
        return {"messages": [AIMessage(content=f"Error generating response: {str(e)}")]}

# Query Complexity Router
def financial_query_router(state: FinancialAgentState) -> Literal["simple_financial", "complex_financial", "basic_query"]:
    """
    Route queries based on enhanced financial complexity analysis
    """
    query_analysis = state.get("query_analysis", {})
    validation_result = state.get("validation_result", {})
    
    # Enhanced routing logic based on comprehensive analysis
    confidence = validation_result.get("confidence", 0)
    complexity = query_analysis.get("complexity", "simple")
    requires_aggregation = query_analysis.get("requires_aggregation", False)
    
    # Statistical queries require complex routing regardless of other factors
    if any(word in query_analysis.get("query_type", "") for word in ["statistical", "aggregation"]):
        return "complex_financial"
    
    if confidence < 0.3:
        return "basic_query"  # Limited data, simple response
    elif complexity == "complex" or query_analysis.get("analysis_type") == "comparative_analysis":
        return "complex_financial"
    elif requires_aggregation:
        return "complex_financial"
    else:
        return "simple_financial"

# Utility Functions
def get_available_sectors() -> List[str]:
    """Get list of available sectors from database with caching"""
    cache_key = "available_sectors"
    if cache_key in financial_cache:
        return financial_cache[cache_key]
    
    try:
        if db:
            with db._engine.connect() as conn:
                result1 = conn.execute(text("SELECT DISTINCT index_name FROM stock_index_price_daily WHERE index_name IS NOT NULL"))
                sectors1 = [row[0] for row in result1 if row[0]]
                
                result2 = conn.execute(text("SELECT DISTINCT index_name FROM stock_company_price_daily WHERE index_name IS NOT NULL"))
                sectors2 = [row[0] for row in result2 if row[0]]
                
                all_sectors = list(set(sectors1 + sectors2))
                financial_cache[cache_key] = all_sectors
                return all_sectors
        return []
    except Exception as e:
        logger.error(f"Error fetching sectors: {e}")
        return []

def get_date_range(period: str = "6 months") -> Dict[str, str]:
    """Calculate date range for financial analysis with caching"""
    cache_key = f"date_range_{period}"
    if cache_key in financial_cache:
        return financial_cache[cache_key]
    
    end_date = datetime.now()
    period_lower = period.lower() if period else "6 months"
    
    if "day" in period_lower or period_lower.isdigit():
        days = int(period_lower.split()[0]) if period_lower.split()[0].isdigit() else int(period_lower)
        start_date = end_date - timedelta(days=days)
    elif "week" in period_lower:
        weeks = int(period_lower.split()[0])
        start_date = end_date - timedelta(weeks=weeks)
    elif "month" in period_lower:
        months = int(period_lower.split()[0])
        start_date = end_date - timedelta(days=months*30)
    elif "year" in period_lower:
        years = int(period_lower.split()[0])
        start_date = end_date - timedelta(days=years*365)
    else:
        start_date = end_date - timedelta(days=180)
    
    date_range = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d")
    }
    
    financial_cache[cache_key] = date_range
    return date_range

# Enhanced Financial Agent Graph
financial_builder = StateGraph(FinancialAgentState)

# Add nodes
financial_builder.add_node("preprocess_query", preprocess_query_validation)
financial_builder.add_node("query_planner", enhanced_financial_query_planner_agent)
financial_builder.add_node("enhanced_sql_generator", enhanced_financial_sql_generator)
financial_builder.add_node("enhanced_data_analyst", enhanced_financial_data_analyst)
financial_builder.add_node("enhanced_response_generator", enhanced_financial_response_generator)

if run_query_tool:
    run_query_node = ToolNode([run_query_tool], name="run_query")
    financial_builder.add_node("run_query", run_query_node)

# Enhanced routing and connections
financial_builder.add_edge(START, "preprocess_query")
financial_builder.add_conditional_edges(
    "preprocess_query",
    lambda state: "continue" if "content" not in state["messages"][-1].content else "end",
    {"continue": "query_planner", "end": END}
)
financial_builder.add_conditional_edges(
    "query_planner",
    financial_query_router,
    {
        "complex_financial": "enhanced_sql_generator",
        "simple_financial": "enhanced_sql_generator",
        "basic_query": "enhanced_sql_generator"
    }
)

if run_query_tool:
    financial_builder.add_edge("enhanced_sql_generator", "run_query")
    financial_builder.add_edge("run_query", "enhanced_data_analyst")
else:
    financial_builder.add_edge("enhanced_sql_generator", "enhanced_data_analyst")

financial_builder.add_edge("enhanced_data_analyst", "enhanced_response_generator")
financial_builder.add_edge("enhanced_response_generator", END)

financial_agent = financial_builder.compile()

# Original agents for fallback
sql_generator_agent = enhanced_financial_query_planner_agent
data_analyst_agent = enhanced_financial_data_analyst
response_generator_agent = enhanced_financial_response_generator

# Optimized async database operations
async def get_history(session_id: str, limit: int = 10) -> list:
    """Optimized history fetch with caching"""
    cache_key = f"history_{session_id}_{limit}"
    
    if cache_key in llm_cache:
        return llm_cache[cache_key]
    
    try:
        async with aiosqlite.connect("stock.db") as db_conn:
            cursor = await db_conn.execute(
                "SELECT messages FROM conversation_history WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit),
            )
            result = await cursor.fetchone()
        
        if result:
            messages = messages_from_dict(json.loads(result[0]))
            cached_messages = messages[-limit:]
            llm_cache[cache_key] = cached_messages
            return cached_messages
        return []
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return []

async def save_history(session_id: str, messages: list):
    """Optimized history save with batching"""
    try:
        messages_data = json.dumps([message_to_dict(m) for m in messages])
        
        async with aiosqlite.connect("stock.db") as db_conn:
            await db_conn.execute(
                "INSERT OR REPLACE INTO conversation_history (session_id, messages, timestamp) VALUES (?, ?, ?)",
                (session_id, messages_data, datetime.now()),
            )
            await db_conn.commit()
    except Exception as e:
        logger.error(f"Error saving history: {e}")

# FastAPI application
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None

@app.post("/query")
async def run_enhanced_agent_query(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    
    # Start background data fetching immediately
    background_task = asyncio.create_task(fetch_background_data())
    
    # Log incoming query for testing
    logger.info(f"🔍 RECEIVED QUERY - Session: {session_id}")
    logger.info(f"📝 Query: {request.query}")
    logger.info("=" * 80)
    
    history = await get_history(session_id)
    messages = history + [HumanMessage(content=request.query)]

    async def stream_response():
        try:
            logger.info(f"🚀 STARTING ENHANCED ANALYSIS - Session: {session_id}")
            logger.info(f"📊 Processing with {len(messages)} total messages")
            
            # Wait for background data
            try:
                sectors, date_range = await asyncio.wait_for(background_task, timeout=2.0)
                logger.info(f"✅ Background data ready - {len(sectors)} sectors")
            except asyncio.TimeoutError:
                logger.warning("⚠️ Background fetch timeout, proceeding without pre-fetched data")
                sectors, date_range = get_available_sectors(), get_date_range()
            
            final_state = None
            stream_count = 0
            
            async for s in financial_agent.astream({
                "messages": messages,
                "financial_context": {},
                "query_plan": None,
                "available_sectors": sectors,
                "date_range": date_range,
                "query_analysis": None,
                "validation_result": None,
                "sql_result": None
            }):
                final_state = s
                stream_count += 1

                for key, value in s.items():
                    if "__end__" == key:
                        break
                    
                    if value and isinstance(value, dict) and "messages" in value:
                        stage_map = {
                            "preprocess_query": "planning",
                            "query_planner": "planning",
                            "enhanced_sql_generator": "sql",
                            "run_query": "query",
                            "enhanced_data_analyst": "analysis",
                            "enhanced_response_generator": "final"
                        }
                        stage = stage_map.get(key)
                        if stage:
                            latest_message = value["messages"][-1] if value["messages"] else None
                            response_data = {"stage": stage}
                            if latest_message and hasattr(latest_message, 'content'):
                                response_data["content"] = latest_message.content
                            
                            logger.info(f"📤 STREAMING STAGE: {stage} - Session: {session_id}")
                            yield json.dumps(response_data) + "\n"

            if not final_state:
                error_response = {"response": "Enhanced agent failed to produce a response.", "session_id": session_id}
                logger.error(f"❌ NO FINAL STATE - Session: {session_id}")
                yield json.dumps(error_response) + "\n"
                return

            final_messages = list(final_state.values())[-1]["messages"]
            await save_history(session_id, final_messages)

            final_answer = "No response."
            for message in reversed(final_messages):
                if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                    final_answer = message.content
                    break
            
            final_response = {"response": final_answer, "session_id": session_id}
            
            # Final logging
            logger.info(f"✅ ENHANCED FINAL RESPONSE - Session: {session_id}")
            logger.info(f"💬 Final Answer: {final_answer[:200]}...")
            logger.info("=" * 80)
            
            yield json.dumps(final_response) + "\n"
            
            # Clean up memory
            del final_state, final_messages
            gc.collect()
            
        except Exception as e:
            error_response = {"error": f"Enhanced processing error: {str(e)}", "session_id": session_id}
            logger.error(f"💥 EXCEPTION - Session: {session_id}")
            logger.error(f"🔴 Error Details: {str(e)}")
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            yield json.dumps(error_response) + "\n"

    return StreamingResponse(stream_response(), media_type="application/x-ndjson")

@app.get("/")
async def root():
    return {
        "message": "Enhanced Financial Analyst API with Aggregation-Optimized SQL Generation",
        "capabilities": [
            "Intelligent query analysis and planning with aggregation detection",
            "Dynamic SQL generation based on schema with statistical analysis templates",
            "Query feasibility validation with fallback mechanisms",
            "Enhanced sector-based stock performance analysis with proper grouping",
            "Statistical analysis support (standard deviation, variance, volatility)",
            "Aggregation queries properly grouped by index/sector",
            "Adaptive response generation based on data availability",
            "Comprehensive error handling and recovery",
            "Real-time database schema introspection",
            "Flexible financial metrics calculation"
        ]
    }

@app.get("/schema")
async def get_schema_info():
    """Endpoint to get database schema information"""
    if not schema_inspector:
        return {"error": "Schema inspector not available"}
    
    return {
        "schema": schema_inspector.schema_info,
        "data_summary": schema_inspector.get_data_summary(),
        "cached_tables": list(schema_inspector.schema_info.get("tables", {}).keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
