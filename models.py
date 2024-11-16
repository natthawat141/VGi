# models.py

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np

@dataclass
class FinancialMetrics:
    """
    คลาสสำหรับเก็บตัวชี้วัดทางการเงินต่าง ๆ
    """
    basic_metrics: Dict[str, float]
    technical_indicators: Dict[str, Any]
    risk_metrics: Dict[str, float]
    statistical_metrics: Dict[str, Any]

@dataclass
class IndustryAnalysis:
    """
    คลาสสำหรับเก็บข้อมูลการวิเคราะห์อุตสาหกรรม
    """
    industry_type: str
    key_indicators: List[str]
    study_topics: List[str]
    peer_companies: List[str]
    industry_metrics: Dict[str, float]

@dataclass
class FinancialData:
    """
    คลาสสำหรับเก็บข้อมูลทางการเงินของหุ้น
    """
    symbol: str
    prices: np.ndarray
    returns: np.ndarray
    dates: np.ndarray
    volume: np.ndarray
    market_cap: float
    sector: str
    financial_metrics: Optional[FinancialMetrics] = None
