# main.py

# ----------------- นำเข้าไลบรารีที่จำเป็น -----------------

import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional, Dict, Any, List
import numpy as np

# ไลบรารีสำหรับการประมวลผลข้อมูลและการวิเคราะห์ทางการเงิน
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
import requests
import re

# ไลบรารีสำหรับการจัดการไฟล์ PDF และการ OCR
import fitz  # PyMuPDF
import io
from PIL import Image
import pytesseract

# ไลบรารีสำหรับการประมวลผลภาษาธรรมชาติ (NLP)
import pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize
from transformers import pipeline

# ไลบรารีสำหรับการวิเคราะห์สถิติและการสร้างกราฟ
import seaborn as sns
from scipy import stats

# ปิดคำเตือนที่ไม่จำเป็น
warnings.filterwarnings('ignore')
sns.set(style='whitegrid')


from dotenv import load_dotenv

load_dotenv()  # โหลด .env
api_key = os.getenv('TYPHOON_API_KEY')




# ----------------- กำหนด Data Classes -----------------

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

# ----------------- คลาส MarketData -----------------

class MarketData:
    """
    คลาสสำหรับดึงข้อมูลตลาดหุ้น
    """
    def __init__(self):
        self.market_index = "^SET.BK"  # ดัชนีตลาดหลักทรัพย์ของไทย
        self.cache = {}  # แคชสำหรับเก็บข้อมูลหุ้นที่ดึงมาแล้ว

    def get_stock_data(self, symbol: str, period: str = "1y") -> Optional[FinancialData]:
        """
        ดึงข้อมูลหุ้นสำหรับสัญลักษณ์ที่ระบุและช่วงเวลา
        """
        print(f"\nดึงข้อมูลหุ้นสำหรับ {symbol}")
        try:
            # ตรวจสอบว่ามีข้อมูลในแคชหรือไม่
            if symbol in self.cache:
                print(f"ใช้ข้อมูลที่แคชไว้สำหรับ {symbol}")
                return self.cache[symbol]

            # ดึงข้อมูลหุ้นจาก yfinance
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)

            # ตรวจสอบว่ามีข้อมูลหรือไม่
            if hist.empty:
                raise ValueError(f"ไม่พบข้อมูลสำหรับ {symbol}")

            # ดึงข้อมูลเพิ่มเติมเกี่ยวกับหุ้น
            info = stock.info
            data = FinancialData(
                symbol=symbol,
                prices=hist['Close'].values,
                returns=hist['Close'].pct_change().dropna().values,
                dates=hist.index.values,
                volume=hist['Volume'].values,
                market_cap=info.get('marketCap', 0),
                sector=info.get('sector', 'Unknown')
            )

            # เก็บข้อมูลในแคช
            self.cache[symbol] = data
            print(f"ดึงข้อมูลสำเร็จสำหรับ {symbol}")
            return data

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

# ----------------- คลาส AdvancedFinancialAnalyzer -----------------

class AdvancedFinancialAnalyzer:
    """
    คลาสหลักสำหรับการวิเคราะห์ทางการเงินขั้นสูง
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key  # API Key สำหรับ Typhoon API
        print("กำลังโหลดโมเดลจำแนกประเภท...")

        # โหลดโมเดลสำหรับการจำแนกประเภทด้วย Zero-Shot Classification
        self.classifier = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
            tokenizer="joeddav/xlm-roberta-large-xnli",
            device=-1  # ใช้ CPU (-1)
        )
        self.market_data = MarketData()  # สร้างอินสแตนซ์ของ MarketData

    def extract_text_from_pdf(self, file_path: str) -> Optional[str]:
        """
        ดึงและทำความสะอาดข้อความจากไฟล์ PDF
        """
        print(f"\nกำลังดึงข้อความจากไฟล์ PDF: {file_path}")
        try:
            text = ""
            # เปิดไฟล์ PDF
            with fitz.open(file_path) as pdf:
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    page_text = page.get_text()
                    if page_text.strip():
                        # ถ้ามีข้อความอยู่แล้ว
                        text += page_text
                    else:
                        # ถ้าไม่มีข้อความ ให้ใช้ OCR
                        pix = page.get_pixmap()
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        ocr_text = pytesseract.image_to_string(img, lang='tha+eng')
                        text += ocr_text
            # ทำความสะอาดข้อความ
            cleaned_text = self._clean_text(text) if text else None
            print("\nข้อความหลังทำความสะอาด:")
            print(cleaned_text[:500])  # แสดง 500 ตัวอักษรแรก
            return cleaned_text
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """
        ทำความสะอาดและปรับปรุงข้อความ โดยรักษาข้อมูลทางการเงินที่สำคัญ
        """
        print("\nกำลังทำความสะอาดข้อความ...")
        # ลบช่องว่างเกินความจำเป็น
        text = re.sub(r'\s+', ' ', text)
        # รักษาตัวเลขและสัญลักษณ์ทางการเงิน และตัวอักษรภาษาไทย
        text = re.sub(r'[^\w\s$.,%-]', '', text, flags=re.UNICODE)
        # ปรับรูปแบบตัวเลข
        text = re.sub(r'(\d),(\d)', r'\1\2', text)
        print("ทำความสะอาดข้อความสำเร็จ")
        return text.strip()

    def find_stock_symbol(self, text: str) -> Optional[str]:
        """
        ค้นหาสัญลักษณ์หุ้นจากข้อความโดยใช้ Typhoon API
        """
        print("\nกำลังค้นหาสัญลักษณ์หุ้นจากข้อความ...")
        if not self.api_key:
            print("ไม่สามารถค้นหาสัญลักษณ์หุ้นได้เนื่องจากไม่ได้ตั้งค่า Typhoon API Key")
            return None

        # สร้าง prompt สำหรับส่งไปยัง Typhoon API
        prompt = f"จากข้อความต่อไปนี้ คุณทราบสัญลักษณ์หุ้นของบริษัทนี้ใน yfinance หรือไม่:\n\n{text[:2000]}"
        response = self._call_typhoon_api(prompt)
        if response:
            # ดึงคำตอบจาก API
            answer = response['choices'][0]['message']['content']
            print(f"\nคำตอบจาก Typhoon API:\n{answer}")
            # ใช้ Regular Expression เพื่อค้นหาสัญลักษณ์หุ้น
            symbol_match = re.search(r'สัญลักษณ์หุ้น.*?คือ\s*"?([A-Z]{1,5}(?:\.BK)?)"?', answer)
            if symbol_match:
                symbol = symbol_match.group(1)
                # เพิ่ม ".BK" หากเป็นหุ้นไทยและไม่มีนามสกุล
                if '.BK' not in symbol and symbol.isupper():
                    symbol += '.BK'
                print(f"\nพบสัญลักษณ์หุ้น: {symbol}")
                return symbol
            else:
                print("ไม่สามารถดึงสัญลักษณ์หุ้นจากคำตอบของ Typhoon")
                return None
        else:
            return None

    def analyze_company(self, text: str) -> (str, Dict[str, Any]):
        """
        วิเคราะห์บริษัทอย่างครบถ้วนจากข้อความที่ให้มา
        """
        # ค้นหาสัญลักษณ์หุ้น
        symbol = self.find_stock_symbol(text)
        if not symbol:
            print("ไม่สามารถระบุสัญลักษณ์หุ้นได้")
            stock_data = None
        else:
            # ดึงข้อมูลตลาดหุ้น
            stock_data = self.market_data.get_stock_data(symbol)
            if stock_data is None:
                print(f"ไม่สามารถดึงข้อมูลหุ้น {symbol} ได้")

        # จำแนกประเภทธุรกิจ
        industry = self._classify_industry(text)

        print(f"\nบริษัทนี้อยู่ในอุตสาหกรรม: {industry}")
        print("คุณควรศึกษาปัจจัยเฉพาะของอุตสาหกรรมนี้เพิ่มเติมเพื่อการตัดสินใจลงทุน")

        # คำนวณตัวชี้วัดทางการเงินถ้ามีข้อมูลหุ้น
        if stock_data:
            financial_metrics = self._calculate_financial_metrics(stock_data)
        else:
            financial_metrics = None

        # วิเคราะห์ปัจจัยเฉพาะของอุตสาหกรรม
        industry_analysis = self._analyze_industry_specifics(text, industry, symbol)

        # รับคำแนะนำจาก AI
        ai_analysis = self._get_ai_analysis(text, industry, financial_metrics)

        return industry, {
            'symbol': symbol,
            'financial_metrics': financial_metrics,
            'industry_analysis': industry_analysis,
            'ai_recommendations': ai_analysis,
            'stock_data': stock_data
        }

    def _classify_industry(self, text: str) -> str:
        """
        จำแนกประเภทธุรกิจของบริษัทจากข้อความโดยใช้ Typhoon API และ Zero-Shot Classification
        """
        print("\nกำลังสรุปข้อความเพื่อจำแนกประเภทอุตสาหกรรม...")
        # สรุปข้อความโดยใช้ Typhoon API
        summary = self._summarize_with_typhoon(text)
        if not summary:
            summary = text[:1024]  # ใช้ข้อความเดิมถ้าสรุปไม่สำเร็จ
        print(f"\nสรุปข้อความ:\n{summary}")

        # ดึงคำที่เกี่ยวข้องกับอุตสาหกรรมจากข้อความ
        dynamic_labels = self._extract_industries_from_text(summary)
        if not dynamic_labels:
            dynamic_labels = ['ธนาคาร', 'ก่อสร้าง', 'เทคโนโลยี', 'สุขภาพ', 'พลังงาน', 'อสังหาริมทรัพย์', 'การขนส่ง', 'การสื่อสาร', 'การเกษตร', 'อาหาร', 'การท่องเที่ยว', 'ค้าปลีก', 'การผลิต', 'การเงิน', 'สื่อ', 'บันเทิง', 'เคมีภัณฑ์']

        print(f"\nLabels สำหรับการจำแนกประเภท: {dynamic_labels}")

        print("\nกำลังจำแนกประเภทอุตสาหกรรม...")
        # ใช้โมเดล Zero-Shot Classification
        result = self.classifier(summary, candidate_labels=dynamic_labels, hypothesis_template="นี่คือข้อความเกี่ยวกับ {}.")
        industry = result['labels'][0]
        print(f"\nประเภทอุตสาหกรรมที่จำแนกได้: {industry}")
        return industry

    def _summarize_with_typhoon(self, text: str) -> Optional[str]:
        """
        สรุปข้อความโดยใช้ Typhoon API
        """
        if not self.api_key:
            print("ไม่สามารถสรุปข้อความด้วย Typhoon API ได้เนื่องจากไม่ได้ตั้งค่า API Key")
            return None

        prompt = f"กรุณาสรุปข้อความต่อไปนี้:\n\n{text[:2000]}"
        response = self._call_typhoon_api(prompt)
        if response:
            summary = response['choices'][0]['message']['content'].strip()
            return summary
        else:
            return None

    def _extract_industries_from_text(self, text: str) -> List[str]:
        """
        สกัดคำที่เกี่ยวข้องกับอุตสาหกรรมจากข้อความโดยใช้ NLP
        """
        # ทำการตัดคำและลบคำซ้ำ
        words = set(word_tokenize(normalize(text), engine='newmm'))

        # คำที่เกี่ยวข้องกับประเภทธุรกิจทั่วไป
        common_industry_terms = [
            'ธนาคาร', 'ก่อสร้าง', 'เทคโนโลยี', 'สุขภาพ', 'พลังงาน', 'อสังหาริมทรัพย์',
            'การขนส่ง', 'การสื่อสาร', 'การเกษตร', 'อาหาร', 'การท่องเที่ยว',
            'ค้าปลีก', 'การผลิต', 'การเงิน', 'สื่อ', 'บันเทิง', 'เคมีภัณฑ์',
            # เพิ่มเติมคำอื่น ๆ ที่เกี่ยวข้อง
        ]

        # สกัดคำที่ตรงกับประเภทธุรกิจ
        extracted_industries = [word for word in words if word in common_industry_terms]

        # ลบคำซ้ำและคืนค่าเป็นรายการ
        return list(set(extracted_industries))

    def _calculate_financial_metrics(self, stock_data: FinancialData) -> FinancialMetrics:
        """
        คำนวณตัวชี้วัดทางการเงินอย่างครอบคลุม
        """
        print("\nกำลังคำนวณตัวชี้วัดทางการเงิน...")
        prices = stock_data.prices
        returns = stock_data.returns

        financial_metrics = FinancialMetrics(
            basic_metrics=self._calculate_basic_metrics(stock_data),
            technical_indicators=self._calculate_technical_indicators(prices),
            risk_metrics=self._calculate_risk_metrics(returns),
            statistical_metrics=self._calculate_statistical_metrics(returns)
        )
        print("\nคำนวณตัวชี้วัดทางการเงินสำเร็จ")
        return financial_metrics

    def _calculate_basic_metrics(self, stock_data: FinancialData) -> Dict[str, float]:
        """
        คำนวณอัตราส่วนทางการเงินพื้นฐาน
        """
        print(" - กำลังคำนวณอัตราส่วนทางการเงินพื้นฐาน...")
        try:
            current_price = stock_data.prices[-1]
            basic_metrics = {
                'price': current_price,
                'market_cap': stock_data.market_cap,
                'pe_ratio': self._safe_calc(lambda: current_price / self._get_eps(stock_data.symbol)),
                'pb_ratio': self._safe_calc(lambda: current_price / self._get_book_value(stock_data.symbol)),
                'roe': self._safe_calc(lambda: self._get_net_income(stock_data.symbol) / self._get_equity(stock_data.symbol))
            }
            print(f"\nอัตราส่วนทางการเงินพื้นฐาน: {basic_metrics}")
            return basic_metrics
        except Exception as e:
            print(f"Error calculating basic metrics: {e}")
            return {}

    def _calculate_technical_indicators(self, prices: np.ndarray) -> Dict[str, Any]:
        """
        คำนวณตัวชี้วัดทางเทคนิคสำหรับการวิเคราะห์กราฟ
        """
        print(" - กำลังคำนวณตัวชี้วัดทางเทคนิค...")
        try:
            technical_indicators = {
                'rsi': self._calculate_rsi(prices),
                'macd': self._calculate_macd(prices),
                'bollinger_bands': self._calculate_bollinger_bands(prices),
                'moving_averages': {
                    'ma20': np.mean(prices[-20:]),
                    'ma50': np.mean(prices[-50:]),
                    'ma200': np.mean(prices[-200:])
                }
            }
            print(f"\nตัวชี้วัดทางเทคนิค: {technical_indicators}")
            return technical_indicators
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return {}

    def _calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        คำนวณตัวชี้วัดความเสี่ยงและประสิทธิภาพการลงทุน
        """
        print(" - กำลังคำนวณตัวชี้วัดความเสี่ยง...")
        try:
            risk_metrics = {
                'volatility': np.std(returns) * np.sqrt(252),
                'var_95': np.percentile(returns, 5),
                'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]),
                'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252),
                'sortino_ratio': np.mean(returns) / np.std(returns[returns < 0]) * np.sqrt(252)
            }
            print(f"\nตัวชี้วัดความเสี่ยง: {risk_metrics}")
            return risk_metrics
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return {}

    def _calculate_statistical_metrics(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        คำนวณตัวชี้วัดทางสถิติสำหรับผลตอบแทน
        """
        print(" - กำลังคำนวณตัวชี้วัดสถิติ...")
        try:
            # ทดสอบความเป็นสถิติคงที่ (Stationarity)
            adf_stat, adf_pvalue = stats.adfuller(returns)[:2]

            # ทดสอบความเป็นปกติ (Normality)
            shapiro_stat, shapiro_pvalue = stats.shapiro(returns)

            statistical_metrics = {
                'stationarity': {
                    'adf_statistic': adf_stat,
                    'adf_pvalue': adf_pvalue
                },
                'normality': {
                    'shapiro_statistic': shapiro_stat,
                    'shapiro_pvalue': shapiro_pvalue
                },
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns)
            }
            print(f"\nตัวชี้วัดสถิติ: {statistical_metrics}")
            return statistical_metrics
        except Exception as e:
            print(f"Error calculating statistical metrics: {e}")
            return {}

    def _get_ai_analysis(self, text: str, industry: str, metrics: Optional[FinancialMetrics]) -> Dict[str, Any]:
        """
        รับการวิเคราะห์และคำแนะนำจาก AI โดยใช้ Typhoon API
        """
        if not self.api_key:
            print("ไม่สามารถขอคำแนะนำจาก Typhoon API ได้เนื่องจากไม่ได้ตั้งค่า API Key")
            return {}

        print("\nกำลังขอคำแนะนำจาก Typhoon API...")
        # สร้าง prompt สำหรับ Typhoon API
        investment_factors_prompt = f"จากข้อมูลนี้ หากต้องการลงทุนในบริษัทนี้ ควรพิจารณาปัจจัยใดบ้างเพื่อทำการตัดสินใจในการลงทุน?"
        indicators_prompt = f"สำหรับธุรกิจในอุตสาหกรรม {industry} กรุณาแนะนำตัวชี้วัดทางการเงินที่ควรศึกษาเพิ่มเติมในการประเมินการลงทุน"

        # ดึงตัวเลขทางการเงินสำคัญจากข้อความ
        financial_numbers = self.extract_financial_numbers(text)
        print("\nตัวเลขทางการเงินที่ดึงได้จากข้อความ:")
        print(financial_numbers)

        # รวม prompt และข้อมูลทางการเงิน
        trimmed_text = "\n".join(financial_numbers)[:2000]
        full_prompt = f"""
นี่คือข้อมูลทางการเงินที่สรุปแล้วจากเอกสาร:
{trimmed_text}

{investment_factors_prompt}

{indicators_prompt}
"""

        # ส่งไปยัง Typhoon API
        response = self._call_typhoon_api(full_prompt)

        if response:
            summary = response['choices'][0]['message']['content']
            print(f"\nคำแนะนำจาก Typhoon API:\n{summary}")
            return {
                'summary': summary,
                'confidence': response.get('confidence', 0.0),
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {}

    def extract_financial_numbers(self, text: str) -> List[str]:
        """
        สกัดตัวเลขทางการเงินสำคัญจากข้อความ
        """
        pattern = r'([ก-๙a-zA-Z ]+)\s+([\d,]+\.\d{2}|\d+,?\d*)'
        matches = re.findall(pattern, text)
        return [f"{item.strip()}: {value.strip()}" for item, value in matches]

    def _call_typhoon_api(self, prompt: str) -> Optional[Dict]:
        """
        เรียกใช้ Typhoon API ด้วย prompt ที่กำหนด
        """
        print("\nเรียกใช้งาน Typhoon API...")
        endpoint = 'https://api.opentyphoon.ai/v1/chat/completions'
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "typhoon-v1.5-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1024,
            "top_p": 0.9,
            "top_k": 0,
            "repetition_penalty": 1.05,
            "min_p": 0
        }

        try:
            response = requests.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            print("เรียกใช้งาน Typhoon API สำเร็จ")
            return response.json()
        except Exception as e:
            print(f"API call error: {e}")
            return None

    def analyze(self, action: str, params: Dict[str, Any]) -> Any:
        """
        ฟังก์ชันหลักสำหรับเรียกใช้ฟังก์ชันต่าง ๆ ตาม action ที่ระบุ
        """
        if action == 'extract_text_from_pdf':
            file_path = params.get('file_path')
            return self.extract_text_from_pdf(file_path)
        elif action == 'find_stock_symbol':
            text = params.get('text')
            return self.find_stock_symbol(text)
        elif action == 'analyze_company':
            text = params.get('text')
            return self.analyze_company(text)
        elif action == 'calculate_rsi':
            prices = params.get('prices')
            period = params.get('period', 14)
            return self._calculate_rsi(np.array(prices), period)
        # เพิ่มเงื่อนไขสำหรับฟังก์ชันอื่น ๆ ตามต้องการ
        else:
            raise ValueError(f"ไม่พบ action '{action}'")

    # ----------------- ฟังก์ชัน Helper และฟังก์ชันเพิ่มเติม -----------------

    def _safe_calc(self, func):
        """
        ดำเนินการคำนวณอย่างปลอดภัย
        """
        try:
            return func()
        except:
            return None

    def _get_eps(self, symbol: str) -> float:
        """
        ดึงค่า EPS สำหรับสัญลักษณ์หุ้น
        """
        # นี่เป็น placeholder สำหรับการดึงข้อมูลจริง
        return 5.0

    def _get_book_value(self, symbol: str) -> float:
        """
        ดึงค่า Book Value ต่อหุ้น
        """
        # นี่เป็น placeholder สำหรับการดึงข้อมูลจริง
        return 50.0

    def _get_net_income(self, symbol: str) -> float:
        """
        ดึงค่า Net Income
        """
        # นี่เป็น placeholder สำหรับการดึงข้อมูลจริง
        return 1000000.0

    def _get_equity(self, symbol: str) -> float:
        """
        ดึงค่า Total Equity
        """
        # นี่เป็น placeholder สำหรับการดึงข้อมูลจริง
        return 5000000.0

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """
        คำนวณค่า Relative Strength Index (RSI)
        """
        delta = np.diff(prices)
        up = delta.clip(min=0)
        down = -1 * delta.clip(max=0)
        avg_gain = np.mean(up[-period:])
        avg_loss = np.mean(down[-period:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: np.ndarray) -> float:
        """
        คำนวณค่า Moving Average Convergence Divergence (MACD)
        """
        exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd.iloc[-1] - signal.iloc[-1]

    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20) -> Dict[str, float]:
        """
        คำนวณค่า Bollinger Bands
        """
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        return {'upper_band': upper_band, 'lower_band': lower_band}

    def _analyze_industry_specifics(self, text: str, industry: str, symbol: Optional[str]) -> Dict[str, Any]:
        """
        วิเคราะห์ปัจจัยเฉพาะของอุตสาหกรรม
        """
        print("\nกำลังวิเคราะห์ปัจจัยเฉพาะของอุตสาหกรรม...")
        # หาบริษัทที่คล้ายกัน
        peer_companies = self._find_similar_companies(text, symbol)
        if symbol and symbol in peer_companies:
            peer_companies.remove(symbol)  # เอาบริษัทที่เราวิเคราะห์ออก

        return {
            'study_topics': self._get_study_topics(industry),
            'peer_companies': peer_companies,
        }

    def _find_similar_companies(self, text: str, symbol: Optional[str]) -> List[str]:
        """
        หาบริษัทที่คล้ายกันโดยใช้ Typhoon API และ NLP
        """
        if not self.api_key:
            print("ไม่สามารถหาบริษัทที่คล้ายกันได้เนื่องจากไม่ได้ตั้งค่า Typhoon API Key")
            return []

        print("\nกำลังหาบริษัทที่คล้ายกันโดยใช้ Typhoon API...")
        if symbol:
            prompt = f"สำหรับบริษัทที่มีชื่อสัญลักษณ์ {symbol} กรุณาระบุชื่อบริษัทที่คล้ายกันในอุตสาหกรรมเดียวกันพร้อมสัญลักษณ์หุ้นของพวกเขา"
        else:
            prompt = f"จากข้อมูลต่อไปนี้ กรุณาระบุชื่อบริษัทที่คล้ายกันในอุตสาหกรรมเดียวกันพร้อมสัญลักษณ์หุ้นของพวกเขา:\n\n{text[:2000]}"

        response = self._call_typhoon_api(prompt)
        if response:
            content = response['choices'][0]['message']['content']
            print(f"\nคำตอบจาก Typhoon API:\n{content}")
            # สกัดสัญลักษณ์หุ้นจากคำตอบ
            stock_symbols = self._extract_stock_symbols(content)
            print(f"\nสัญลักษณ์หุ้นที่พบ: {stock_symbols}")
            return stock_symbols
        return []

    def _extract_stock_symbols(self, text: str) -> List[str]:
        """
        สกัดสัญลักษณ์หุ้นจากข้อความโดยใช้ regex
        """
        pattern = r'\b[A-Z]{1,5}(?:\.BK)?\b'
        symbols = re.findall(pattern, text)
        # ลบสัญลักษณ์ซ้ำ
        symbols = list(set(symbols))
        return symbols

    def _get_study_topics(self, industry: str) -> List[str]:
        """
        ดึงหัวข้อที่ควรศึกษาเพิ่มเติมตามอุตสาหกรรม
        """
        topics = {
            'ธนาคาร': ['การจัดการความเสี่ยงทางการเงิน', 'กฎระเบียบของธนาคารกลาง'],
            'เทคโนโลยี': ['นวัตกรรมใหม่', 'การแข่งขันในตลาด'],
            # เพิ่มหัวข้อสำหรับอุตสาหกรรมอื่น ๆ ตามต้องการ
        }
        return topics.get(industry, ['ศึกษาภาพรวมของอุตสาหกรรมเพิ่มเติม'])

    # ฟังก์ชันอื่น ๆ คุณควรเพิ่มเข้ามาตามที่มีในโค้ดเดิมของคุณ

# ----------------- สร้างแอป FastAPI -----------------

app = FastAPI()


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # หรือระบุ domain ที่อนุญาต
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# กำหนดเส้นทางไปยังโฟลเดอร์ templates สำหรับ Jinja2 Templates
templates = Jinja2Templates(directory="templates")

# สร้างอินสแตนซ์ของคลาส
api_key = os.getenv('TYPHOON_API_KEY', None)
if not api_key:
    print("หมายเหตุ: คุณไม่ได้ตั้งค่า Typhoon API Key ผลลัพธ์บางส่วนอาจไม่ทำงาน")
analyzer = AdvancedFinancialAnalyzer(api_key=api_key)

# ----------------- Endpoint หลัก /analyze -----------------

@app.post("/analyze")
async def analyze_endpoint(request: Request, file: UploadFile = File(None)):
    """
    Endpoint หลักสำหรับรับคำขอและเรียกใช้ฟังก์ชันตาม action ที่ระบุ
    """
    # ตรวจสอบว่าเป็นคำขอแบบ multipart/form-data หรือไม่
    if request.headers.get('content-type', '').startswith('multipart/form-data'):
        form = await request.form()
        action = form.get('action')
        params = {}
        # ถ้ามีไฟล์ ให้บันทึกไฟล์ชั่วคราว
        if file:
            contents = await file.read()
            temp_file_path = f"temp_{file.filename}"
            with open(temp_file_path, 'wb') as f:
                f.write(contents)
            params['file_path'] = temp_file_path
        # รับพารามิเตอร์อื่น ๆ จากฟอร์ม
        for key in form.keys():
            if key != 'action' and key != 'file':
                params[key] = form.get(key)
    else:
        # รับคำขอแบบ JSON
        data = await request.json()
        action = data.get('action')
        params = data.get('params', {})

    # เรียกใช้ฟังก์ชัน analyze
    try:
        result = analyzer.analyze(action, params)
        # ลบไฟล์ชั่วคราวถ้ามี
        if 'file_path' in params and os.path.exists(params['file_path']):
            os.remove(params['file_path'])
        return {"result": result}
    except Exception as e:
        # ลบไฟล์ชั่วคราวถ้ามี
        if 'file_path' in params and os.path.exists(params['file_path']):
            os.remove(params['file_path'])
        raise HTTPException(status_code=400, detail=str(e))

# ----------------- Endpoint สำหรับหน้าเว็บหลัก -----------------

@app.get("/")
def read_root(request: Request):
    """
    แสดงหน้าเว็บหลักสำหรับการทดสอบการใช้งาน
    """
    return templates.TemplateResponse("index.html", {"request": request})

# ----------------- รันแอปพลิเคชัน -----------------

if __name__ == "__main__":
    # เพิ่ม workers และปรับ timeout
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000,
        workers=40,  # เพิ่มจำนวน workers
        timeout_keep_alive=120,
        reload=True  # ใช้สำหรับ development
    )

    