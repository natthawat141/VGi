# market_data.py
from together import Together
import yfinance as yf
from typing import Optional
from models import FinancialData
import numpy as np

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
