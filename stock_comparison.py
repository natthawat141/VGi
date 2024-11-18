# stock_comparison.py
from typing import List, Dict, Optional
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

class StockComparison:
    def __init__(self):
        self.cache = {}  # สำหรับเก็บข้อมูลหุ้นที่ดึงมาแล้ว

    def get_peer_comparison_data(self, 
                               main_symbol: str, 
                               peer_symbols: List[str], 
                               period: str = '1y') -> Dict:
        """
        ดึงข้อมูลราคาหุ้นของบริษัทหลักและบริษัทที่คล้ายกัน
        """
        try:
            # รวมทุก symbol ที่ต้องการดึงข้อมูล
            all_symbols = [main_symbol] + peer_symbols
            comparison_data = {
                'dates': [],
                'prices': {},
                'normalized_prices': {},
                'performance_metrics': {},
                'correlation_matrix': None
            }

            # ดึงข้อมูลทุกหุ้น
            stock_data = {}
            for symbol in all_symbols:
                if symbol in self.cache:
                    data = self.cache[symbol]
                else:
                    stock = yf.Ticker(symbol)
                    data = stock.history(period=period)
                    self.cache[symbol] = data
                stock_data[symbol] = data

            # สร้าง DataFrame รวม
            merged_data = pd.DataFrame()
            for symbol, data in stock_data.items():
                if not data.empty:
                    merged_data[symbol] = data['Close']

            # คำนวณ normalized prices (เริ่มที่ 100)
            normalized_data = merged_data.div(merged_data.iloc[0]) * 100

            # คำนวณ correlation matrix
            correlation_matrix = merged_data.corr()

            # คำนวณ performance metrics
            performance_metrics = {}
            for symbol in all_symbols:
                if symbol in merged_data.columns:
                    prices = merged_data[symbol]
                    returns = prices.pct_change().dropna()
                    performance_metrics[symbol] = {
                        'total_return': ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100,
                        'volatility': returns.std() * np.sqrt(252) * 100,
                        'max_drawdown': ((prices / prices.cummax()) - 1).min() * 100
                    }

            # เตรียมข้อมูลสำหรับส่งกลับ
            comparison_data['dates'] = merged_data.index.strftime('%Y-%m-%d').tolist()
            comparison_data['prices'] = merged_data.to_dict('dict')
            comparison_data['normalized_prices'] = normalized_data.to_dict('dict')
            comparison_data['performance_metrics'] = performance_metrics
            comparison_data['correlation_matrix'] = correlation_matrix.to_dict('dict')

            return comparison_data

        except Exception as e:
            print(f"Error in get_peer_comparison_data: {e}")
            return None