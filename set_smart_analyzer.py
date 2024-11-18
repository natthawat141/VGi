# set_smart_analyzer.py

import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import yfinance as yf
import requests

class SetSmartAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        
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

    def get_smart_symbols(self) -> List[str]:
        """
        ดึงรายชื่อหุ้นใน SET SMART
        """
        print("\nกำลังดึงรายชื่อหุ้นใน SET SMART...")
        
        if not self.api_key:
            print("ไม่สามารถดึงข้อมูลได้เนื่องจากไม่ได้ตั้งค่า Typhoon API Key")
            return []

        prompt = """กรุณาระบุรายชื่อหุ้นที่อยู่ใน SET SMART ล่าสุด โดยแสดงเป็นรูปแบบ:
        ชื่อย่อหลักทรัพย์ (.BK) | ชื่อบริษัท
        เช่น
        AOT.BK | บริษัท ท่าอากาศยานไทย จำกัด (มหาชน)
        PTT.BK | บริษัท ปตท. จำกัด (มหาชน)
        """

        response = self._call_typhoon_api(prompt)
        if response:
            content = response['choices'][0]['message']['content']
            symbols = self._extract_symbols(content)
            valid_symbols = self._validate_symbols(symbols)
            
            print(f"\nพบหุ้น SET SMART จำนวน {len(valid_symbols)} ตัว")
            return valid_symbols
        return []

    def _extract_symbols(self, content: str) -> List[str]:
        """
        แยกรายชื่อหุ้นจากข้อความ
        """
        lines = content.strip().split('\n')
        symbols = []
        
        for line in lines:
            match = re.search(r'([A-Z]+\.BK)', line)
            if match:
                symbols.append(match.group(1))
        
        return symbols

    def _validate_symbols(self, symbols: List[str]) -> List[str]:
        """
        ตรวจสอบความถูกต้องของ symbol
        """
        valid_symbols = []
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                if info and 'regularMarketPrice' in info:
                    valid_symbols.append(symbol)
            except:
                continue
        return valid_symbols

    def analyze_smart_stocks(self) -> Dict[str, Any]:
        """
        วิเคราะห์หุ้นทั้งหมดใน SET SMART
        """
        smart_symbols = self.get_smart_symbols()
        results = {}
        
        for symbol in smart_symbols:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="1y")
                info = stock.info
                
                analysis = {
                    'symbol': symbol,
                    'company_name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'market_cap': info.get('marketCap', 0),
                    'current_price': info.get('regularMarketPrice', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'pb_ratio': info.get('priceToBook', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'fifty_two_week': {
                        'high': info.get('fiftyTwoWeekHigh', 0),
                        'low': info.get('fiftyTwoWeekLow', 0)
                    },
                    'volume': info.get('volume', 0),
                }
                
                if not hist.empty:
                    analysis['price_history'] = {
                        'dates': hist.index.strftime('%Y-%m-%d').tolist(),
                        'prices': hist['Close'].tolist(),
                        'volumes': hist['Volume'].tolist()
                    }
                
                results[symbol] = analysis
                
            except Exception as e:
                print(f"ไม่สามารถวิเคราะห์ {symbol}: {e}")
                continue
                
        return {
            'timestamp': datetime.now().isoformat(),
            'total_stocks': len(results),
            'analyses': results
        }

        