import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from set_smart_analyzer import SetSmartAnalyzer

from services import AdvancedFinancialAnalyzer
from fastapi.middleware.cors import CORSMiddleware


# โหลดตัวแปรสภาพแวดล้อม
load_dotenv()
typhoon_api_key = os.getenv('TYPHOON_API_KEY')
setsmart_api_key = os.getenv('SETSMART_API_KEY')

# Debugging: แสดง API Key ที่โหลดได้
print(f"TYPHOON_API_KEY: {typhoon_api_key}")
print(f"SETSMART_API_KEY: {setsmart_api_key}")

# ตรวจสอบว่า API Key ถูกต้องหรือไม่
if not typhoon_api_key or not setsmart_api_key:
    raise Exception("API Key สำหรับ Typhoon หรือ SETSMART ยังไม่ได้ตั้งค่าในไฟล์ .env")

# สร้าง instance ของ SetSmartAnalyzer และ AdvancedFinancialAnalyzer
smart_analyzer = SetSmartAnalyzer(api_key=setsmart_api_key)
analyzer = AdvancedFinancialAnalyzer(api_key=typhoon_api_key)

# ----------------- กำหนดค่าและสร้างแอป FastAPI -----------------

app = FastAPI()

# เพิ่ม Middleware สำหรับ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ตั้งค่าเส้นทางสำหรับเทมเพลต
templates = Jinja2Templates(directory="templates")

# ----------------- Endpoint สำหรับหน้าเว็บหลัก -----------------

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    แสดงหน้าเว็บหลักสำหรับการทดสอบการใช้งาน
    """
    return templates.TemplateResponse("index.html", {"request": request})

# ----------------- Endpoint สำหรับการวิเคราะห์ (GET และ POST) -----------------

@app.post("/analyze")
async def analyze_endpoint(request: Request, file: UploadFile = File(...)):
    """
    Endpoint สำหรับรับไฟล์ PDF และส่งกลับผลการวิเคราะห์
    """
    # บันทึกไฟล์ชั่วคราว
    contents = await file.read()
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, 'wb') as f:
        f.write(contents)

    # เรียกใช้ฟังก์ชัน generate_investment_report
    try:
        result = analyzer.generate_investment_report(temp_file_path)
        # ลบไฟล์ชั่วคราว
        os.remove(temp_file_path)
        return {"result": result}
    except Exception as e:
        # ลบไฟล์ชั่วคราว
        os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze")
async def analyze_get_endpoint(request: Request, file_url: str):
    """
    Endpoint สำหรับรับ URL ของไฟล์ PDF และส่งกลับผลการวิเคราะห์ (GET)
    """
    # ดาวน์โหลดไฟล์จาก URL
    try:
        response = requests.get(file_url)
        response.raise_for_status()
        temp_file_path = "temp_downloaded.pdf"
        with open(temp_file_path, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ไม่สามารถดาวน์โหลดไฟล์: {e}")

    # เรียกใช้ฟังก์ชัน generate_investment_report
    try:
        result = analyzer.generate_investment_report(temp_file_path)
        # ลบไฟล์ชั่วคราว
        os.remove(temp_file_path)
        return {"result": result}
    except Exception as e:
        # ลบไฟล์ชั่วคราว
        os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- Endpoint สำหรับการแชทกับ Typhoon API -----------------

@app.post("/chat")
async def chat_with_typhoon(request: Request):
    """
    Endpoint สำหรับการแชทกับ Typhoon API
    """
    data = await request.json()
    message = data.get('message')
    if not message:
        raise HTTPException(status_code=400, detail="กรุณาส่งข้อความเพื่อแชท")

    response = analyzer.chat_with_typhoon(message)
    return {"response": response}
    
@app.get("/smart-stocks")
async def get_smart_stocks():
    """
    ดึงรายชื่อหุ้น SET SMART
    """
    try:
        symbols = smart_analyzer.get_smart_symbols()
        return {
            "status": "success",
            "count": len(symbols),
            "symbols": symbols
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze-smart-stocks")
async def analyze_smart_stocks():
    """
    วิเคราะห์หุ้นทั้งหมดใน SET SMART
    """
    try:
        results = smart_analyzer.analyze_smart_stocks()
        return {
            "status": "success",
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- รันแอปพลิเคชัน -----------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000,
        workers=81,  # (2 x 40) + 1
        loop="uvloop",  # ใช้ uvloop สำหรับประสิทธิภาพที่ดีขึ้น
        http="httptools",  # ใช้ httptools สำหรับ HTTP parsing ที่เร็วขึ้น
        limit_concurrency=1000,  # จำกัดจำนวน concurrent connections
        backlog=2048,  # ขนาดคิวการเชื่อมต่อ
        timeout_keep_alive=30,  # timeout สำหรับ keep-alive connections
    )