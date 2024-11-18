# main.py

import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any
import uvicorn

from services import AdvancedFinancialAnalyzer

# ----------------- กำหนดค่าและสร้างแอป FastAPI -----------------

app = FastAPI()

# ตั้งค่าเส้นทางสำหรับเทมเพลตและไฟล์สถิต
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# สร้างอินสแตนซ์ของคลาส Analyzer โดยใช้ API Key จากตัวแปรสภาพแวดล้อม
api_key = os.getenv('TYPHOON_API_KEY')
if not api_key:
    print("หมายเหตุ: คุณไม่ได้ตั้งค่า Typhoon API Key ผลลัพธ์บางส่วนอาจไม่ทำงาน")
analyzer = AdvancedFinancialAnalyzer(api_key=api_key)

# ----------------- Endpoint สำหรับหน้าเว็บหลัก -----------------

@app.get("/")
async def read_root(request: Request):
    """
    แสดงหน้าเว็บหลักสำหรับการทดสอบการใช้งาน
    """
    return templates.TemplateResponse("index.html", {"request": request})
    
@app.get("/test/hello")
def test_hello():
    return {"message": "Hello, World!"}

# ----------------- Endpoint หลักสำหรับการวิเคราะห์ -----------------

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

# ----------------- รันแอปพลิเคชัน -----------------

if __name__ == "__main__":
    # รันแอปด้วย Uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
