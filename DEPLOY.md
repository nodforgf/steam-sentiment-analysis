# Deploy ขึ้น Render (ให้คนอื่นเปิดดูได้โดยไม่ต้องรันเครื่องตัวเอง)

## 1. ขึ้นไฟล์ทั้งหมดไป GitHub
ไฟล์ที่ต้องมีใน repo: `main.py`, `steam_reviews.py`, `train_tfidf.py`, `requirements.txt`,
`render.yaml`, `static/index.html` และโฟลเดอร์ `train 1811/` (ใช้เทรนโมเดลตอน build)

```
git add .
git commit -m "Prepare for Render deploy"
git push
```

## 2. สร้าง service บน Render
1. เข้า https://dashboard.render.com แล้ว login ด้วย GitHub
2. กด **New +** → **Blueprint** → เลือก repo `steam-sentiment-analysis`
3. Render จะอ่าน `render.yaml` ให้เอง → ช่อง `DEEPL_API_KEY` ให้ใส่ key ของ DeepL (ไม่ใส่ก็ได้ แต่รีวิวจะไม่ถูกแปลไทย)
4. กด **Apply** แล้วรอ build ~3–5 นาที

(ถ้าไม่ใช้ Blueprint: New + → Web Service → เลือก repo แล้วตั้งค่าเอง)
- Runtime: Python 3
- Build Command: `pip install -r requirements.txt && python train_tfidf.py`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Instance Type: Free
- Environment: `PYTHON_VERSION=3.12.7`, `DEEPL_API_KEY=...`

## 3. ได้ลิงก์
ลิงก์จะเป็น `https://steam-sentiment-xxxx.onrender.com` ส่งให้ใครเปิดก็ได้
ทุกครั้งที่ push ขึ้น GitHub ตัวเว็บจะ deploy ใหม่ให้อัตโนมัติ

## หมายเหตุ
- แพ็กเกจฟรีจะ "หลับ" ถ้าไม่มีคนเข้า 15 นาที คนแรกที่เข้าจะรอ ~1 นาทีให้ตื่น
- แพ็กเกจฟรีมี RAM 512MB จึงใช้โมเดล TF-IDF + Logistic Regression รวมกับ VADER (ค่าเริ่มต้น) เทรนจาก `train 1811`
- ถ้าจะใช้โมเดล Transformer `pansaporn/steam-sentiment` ต้องเปลี่ยนเป็นแพ็กเกจที่มี RAM 2GB ขึ้นไป แล้วตั้ง
  - Build Command: `pip install -r requirements-transformer.txt`
  - Environment: `SENTIMENT_BACKEND=transformer`
