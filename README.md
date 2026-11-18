# 🎮 Steam Review Sentiment Analysis

ระบบวิเคราะห์ความรู้สึก (Sentiment Analysis) ของรีวิวเกมจาก Steam โดยอัตโนมัติ

---

## 💡 Idea

เกมบนแพลตฟอร์ม Steam มีรีวิวจากผู้ใช้จำนวนหลักแสนรายการต่อเกม ทำให้นักพัฒนาและผู้บริโภคไม่สามารถอ่านได้ครบถ้วน ระบบนี้จึงถูกพัฒนาขึ้นเพื่อวิเคราะห์และสรุปภาพรวมความคิดเห็นโดยอัตโนมัติ

**กลุ่มเป้าหมาย**
- นักพัฒนาเกม / Publisher — ใช้วิเคราะห์ feedback เพื่อปรับปรุงผลิตภัณฑ์
- Gamer ทั่วไป — ใช้ประกอบการตัดสินใจซื้อเกม

---

## 🚀 How to Run

### Requirements
```
pip install fastapi uvicorn transformers torch requests
```

### Run
```
git clone https://github.com/nodforgf/steam-sentiment-analysis
cd steam-sentiment-analysis
python -m uvicorn main:app --reload
```

เปิด browser ไปที่ `http://localhost:8000`

### How to Use
1. พิมพ์ชื่อเกมในช่องค้นหา เช่น `Elden Ring`
2. กดปุ่ม **วิเคราะห์**
3. รอประมาณ 10-15 วินาที
4. ดูผลลัพธ์ที่แสดง % บวก/ลบ/กลาง พร้อม verdict และรีวิวยอดนิยม

---

## ⚙️ How to Customize

เปลี่ยน model ได้ใน `main.py`
```python
clf = pipeline("text-classification", model="YOUR_MODEL", top_k=None)
NEUTRAL_THRESHOLD = 0.6  # ปรับค่า threshold สำหรับคลาสกลาง
```

เปลี่ยนจำนวนรีวิวที่ดึงได้ใน `steam_reviews.py`
```python
TOP_N = 300  # เพิ่มหรือลดได้ตามต้องการ
```

---

## 📊 Data + Results

**Dataset**
- Steam Review dataset จาก Kaggle
- แบ่ง label จาก voted_up: positive / negative / neutral

**Model**
- Base model: cardiffnlp/twitter-roberta-base-sentiment
- Fine-tuned บน Steam Review dataset

**Results**
| Metric | Score |
|---|---|
| Accuracy | ใส่จาก classification_report.txt |
| F1-score (macro) | ใส่จาก classification_report.txt |

**Baseline Comparison**
| Model | Accuracy |
|---|---|
| Majority Class (baseline) | ใส่จาก classification_report.txt |
| Fine-tuned BERT (ours) | ใส่จาก classification_report.txt |

---

## 👥 Team Members

| ชื่อ | บทบาท |
|---|---|
| [ชื่อคนที่ 1] | Data & Model — เก็บ dataset, train และ evaluate model |
| รัตนาภรณ์ วรรณศรี | Steam API — เขียนโค้ดดึงรีวิวจาก Steam |
| ชนันท์รัช เกตวันดี | Web Application — ออกแบบและพัฒนา UI |
| [ชื่อคนที่ 4] | Deployment — deploy ระบบขึ้น Render |
| ฐิติมา ธาดาบุษบง | Report & Presentation — จัดทำรายงานและสไลด์ |

