# Relevance Classification - งานคนที่ 1

## หน้าที่ของโค้ด
เทรนโมเดลคัดกรองรีวิว Steam เป็น 3 คลาส:
- `relevant`: มีความคิดเห็นหรือประสบการณ์เกี่ยวกับเกม
- `no_opinion`: กล่าวถึงเกมแต่ไม่มีความคิดเห็น
- `off_topic`: นอกเรื่องหรือสแปม

คลาส `no_opinion` และ `off_topic` จะเป็นกลุ่มที่คัดออกก่อนส่งไปทำ Sentiment Analysis

## วิธีใช้งาน
1. ติดตั้ง Python packages:
   `pip install -r requirements.txt`
2. เติมข้อมูลที่ติดป้ายกำกับโดยคนลงใน `data/relevance_dataset.csv`
   โดยต้องมีคอลัมน์ `text,label`
3. ใช้ label ตามนี้เท่านั้น: `relevant`, `no_opinion`, `off_topic`
4. รัน:
   `python train_relevance.py`

## ข้อมูลสำคัญ
- ไฟล์ `relevance_dataset_template.csv` เป็นเพียงตัวอย่างรูปแบบข้อมูล ไม่เพียงพอสำหรับ train
- โค้ดกำหนดขั้นต่ำ 30 ตัวอย่างต่อคลาส แนะนำให้มีหลายร้อยตัวอย่างต่อคลาส
- ห้ามใช้ `voted_up` จาก Steam เป็น label ของ relevance เพราะ voted_up คือผลโหวตความรู้สึก ไม่ได้บอกว่าข้อความเกี่ยวข้องหรือไม่
- ใช้ CPU และบันทึกโมเดลไว้ใน `relevance_model/`
- โค้ดนี้ไม่ทำ Sentiment Analysis, Dashboard, Steam API หรือ Hugging Face upload

## อัปโหลดโมเดลไป Hugging Face Hub อัตโนมัติ
หลัง train และประเมินผลสำเร็จ โปรแกรมจะถามข้อมูลต่อไปนี้:
1. Repo ID รูปแบบ `username/model-name` เช่น `myname/steam-review-relevance`
2. Hugging Face User Access Token โดยจะรับแบบซ่อนตัวอักษร ไม่แสดง token บนหน้าจอ
3. เลือก Private หรือ Public (ค่าเริ่มต้นคือ Private)

จากนั้นโปรแกรมจะสร้าง Model repository (ถ้ายังไม่มี) และอัปโหลดโมเดล, tokenizer, ผลประเมิน และ `README.md` ซึ่งเป็นหน้า Model Card ให้อัตโนมัติ

### เตรียมบัญชี Hugging Face
- เข้าสู่ระบบที่ https://huggingface.co/
- สร้าง User Access Token ที่ https://huggingface.co/settings/tokens โดยกำหนดสิทธิ์ Write
- ห้ามใส่ token ลงในไฟล์ Python หรือแชร์ token ให้ผู้อื่น
- ต้องเชื่อมต่ออินเทอร์เน็ตขณะอัปโหลด

สามารถกำหนด environment variables `HF_TOKEN` และ `HF_REPO_ID` ไว้ก่อนรันได้ เพื่อลดการถามข้อมูลในหน้าจอ


หมายเหตุการแก้ไขเวอร์ชันนี้:
- ปรับ pd.read_csv ให้ส่ง path เป็น str เพื่อรองรับ type checker บางรุ่นใน PyCharm
- เรียก upload_to_huggingface() จริงหลังบันทึกโมเดลและผลประเมินแล้ว
- คำว่า roberta เป็นชื่อโมเดล/สถาปัตยกรรม ไม่ใช่คำสะกดผิด
