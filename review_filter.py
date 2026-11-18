"""
ไฟล์เสริม: กรองรีวิวที่ไม่อยากให้แสดงในหน้าผลการค้นหา (รีวิวยอดนิยม 5 อัน)

- กรองเฉพาะ "รีวิวที่แสดง" เท่านั้น ไม่กระทบการคำนวณ % บวก/ลบ/กลาง
- ถ้าไม่อยากใช้: ลบไฟล์นี้ทิ้งได้เลย เว็บจะกลับไปแสดงเหมือนเดิมทุกอย่าง
- อยากกรองคำเพิ่ม: เติมใน BLOCKED_WORDS (ตัวพิมพ์เล็ก)
"""
import re

# ===== คำที่ไม่ให้แสดง (เติม/ลบได้ตามต้องการ) =====
# เช็กแบบ "ทั้งคำ" เช่นใส่ "gay" จะจับ "gay" แต่ไม่จับคำอื่นที่มีตัวอักษรนี้อยู่ข้างใน
BLOCKED_WORDS = [
    "gay", "retard", "retarded", "fuck", "fucking", "fucked", "shit", "bitch",
    "cunt", "whore", "slut", "porn", "sex", "nazi", "faggot", "fag",
    "nigga", "nigger",
]

# ===== ตั้งค่ากรองมีม/สแปม =====
MIN_CHARS = 30            # สั้นกว่านี้ไม่แสดง (มักเป็นมุกสั้น ๆ)
MIN_ALPHA_RATIO = 0.6     # ตัวอักษรน้อยกว่านี้ = ASCII art / สัญลักษณ์
REPEAT_CHAR = re.compile(r"(\S)\1{9,}")          # อักขระเดียวซ้ำ 10 ตัวขึ้นไป เช่น aaaaaaaaaa
ART_CHARS = re.compile(r"[─-▟⠀-⣿]")  # ตัวอักษรวาดรูป/อักษรเบรลล์

_BLOCKED = re.compile(r"\b(" + "|".join(re.escape(w) for w in BLOCKED_WORDS) + r")\b",
                      re.IGNORECASE)


def reject_reason(text):
    """คืนเหตุผลที่ไม่ให้แสดง หรือ None ถ้าแสดงได้"""
    t = str(text).strip()
    if _BLOCKED.search(t):
        return "มีคำไม่เหมาะสม"
    if len(t) < MIN_CHARS:
        return "สั้นเกินไป"
    if ART_CHARS.search(t) or REPEAT_CHAR.search(t):
        return "ASCII art / พิมพ์ซ้ำ"
    no_space = re.sub(r"\s", "", t)
    if no_space and sum(c.isalpha() for c in no_space) / len(no_space) < MIN_ALPHA_RATIO:
        return "สัญลักษณ์เยอะเกินไป"
    return None


def is_clean(text):
    return reject_reason(text) is None
