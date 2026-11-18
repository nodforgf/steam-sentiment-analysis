"""
เทรนโมเดล TF-IDF + Logistic Regression แล้วบันทึกเป็น model/tfidf_lr.joblib

ข้อมูลที่ใช้เทรน:
1. ไฟล์ใน Train/Train2/train ของกลุ่ม (รีวิวบวกเยอะมาก รีวิวลบน้อย)
2. ดึงรีวิวเพิ่มจาก Steam ตอน build แบบ "บวก/ลบ เท่ากัน" หลายเกม
   เพื่อให้โมเดลรู้จักรีวิวลบมากขึ้น (label = voted_up ของ Steam)
   ถ้าดึงจาก Steam ไม่ได้ จะใช้แค่ไฟล์ใน Train/Train2/train แทน

ทำไมใช้โมเดลนี้บนเว็บ: ไฟล์เล็ก ใช้ RAM น้อย รันบน Render แพ็กเกจฟรี (512MB) ได้

รัน:  python train_tfidf.py
ปิดการดึงจาก Steam:  ตั้ง env TRAIN_FETCH_STEAM=0
"""
import os
import re
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
# ใช้ข้อมูลจาก Train/Train2/train (เวอร์ชันล่าสุดของกลุ่ม)
# = reviews_all_games.csv + ไฟล์ web_*.csv ทุกเกมที่เคยวิเคราะห์บนเว็บ
TRAIN_DIR = ROOT / "Train" / "Train2" / "train"
TRAIN_FILES = [TRAIN_DIR / "reviews_all_games.csv"] + sorted(TRAIN_DIR.glob("web_*.csv"))
if not TRAIN_DIR.exists():  # เผื่อมีคนย้าย/ลบ Train2 -> ใช้ Train/ เดิมแทน
    TRAIN_FILES = [ROOT / "Train" / "reviews_all_games.csv"] + sorted((ROOT / "Train").glob("web_*.csv"))
OUT = ROOT / "model" / "tfidf_lr.joblib"

# เกมที่มีทั้งรีวิวบวกและลบเยอะ (appid ของ Steam)
STEAM_APPIDS = [730, 570, 578080, 1091500, 275850, 413150, 1245620, 1086940,
                105600, 1517290, 1151340, 1716740, 1272080, 1145350, 632470]
PER_CLASS_PER_GAME = int(os.getenv("TRAIN_PER_CLASS", "200"))


def clean(text):
    text = re.sub(r"\[.*?\]", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def load_csv_data():
    frames = [pd.read_csv(str(f), encoding="utf-8-sig") for f in TRAIN_FILES if f.exists()]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="review_id")
    text_col = "text_full" if "text_full" in df.columns else "text"
    df["x"] = df[text_col].fillna(df["text"])
    df["y"] = df["voted_up"].map({1: "positive", 0: "negative"})
    return df[["review_id", "x", "y"]]


APPREVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"


def _get_page(appid, review_type, cursor):
    import requests
    for attempt in range(3):
        try:
            r = requests.get(APPREVIEWS_URL.format(appid=appid), params={
                "json": 1, "filter": "recent", "language": "english",
                "review_type": review_type,      # positive / negative
                "purchase_type": "all", "num_per_page": 100, "cursor": cursor,
            }, timeout=20)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(3)
    return None


def fetch_steam_balanced():
    """ดึงรีวิวบวกและลบอย่างละเท่า ๆ กันจากหลายเกม (label ใช้ voted_up ที่ Steam ส่งมาจริง)"""
    rows = []
    for appid in STEAM_APPIDS:
        for review_type in ["positive", "negative"]:
            cursor, got = "*", 0
            while got < PER_CLASS_PER_GAME:
                data = _get_page(appid, review_type, cursor)
                if not data or not data.get("reviews"):
                    break
                for rv in data["reviews"]:
                    text = clean(rv.get("review", ""))
                    if len(text) < 20:          # สั้นเกินไป มักเป็นมุก/อีโมจิ
                        continue
                    rows.append({"review_id": rv["recommendationid"], "x": text[:3000],
                                 "y": "positive" if rv.get("voted_up") else "negative"})
                    got += 1
                next_cursor = data.get("cursor")
                if not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor
                time.sleep(0.3)
        print(f"  appid {appid}: รวมแล้ว {len(rows)} รีวิว")
    return pd.DataFrame(rows, columns=["review_id", "x", "y"])


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                  max_features=50000, sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=3000, C=4, class_weight="balanced")),
    ])


def main():
    df = load_csv_data()
    if os.getenv("TRAIN_FETCH_STEAM", "1") != "0":
        print("กำลังดึงรีวิวบวก/ลบจาก Steam เพื่อใช้เทรน...")
        try:
            extra = fetch_steam_balanced()
        except Exception as e:  # เน็ตหรือ Steam มีปัญหา -> ใช้แค่ไฟล์ใน Train/
            print(f"  ดึงจาก Steam ไม่ได้ ({e.__class__.__name__}) ใช้แค่ไฟล์ใน Train/Train2/train")
            extra = pd.DataFrame(columns=["review_id", "x", "y"])
        df = pd.concat([df, extra], ignore_index=True)

    df["review_id"] = df["review_id"].astype(str)
    df = df.drop_duplicates(subset="review_id")
    df["x"] = df["x"].map(clean)
    df = df[df["x"] != ""]
    X, y = df["x"].tolist(), df["y"].tolist()
    print(f"ข้อมูลเทรน {len(X)} รีวิว (positive {y.count('positive')} / negative {y.count('negative')})")

    # วัดผลคร่าว ๆ ก่อน (แบ่ง 20% ไว้ทดสอบ) ดูได้ใน Logs ของ Render
    if y.count("negative") >= 20:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        m = build_model().fit(Xtr, ytr)
        print(classification_report(yte, m.predict(Xte), digits=3))

    model = build_model().fit(X, y)
    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(model, OUT)
    print(f"บันทึกโมเดล -> {OUT}")


if __name__ == "__main__":
    main()
