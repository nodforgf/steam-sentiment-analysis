"""
เทรนโมเดล TF-IDF + Logistic Regression สำหรับเว็บ แล้วบันทึกเป็น model/tfidf_lr.joblib

ใช้ข้อมูลจากโฟลเดอร์ "train 1811" เท่านั้น (ชุดเดียวกับที่ใช้ fine-tune RoBERTa)
- train_set.csv       -> ใช้เทรน
- validation_set.csv  -> ใช้เทรนรอบสุดท้ายด้วย
- test_set.csv        -> ใช้วัดผลอย่างเดียว (ไม่เอาไปเทรน) เทียบกับโมเดล RoBERTa ได้ตรง ๆ
label: 0 = negative, 1 = positive

ทำไมเว็บใช้ TF-IDF ไม่ใช่ RoBERTa:
Render แพ็กเกจฟรีมี RAM 512MB โหลด RoBERTa (ต้องใช้ ~1.5-2GB) ไม่ได้

รัน:  python train_tfidf.py
"""
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "train 1811"
OUT = ROOT / "model" / "tfidf_lr.joblib"
LABELS = {0: "negative", 1: "positive"}


def clean(text):
    text = re.sub(r"\[.*?\]", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def load(name):
    df = pd.read_csv(str(DATA_DIR / name), encoding="utf-8-sig")
    df["x"] = df["text"].map(clean)
    df = df[df["x"] != ""]
    return df["x"].tolist(), df["label"].map(LABELS).tolist()


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                  max_features=50000, sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=3000, C=4)),
    ])


def main():
    X_tr, y_tr = load("train_set.csv")
    X_va, y_va = load("validation_set.csv")
    X_te, y_te = load("test_set.csv")
    print(f"train {len(X_tr)} / validation {len(X_va)} / test {len(X_te)} รีวิว")

    # วัดผลบน test set (เทรนจาก train อย่างเดียว) -> ดูได้ใน Logs ของ Render
    m = build_model().fit(X_tr, y_tr)
    print("ผลบน test set (TF-IDF + Logistic Regression):")
    print(classification_report(y_te, m.predict(X_te), digits=4))

    # โมเดลที่เว็บใช้จริง: เทรนจาก train + validation (ไม่แตะ test)
    model = build_model().fit(X_tr + X_va, y_tr + y_va)
    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(model, OUT)
    print(f"บันทึกโมเดล -> {OUT}")


if __name__ == "__main__":
    main()
