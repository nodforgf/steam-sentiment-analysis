"""
เทรนโมเดล TF-IDF + Logistic Regression สำหรับเว็บ แล้วบันทึกเป็น model/tfidf_lr.joblib

ใช้ข้อมูลจากโฟลเดอร์ "train 1811" เท่านั้น (ชุดเดียวกับที่ใช้ fine-tune RoBERTa)
- train_set.csv       -> ใช้เทรน
- validation_set.csv  -> ใช้เทรนรอบสุดท้ายด้วย
- test_set.csv        -> ใช้วัดผลอย่างเดียว (ไม่เอาไปเทรน) เทียบกับโมเดล RoBERTa ได้ตรง ๆ
label: 0 = negative, 1 = positive

เว็บใช้ผลเฉลี่ยของ TF-IDF+LR กับ VADER (ดูฟังก์ชัน ensemble_pos_prob)

ทำไมเว็บใช้ TF-IDF ไม่ใช่ RoBERTa:
Render แพ็กเกจฟรีมี RAM 512MB โหลด RoBERTa (ต้องใช้ ~1.5-2GB) ไม่ได้

รัน:  python train_tfidf.py
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion, Pipeline

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
    features = FeatureUnion([
        # คำ 1-2 คำติดกัน
        ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                 max_features=50000, sublinear_tf=True)),
        # ตัวอักษร 2-5 ตัว (ช่วยกับคำสะกดผิด/คำแสลง)
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 max_features=100000, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", features), ("lr", LogisticRegression(max_iter=3000, C=10))])


# ===== ใช้ร่วมกับ VADER (พจนานุกรมคำบวก/ลบภาษาอังกฤษทั่วไป) =====
# ข้อมูลเทรนมีแค่ 3 เกม โมเดล TF-IDF อย่างเดียวจึงเดาเกมอื่นพลาดง่าย
# เลยเฉลี่ยความน่าจะเป็นกับ VADER ครึ่งต่อครึ่ง ให้ใช้กับเกมอื่นได้ดีขึ้น
LR_WEIGHT = 0.5
_vader = None


def vader_pos_prob(texts):
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return np.array([(_vader.polarity_scores(t[:3000])["compound"] + 1) / 2 for t in texts])


def ensemble_pos_prob(model, texts):
    """ความน่าจะเป็นที่เป็นบวก (0-1) = เฉลี่ย TF-IDF+LR กับ VADER"""
    pos_idx = list(model.classes_).index("positive")
    p_lr = model.predict_proba(texts)[:, pos_idx]
    return LR_WEIGHT * p_lr + (1 - LR_WEIGHT) * vader_pos_prob(texts)


def main():
    X_tr, y_tr = load("train_set.csv")
    X_va, y_va = load("validation_set.csv")
    X_te, y_te = load("test_set.csv")
    print(f"train {len(X_tr)} / validation {len(X_va)} / test {len(X_te)} รีวิว")

    # วัดผลบน test set (เทรนจาก train อย่างเดียว) -> ดูได้ใน Logs ของ Render
    m = build_model().fit(X_tr, y_tr)
    print("ผลบน test set (TF-IDF + Logistic Regression อย่างเดียว):")
    print(classification_report(y_te, m.predict(X_te), digits=4))
    pred = ["positive" if p >= 0.5 else "negative" for p in ensemble_pos_prob(m, X_te)]
    print("ผลบน test set (TF-IDF+LR รวมกับ VADER = ที่เว็บใช้จริง):")
    print(classification_report(y_te, pred, digits=4))

    # โมเดลที่เว็บใช้จริง: เทรนจาก train + validation (ไม่แตะ test)
    model = build_model().fit(X_tr + X_va, y_tr + y_va)
    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(model, OUT)
    print(f"บันทึกโมเดล -> {OUT}")


if __name__ == "__main__":
    main()
