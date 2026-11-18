"""
เทรนโมเดล TF-IDF + Logistic Regression จากไฟล์ใน Train/ แล้วบันทึกเป็น model/tfidf_lr.joblib

ทำไมใช้ตัวนี้บนเว็บ:
- ในผลเทียบ 3 โมเดล (Train/three_model_comparison.csv) ตัวนี้ได้ macro F1 สูงสุด
- ไฟล์โมเดลเล็กมากและใช้ RAM น้อย จึงรันบน Render แพ็กเกจฟรี (RAM 512MB) ได้
  ส่วนโมเดล Transformer (roberta) ต้องใช้ RAM มากกว่านั้น

รัน:  python train_tfidf.py
(Render รันให้เองตอน build ดูใน render.yaml)
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
TRAIN_FILES = [
    ROOT / "Train" / "reviews_all_games.csv",
    ROOT / "Train" / "web_disco_elysium_the_final_cut.csv",
]
OUT = ROOT / "model" / "tfidf_lr.joblib"


def load_data():
    frames = []
    for f in TRAIN_FILES:
        if f.exists():
            frames.append(pd.read_csv(str(f), encoding="utf-8-sig"))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="review_id")
    text_col = "text_full" if "text_full" in df.columns else "text"
    df["x"] = df[text_col].fillna(df["text"]).astype(str)
    df = df[df["x"].str.strip() != ""]
    # label มาจาก voted_up ของ Steam: 1 = positive, 0 = negative
    df["y"] = df["voted_up"].map({1: "positive", 0: "negative"})
    return df["x"].tolist(), df["y"].tolist()


def main():
    X, y = load_data()
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 1),
                                  min_df=2, max_features=20000, sublinear_tf=True)),
        # class_weight="balanced" เพราะรีวิวลบมีน้อยกว่ารีวิวบวกมาก
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    model.fit(X, y)
    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(model, OUT)
    print(f"เทรนจาก {len(X)} รีวิว "
          f"(positive {y.count('positive')} / negative {y.count('negative')}) -> {OUT}")


if __name__ == "__main__":
    main()
