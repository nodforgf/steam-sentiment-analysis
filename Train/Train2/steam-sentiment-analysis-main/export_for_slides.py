"""
Export 3 ไฟล์สำหรับใส่สไลด์:
  1. classification_report_new.txt
  2. confusion_matrix_new.png
  3. three_model_comparison.csv

เทียบ 3 โมเดลบน test set ชุดเดียวกัน:
  - TF-IDF + Logistic Regression   (baseline)
  - TF-IDF + Naive Bayes           (baseline)
  - pansaporn/steam-sentiment      (โมเดลที่เว็บใช้จริงใน main.py)

label จริง = voted_up จาก Steam (1 = Recommended, 0 = Not Recommended)

วิธีใช้:
  pip install pandas scikit-learn matplotlib transformers torch
  python export_for_slides.py   (อ่าน ../train/reviews_all_games.csv แล้วเซฟผลลง ../train/)

ถ้ามี test_set แยกไว้แล้ว:
  python export_for_slides.py --train train_set.csv --test test_set.csv
"""
import argparse
import os
import re
import time

TRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

p = argparse.ArgumentParser()
p.add_argument("--input", default=os.path.join(TRAIN_DIR, "reviews_all_games.csv"), help="CSV เดียว (จาก steam_reviews.py) จะแบ่ง train/test ให้เอง 80/20")
p.add_argument("--train", help="train_set.csv (ถ้าแบ่งไว้แล้ว)")
p.add_argument("--test", help="test_set.csv (ถ้าแบ่งไว้แล้ว)")
p.add_argument("--text-col", default="text")
p.add_argument("--label-col", default="voted_up")
p.add_argument("--hf-model", default="pansaporn/steam-sentiment",
               help="โมเดล HuggingFace ของทีม ใส่ none ถ้าจะใช้ LinearSVM แทน")
p.add_argument("--out", default=TRAIN_DIR)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
out = lambda name: os.path.join(args.out, name)
LABELS = ["negative", "positive"]


def clean_text(text):  # ตัวเดียวกับใน main.py เพื่อให้ผลตรงกับเว็บ
    text = re.sub(r"\[.*?\]", "", str(text))
    return re.sub(r"\s+", " ", text).strip()


def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[[args.text_col, args.label_col]].dropna()
    df.columns = ["text", "label"]
    df["text"] = df["text"].map(clean_text)
    df = df[df.text.str.len() > 0].drop_duplicates(subset="text")
    df["label"] = df.label.map(lambda v: 1 if str(v).strip().lower() in ("1", "true", "positive") else 0)
    return df.reset_index(drop=True)


# ---------- 1. ข้อมูล ----------
if args.train and args.test:  # ถ้าใส่ทั้งคู่ จะใช้แทน --input
    train_df, test_df = load(args.train), load(args.test)
    test_df = test_df[~test_df.text.isin(set(train_df.text))]  # กันรีวิวซ้ำข้าม train/test
elif args.input:
    df = load(args.input)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.label, random_state=args.seed)
else:
    p.error("ต้องใส่ --input หรือ --train กับ --test")
print(f"train {len(train_df)} | test {len(test_df)} "
      f"(positive {test_df.label.sum()} / negative {(test_df.label == 0).sum()})")


# ---------- 2. โมเดล ----------
def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)


def sklearn_model(clf):
    def run(texts):
        m = Pipeline([("tfidf", tfidf()), ("clf", clf)])
        m.fit(train_df.text, train_df.label)
        return m.predict(texts)
    return run


def label_to_int(label):
    """แปลง label ของโมเดล HF เป็น 0/1 (รองรับทั้ง positive/negative และ LABEL_0/LABEL_1)"""
    l = label.lower()
    if "pos" in l or l == "label_1":
        return 1
    if "neg" in l or l == "label_0":
        return 0
    raise ValueError(f"ไม่รู้จัก label '{label}' ของโมเดล ต้องแก้ label_to_int")


def hf_model(name):
    def run(texts):
        from transformers import pipeline
        clf = pipeline("text-classification", model=name, top_k=None)
        res = clf(list(texts), truncation=True, max_length=128, batch_size=64)
        # ใช้ label ที่คะแนนสูงสุด (ไม่ใช้ neutral เพราะ label จริงจาก Steam มีแค่บวก/ลบ)
        return np.array([label_to_int(max(r, key=lambda x: x["score"])["label"]) for r in res])
    return run


models = {
    "TF-IDF + Logistic Regression": sklearn_model(LogisticRegression(max_iter=2000, class_weight="balanced")),
    "TF-IDF + Naive Bayes": sklearn_model(MultinomialNB()),
}
if args.hf_model.lower() == "none":
    models["TF-IDF + Linear SVM"] = sklearn_model(LinearSVC(class_weight="balanced"))
else:
    models[f"Fine-tuned Transformer ({args.hf_model})"] = hf_model(args.hf_model)

# ---------- 3. ประเมินทุกโมเดลบน test set เดียวกัน ----------
rows, preds = [], {}
for name, run in models.items():
    t0 = time.time()
    pred = run(test_df.text)
    secs = time.time() - t0
    preds[name] = pred
    y = test_df.label
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1_negative": f1_score(y, pred, pos_label=0, zero_division=0),
        "time_sec": secs,
    })
    print(f"{name}: f1_macro = {rows[-1]['f1_macro']:.4f}  ({secs:.1f}s)")

comparison = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).round(4)
comparison.to_csv(out("three_model_comparison.csv"), index=False, encoding="utf-8-sig")
best = comparison.iloc[0]["model"]

# ---------- 4. classification report (ทุกโมเดล, โมเดลที่ดีที่สุดอยู่บนสุด) ----------
with open(out("classification_report_new.txt"), "w", encoding="utf-8") as f:
    f.write(f"Test set: {len(test_df)} reviews | label = Steam voted_up\n")
    f.write(f"Best model (macro F1): {best}\n")
    for name in comparison.model:
        f.write(f"\n{'=' * 60}\n{name}\n{'=' * 60}\n")
        f.write(classification_report(test_df.label, preds[name], target_names=LABELS, digits=4))

# ---------- 5. confusion matrix ของโมเดลที่ดีที่สุด (ขนาดพอดีสไลด์) ----------
plt.rcParams.update({"font.size": 16})
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(test_df.label, preds[best])
ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
ax.set_title(best if len(best) < 40 else best.split(" (")[0], fontsize=16)
fig.tight_layout()
fig.savefig(out("confusion_matrix_new.png"), dpi=200)
plt.close(fig)

print(f"\nโมเดลที่ดีที่สุด: {best}")
print(f"ได้ 3 ไฟล์ใน {os.path.abspath(args.out)} แล้ว")
