import csv
import os
import re
import time
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from steam_reviews import search_games, fetch_top_reviews, safe_filename, FIELDS

# ทุกเกมที่วิเคราะห์บนเว็บ จะเซฟรีวิว + ผลโมเดลลงโฟลเดอร์ train (อยู่ข้างนอกโฟลเดอร์โปรเจค)
TRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train")


def save_web_result(reviews, game_name):
    """เซฟรีวิวที่เว็บวิเคราะห์ลง train/web_<ชื่อเกม>.csv (มีคอลัมน์ sentiment = ผลของโมเดล)"""
    try:
        os.makedirs(TRAIN_DIR, exist_ok=True)
        path = os.path.join(TRAIN_DIR, f"web_{safe_filename(game_name)}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS + ["sentiment"], extrasaction="ignore")
            w.writeheader()
            w.writerows(reviews)
        print(f"เซฟผลวิเคราะห์ {game_name} -> {os.path.abspath(path)}")
    except Exception as e:  # เซฟไม่ได้ก็ไม่ให้เว็บพัง
        print(f"เซฟไฟล์ไม่ได้: {e}")

app = FastAPI()

from transformers import pipeline

clf = pipeline("text-classification", model="pansaporn/steam-sentiment", top_k=None)
NEUTRAL_THRESHOLD = 0.6

def mock_classify(texts):
    results = clf(texts, truncation=True, max_length=128, batch_size=64)
    return [
        "neutral" if max(r, key=lambda x: x["score"])["score"] < NEUTRAL_THRESHOLD
        else max(r, key=lambda x: x["score"])["label"]
        for r in results
    ]

POS_KEYWORDS = ["story", "graphic", "gameplay", "music", "open world", "fun", "content", "worth"]
NEG_KEYWORDS = ["crash", "bug", "hard", "price", "fps", "lag", "boring", "refund"]

# ===== Cache =====
cache = {}

def clean_text(text: str) -> str:
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

DEEPL_API_KEY = "d5763e2c-4274-479d-ab86-038821c4fca8:fx"

def translate_to_thai(text: str) -> str:
    try:
        cleaned = clean_text(text)[:500]
        import requests as req
        r = req.post(
            "https://api-free.deepl.com/v2/translate",
            headers={"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"},
            json={"text": [cleaned], "target_lang": "TH"},
            timeout=10
        )
        return r.json()["translations"][0]["text"]
    except Exception:
        return clean_text(text)

def extract_tags(reviews, sentiment):
    keywords = POS_KEYWORDS if sentiment == "pos" else NEG_KEYWORDS
    text_all = " ".join(clean_text(r["text"]).lower() for r in reviews)
    return [k for k in keywords if k in text_all][:4]

@app.get("/search")
def search(q: str):
    games = search_games(q, limit=6)
    return {"games": games}

@app.get("/analyze")
def analyze(game: str):
    # เช็ค cache ก่อน ถ้าเคยค้นแล้วคืนผลเก่าเลย
    cache_key = game.strip().lower()
    if cache_key in cache:
        return cache[cache_key]

    # ค้นหาเกม
    try:
        games = search_games(game, limit=1)
    except Exception:
        return JSONResponse(status_code=503, content={"error": "ค้นหาเกมไม่ได้ตอนนี้ ลองใหม่อีกครั้ง"})

    if not games:
        return JSONResponse(status_code=404, content={"error": f"ไม่พบเกม '{game}' บน Steam"})

    app_name = games[0]["name"]
    appid = games[0]["appid"]

    # ดึงรีวิว
    try:
        reviews = fetch_top_reviews(appid, app_name, top_n=300)
    except Exception:
        return JSONResponse(status_code=503, content={"error": "Steam API ไม่ตอบสนอง ลองใหม่อีกครั้ง"})

    if not reviews:
        return JSONResponse(status_code=503, content={"error": "ดึงรีวิวไม่ได้ เกมนี้อาจมีรีวิวน้อยเกินไป"})

    # วิเคราะห์ sentiment
    texts = [clean_text(r["text"])[:512] for r in reviews]
    sentiments = mock_classify(texts)

    pos = sentiments.count("positive")
    neg = sentiments.count("negative")
    neu = sentiments.count("neutral")
    total = len(sentiments)

    for r, s in zip(reviews, sentiments):
        r["sentiment"] = "pos" if s == "positive" else "neg" if s == "negative" else "neu"

    # เซฟรีวิวทั้งหมด + ผลโมเดลลงโฟลเดอร์ train
    save_web_result(reviews, app_name)

    # แปลเฉพาะ top 5
    top5 = reviews[:5]
    for r in top5:
        r["text_clean"] = clean_text(r["text"])
        r["text_th"] = translate_to_thai(r["text"])

    result = {
        "game_name": app_name,
        "total": total,
        "pos": round(pos / total * 100),
        "neg": round(neg / total * 100),
        "neu": 100 - round(pos / total * 100) - round(neg / total * 100),
        "pos_tags": extract_tags(reviews, "pos"),
        "neg_tags": extract_tags(reviews, "neg"),
        "reviews": [
            {
                "text": r["text_clean"][:200],
                "text_th": r.get("text_th", r["text_clean"][:200]),
                "votes_up": r["votes_helpful"],
                "sentiment": r["sentiment"]
            }
            for r in top5
        ]
    }

    # เก็บ cache ไว้
    cache[cache_key] = result
    return result

app.mount("/", StaticFiles(directory="static", html=True), name="static")