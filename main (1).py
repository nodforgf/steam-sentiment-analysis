import os
import re
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from steam_reviews import search_games, fetch_top_reviews

app = FastAPI()

# ===== เลือกโมเดล =====
# SENTIMENT_BACKEND=tfidf        -> TF-IDF + Logistic Regression (ค่าเริ่มต้น, ใช้ RAM น้อย รันบน Render ฟรีได้)
# SENTIMENT_BACKEND=transformer  -> pansaporn/steam-sentiment (ต้องใช้ RAM ~2GB ขึ้นไป)
BACKEND = os.getenv("SENTIMENT_BACKEND", "tfidf").lower()
NEUTRAL_THRESHOLD = float(os.getenv("NEUTRAL_THRESHOLD", "0.6"))

if BACKEND == "transformer":
    from transformers import pipeline
    clf = pipeline("text-classification",
                   model=os.getenv("HF_MODEL", "pansaporn/steam-sentiment"), top_k=None)

    def classify(texts):
        results = clf(texts, truncation=True, max_length=128, batch_size=16)
        out = []
        for r in results:
            best = max(r, key=lambda x: x["score"])
            out.append("neutral" if best["score"] < NEUTRAL_THRESHOLD else best["label"])
        return out
else:
    import joblib
    from pathlib import Path

    model_path = Path(__file__).parent / "model" / "tfidf_lr.joblib"
    if not model_path.exists():
        # ยังไม่เคยเทรน -> เทรนเลย (ใช้เวลาไม่กี่วินาที)
        import train_tfidf
        train_tfidf.main()
    clf = joblib.load(model_path)

    def classify(texts):
        probs = clf.predict_proba(texts)
        labels = clf.classes_
        out = []
        for p in probs:
            i = p.argmax()
            out.append("neutral" if p[i] < NEUTRAL_THRESHOLD else str(labels[i]))
        return out

POS_KEYWORDS = ["story", "graphic", "gameplay", "music", "open world", "fun", "content", "worth"]
NEG_KEYWORDS = ["crash", "bug", "hard", "price", "fps", "lag", "boring", "refund"]

# ===== Cache =====
cache = {}

def clean_text(text: str) -> str:
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ใส่ key ใน Environment Variables ของ Render (ห้ามใส่ในโค้ด)
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "")

def _translate_deepl(text: str) -> str:
    import requests as req
    r = req.post(
        "https://api-free.deepl.com/v2/translate",
        headers={"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"},
        json={"text": [text], "target_lang": "TH"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["translations"][0]["text"]


def _translate_google(text: str) -> str:
    # Google Translate แบบไม่ต้องใช้ key (ฟรี ไม่เป็นทางการ เหมาะกับงานเดโม)
    import requests as req
    r = req.get(
        "https://translate.googleapis.com/translate_a/single",
        params={"client": "gtx", "sl": "en", "tl": "th", "dt": "t", "q": text},
        timeout=10,
    )
    r.raise_for_status()
    return "".join(part[0] for part in r.json()[0] if part and part[0])


def translate_to_thai(text: str) -> str:
    cleaned = clean_text(text)[:500]
    # มี DeepL key ใช้ DeepL ก่อน / ไม่มีหรือใช้ไม่ได้ -> Google / พังหมด -> คืนภาษาอังกฤษ
    translators = ([_translate_deepl] if DEEPL_API_KEY else []) + [_translate_google]
    for tr in translators:
        try:
            out = tr(cleaned)
            if out:
                return out
        except Exception:
            continue
    return cleaned

def extract_tags(reviews, sentiment):
    keywords = POS_KEYWORDS if sentiment == "pos" else NEG_KEYWORDS
    text_all = " ".join(clean_text(r["text"]).lower() for r in reviews)
    return [k for k in keywords if k in text_all][:4]

@app.get("/healthz")
def healthz():
    return {"status": "ok", "backend": BACKEND}

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
    sentiments = classify(texts)

    pos = sentiments.count("positive")
    neg = sentiments.count("negative")
    neu = sentiments.count("neutral")
    total = len(sentiments)

    for r, s in zip(reviews, sentiments):
        r["sentiment"] = "pos" if s == "positive" else "neg" if s == "negative" else "neu"

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