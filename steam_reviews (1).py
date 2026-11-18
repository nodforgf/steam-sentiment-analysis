"""
ค้นหาเกมจากชื่อ -> ดึงรีวิวภาษาอังกฤษที่ยอดโหวต helpful สูงสุด (สูงสุด 300 รีวิวต่อครั้ง)
บวก/ลบรวมกัน ใครไลค์เยอะก็ติด โดยกรองมีม / สแปม / เรื่องส่วนตัว / ไม่เกี่ยวกับเกมออกก่อน -> CSV

ใช้แบบถาม-ตอบ:   python steam_reviews.py
ใส่ชื่อเกมเลย:    python steam_reviews.py "stardew valley"
หลายเกม:         python steam_reviews.py "stardew valley, elden ring, hades"

เรียกจากโค้ดส่วนอื่นของทีม (pipeline):
    from steam_reviews import search_games, fetch_top_reviews, save_csv
    games = search_games("hades")
    rejected = []                      # (ไม่ใส่ก็ได้) เก็บรีวิวที่ถูกกรองทิ้ง
    rows = fetch_top_reviews(games[0]["appid"], games[0]["name"], rejected=rejected)  # บวก+ลบ
    save_csv(rows, "hades.csv")
"""
import csv
import json
import os
import re
import sys
import time
from collections import Counter

import requests

SEARCH_URL = "https://store.steampowered.com/api/storesearch/"
REVIEWS_URL = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"
LANGUAGE = "english"   # โปรเจคนี้เอาเฉพาะภาษาอังกฤษ
MAX_REVIEWS = 300      # เพดานจำนวนรีวิวต่อครั้ง (ต่อให้ขอมากกว่านี้ก็ไม่เกิน)
TOP_N = 300            # จำนวนรีวิวที่ต้องการต่อเกม (ค่าเริ่มต้น)
MODE = "mixed"         # "mixed" = เอา TOP_N อันที่ไลค์สูงสุดรวมกันเลย (เร็วสุด, ไม่สนบวก/ลบ)
                       # "proportional" = แบ่งบวก/ลบตามสัดส่วนจริงของ Steam
                       # "balanced" = แบ่งบวก/ลบอย่างละครึ่ง
MIN_PER_SIDE = 10      # อย่างน้อยฝั่งละกี่รีวิว (กันฝั่งลบเหลือ 0 ในเกมที่คนชมเกือบหมด)
SUMMARY_URL = "https://store.steampowered.com/appreviews/{appid}"
SUMMARY_LANGUAGE = "all"   # "all" = ตรงกับตัวเลขบนหน้าร้าน Steam, "english" = เฉพาะรีวิวอังกฤษ
MAX_SCAN = 300        # ดึงดิบจาก Steam สูงสุดกี่รีวิวต่อครั้ง (เผื่อโดนกรองทิ้ง) ยิ่งน้อยยิ่งเร็ว
# โฟลเดอร์เก็บไฟล์ CSV = โฟลเดอร์ train ที่อยู่ข้างนอกโฟลเดอร์โปรเจค
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train")
FIELDS = ["rank", "appid", "game_name", "review_id", "text", "voted_up",
          "votes_helpful", "playtime_hours", "timestamp", "text_full"]

# ---------- ค่าตั้งของตัวกรองมีม ----------
ENABLE_FILTER = True
TRIM_TO_GAME_PARTS = True   # ตัดประโยคที่เล่าเรื่องส่วนตัว/ไม่เกี่ยวกับเกมออก เก็บเฉพาะส่วนที่พูดถึงเกม
MIN_CHARS = 0         # สั้นกว่านี้ตัดทิ้ง
MIN_WORDS = 0
MIN_ALPHA_RATIO = 0.6   # สัดส่วนตัวอักษรต่อตัวที่ไม่ใช่ช่องว่าง ต่ำกว่านี้ = ASCII art / สัญลักษณ์
MIN_UNIQUE_RATIO = 0.4  # สัดส่วนคำไม่ซ้ำ ต่ำกว่านี้ = พิมพ์ซ้ำๆ (เช็คเมื่อมี >= 10 คำ)
DUP_THRESHOLD = 2       # ข้อความเดียวกันโผล่ >= กี่ครั้งในพูล = copypasta

# แพทเทิร์น copypasta/มีมยอดฮิต (เพิ่มเองได้ตามที่เจอในข้อมูลจริง)
COPYPASTA = re.compile("|".join([
    r"never gonna give you up",
    r"according to all known laws",
    r"what the (fuck|f\*+k|heck) did you just say",
    r"navy seal",
    r"ate the whole thing",
    r"\balways has been\b",
    r"\bbee movie\b",
    r"\brickroll",
    r"\bskibidi\b",
    r"\b(like|upvote|thumbs? up) (this )?(comment |review |post )?if\b",   # like if you ...
    r"\blike and subscribe\b",
    r"\bif you('re| are)? (reading|read) this\b",
    r"\bsexual fantas",
    r"\b(think|thinks) (i'?m|im|you'?re) gay\b",
]), re.I)

# สแปม / โฆษณา / ลิงก์ / ชวนเข้ากลุ่ม
SPAM = re.compile("|".join([
    r"https?://|\bwww\.|\b[a-z0-9-]+\.(com|net|org|gg|io|ru|xyz|tv|me|ly|cc|shop|store)\b/?",
    r"\b(discord\.gg|t\.me|bit\.ly|tinyurl|linktr\.ee)\b",
    r"[\w.+-]+@[\w-]+\.[\w.]+",                                   # อีเมล
    r"\b(giveaway|promo ?code|coupon|referral|cd-?keys?|key ?shop|gift ?cards?)\b",
    r"\bfree (skins?|keys?|gift|robux|v-?bucks|money|coins?|gems?)\b",
    r"\b(buy|get|sell(ing)?) (cheap|cheapest) (keys?|games?|accounts?)\b",
    r"\b(dm|pm) me\b|\badd me\b|\bfriend request\b",
    r"\bfollow (my|me)\b|\bsubscribe (to )?(my|me|the channel)\b|\bcheck out my\b|\bclick (the |this )?link\b",
    r"\b(join|visit) (my|our) (discord|server|group|clan|guild|channel|community)\b",
    r"\bmy (curator|profile|channel|discord|youtube|twitch|steam group)\b",
    r"\b(whatsapp|telegram|bitcoin|onlyfans|online casino|sports betting)\b",
]), re.I)
REPEAT_CHAR = re.compile(r"(\S)\1{9,}")                  # อักขระเดียวซ้ำ 10 ตัวขึ้นไป
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
HASHTAG = re.compile(r"#\w+")
STEAM_EMOTE = re.compile(r":[A-Za-z0-9_]{3,}:")            # :steamhappy: ฯลฯ

# รีวิวที่เล่าเรื่องชีวิตส่วนตัว (อายุ ครอบครัว อาชีพ) ไม่ได้พูดถึงตัวเกม
PERSONAL = re.compile("|".join([
    r"\b\d{1,2} ?(yo|y/o|yrs? old|years? old|year[- ]old)\b",
    r"\b(my|our) (son|daughter|wife|husband|girlfriend|boyfriend|fianc[ée]e?|mom|mum|dad|"
    r"mother|father|kids?|children|grandson|granddaughter|parents|brother|sister|uncle|aunt|boss)\b",
    r"\bsingle (father|farther|mother|dad|mom|parent)\b",
    r"\bas an? (father|mother|dad|mom|parent|husband|wife|nurse|teacher|student|biker|trucker)\b",
    r"\bmy (job|marriage|divorce|depression|childhood|therapy|funeral)\b",
]), re.I)
PERSONAL_MAX_STRONG = 3   # ถ้ามีเรื่องส่วนตัว แต่พูดถึงเนื้อเกมเฉพาะ >= กี่คำ ถึงจะยอมให้ผ่าน

# อักขระที่ใช้วาด ASCII art (block elements, box drawing, braille)
ART_CHARS = re.compile(r"[\u2500-\u259F\u2800-\u28FF]")

# คำเฉพาะของการเล่นเกม = หลักฐานว่าพูดถึงตัวเกมจริง (เช็คแบบขึ้นต้นด้วย)
STRONG_STEMS = (
    "gameplay", "graphic", "visual", "music", "soundtrack", "level", "boss", "combat",
    "fight", "charact", "quest", "mission", "campaign", "multiplayer", "co-op", "coop",
    "control", "bug", "glitch", "crash", "dlc", "update", "patch", "developer", "devs",
    "price", "worth", "achiev", "grind", "puzzle", "mechanic", "difficult", "weapon",
    "enem", "skill", "upgrade", "unlock", "fps", "lag", "server", "refund", "replay",
    "sandbox", "rpg", "roguelike", "farming", "explor", "design", "animation",
    "narrative", "ending", "loot", "optimi", "story", "plot", "crafting", "content",
    "pvp", "pve", "dungeon", "npc", "gear", "inventory", "progress", "soundtrack",
)
# คำกว้างๆ ที่มีมก็ใช้ (ต้องมาคู่กับคำเฉพาะอย่างน้อย 1 คำถึงจะนับ)
WEAK_STEMS = (
    "game", "play", "fun", "world", "build", "buy", "purchase", "recommend", "addict",
    "relax", "steam", "port", "mod", "map", "item", "class", "stat", "save", "hour",
    "art", "sound",
)
NAME_STOP = {"the", "of", "and", "for", "edition", "game", "remastered",
             "definitive", "complete", "deluxe", "goty"}


def _tokens(s):
    return re.findall(r"[a-z0-9'\-]+", s.lower())


def _name_tokens(name):
    return {t for t in _tokens(name) if len(t) >= 4 and t not in NAME_STOP}


def _hard_reject(row):
    """กฎที่ดูจากข้อความเต็ม: สั้น/สแปม/ASCII art/พิมพ์ซ้ำ/copypasta คืนเหตุผลที่ตัด หรือ None"""
    text = row["text"]
    raw = row.get("_raw", text)
    words = _tokens(text)

    if len(text) < MIN_CHARS or len(words) < MIN_WORDS:
        return "สั้นเกินไป"
    if SPAM.search(text):
        return "สแปม/โฆษณา/ลิงก์"
    if REPEAT_CHAR.search(text):
        return "สแปม (พิมพ์ตัวเดิมซ้ำ)"
    if len(EMOJI.findall(text)) >= 8 or len(HASHTAG.findall(text)) >= 3 \
            or len(STEAM_EMOTE.findall(text)) >= 6:
        return "สแปม (อีโมจิ/แฮชแท็กเยอะ)"
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 30 and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        return "สแปม (ตัวพิมพ์ใหญ่ล้วน)"
    if len(ART_CHARS.findall(text)) >= 10:
        return "ascii_art"
    non_space = [c for c in text if not c.isspace()]
    if non_space and sum(c.isalpha() for c in non_space) / len(non_space) < MIN_ALPHA_RATIO:
        return "สัญลักษณ์เยอะ/ascii_art"
    n_lines = raw.count("\n") + 1
    if n_lines >= 8 and len(raw) / n_lines < 25:
        return "ascii_art (บรรทัดสั้นๆ ซ้อนกัน)"
    if len(words) >= 10 and len(set(words)) / len(words) < MIN_UNIQUE_RATIO:
        return "พิมพ์ซ้ำๆ"
    if COPYPASTA.search(text):
        return "copypasta"
    return None


def _relevance_reject(text, name_tokens):
    """เช็คว่าข้อความนี้พูดถึงตัวเกมจริงไหม (ไม่ใช่เล่าชีวิตส่วนตัว / ไม่เกี่ยวกับเกม)"""
    words = _tokens(text)
    uniq = set(words)
    strong = sum(w.startswith(STRONG_STEMS) for w in uniq)
    weak = sum(w.startswith(WEAK_STEMS) for w in uniq)
    personal = len(PERSONAL.findall(text))
    if personal and (strong < PERSONAL_MAX_STRONG or personal >= 3):
        return "เล่าเรื่องชีวิตส่วนตัว"
    # ผ่านเมื่อ: มีชื่อเกม หรือ มีคำเฉพาะเกม >= 2 คำ หรือ คำเฉพาะ 1 + คำกว้างๆ 1
    on_topic = bool(uniq & name_tokens) or strong >= 2 or (strong >= 1 and weak >= 1)
    if not on_topic:
        return "ไม่เกี่ยวกับเกม"
    return None




def _reject_reason(row, name_tokens):
    """รวมกฎทั้งหมด (ไม่ตัดประโยค) คืนเหตุผลที่ควรตัดทิ้ง หรือ None ถ้ารีวิวนี้ใช้ได้"""
    return _hard_reject(row) or _relevance_reject(row["text"], name_tokens)


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _trim_to_game(raw, name_tokens):
    """ตัดประโยคที่ไม่เกี่ยวกับเกมออก เก็บเฉพาะส่วนที่พูดถึงเกม
    ประโยคที่โดนตัด = เล่าเรื่องส่วนตัว/สแปม/copypasta และไม่มีชื่อเกมหรือคำเฉพาะเกมอยู่เลย
    คืน (ข้อความที่เหลือ, จำนวนประโยคที่ตัด)"""
    kept, dropped = [], 0
    for sent in SENT_SPLIT.split(raw):
        sent = sent.strip()
        if not sent:
            continue
        uniq = set(_tokens(sent))
        has_game = bool(uniq & name_tokens) or any(w.startswith(STRONG_STEMS) for w in uniq)
        if not has_game and (PERSONAL.search(sent) or SPAM.search(sent) or COPYPASTA.search(sent)):
            dropped += 1
            continue
        kept.append(sent)
    return " ".join(kept), dropped


def filter_reviews(pool, game_name):
    """แยกรีวิวเป็น (ใช้ได้, ถูกตัดทิ้ง[(เหตุผล, แถว)])"""
    if not ENABLE_FILTER:
        return list(pool), []
    name_tokens = _name_tokens(game_name)
    # ข้อความเดียวกันซ้ำหลายรีวิว = copypasta
    keys = [" ".join(_tokens(r.get("text_full", r["text"])))[:200] for r in pool]
    counts = Counter(keys)
    clean, rejected = [], []
    for row, key in zip(pool, keys):
        if "_r" not in row:
            row["text_full"] = row["text"]            # เก็บข้อความเต็มไว้เสมอ
            reason = _hard_reject(row)
            if reason is None and TRIM_TO_GAME_PARTS:
                trimmed, dropped = _trim_to_game(row.get("_raw", row["text"]), name_tokens)
                if dropped:
                    row["text"] = trimmed
                    if len(trimmed) < MIN_CHARS or len(_tokens(trimmed)) < MIN_WORDS:
                        reason = "ไม่เหลือส่วนที่เกี่ยวกับเกมพอ (ตัดเรื่องส่วนตัวออกแล้วสั้นเกินไป)"
            if reason is None:
                reason = _relevance_reject(row["text"], name_tokens)
            row["_r"] = reason
        reason = row["_r"]
        if reason is None and counts[key] >= DUP_THRESHOLD:
            reason = "ข้อความซ้ำกันหลายรีวิว (copypasta)"
        if reason:
            rejected.append((reason, row))
        else:
            clean.append(row)
    return clean, rejected


# ---------- 1. ค้นหาเกมจากชื่อ ----------
def search_games(term, limit=5):
    """คืนรายการเกมที่ชื่อตรงกับคำค้น: [{"appid": 413150, "name": "Stardew Valley"}, ...]"""
    r = requests.get(SEARCH_URL, params={"term": term, "l": "english", "cc": "US"}, timeout=30)
    r.raise_for_status()
    items = r.json().get("items", [])
    return [{"appid": it["id"], "name": it["name"]} for it in items[:limit]]


# ---------- 1.5 ตัวเลขจริงจาก Steam (ใช้แสดง % บวก/ลบ ให้ตรงกับหน้าร้าน) ----------
def get_review_summary(appid, language=SUMMARY_LANGUAGE):
    """คืนสถิติรีวิวจริงของเกม เช่น
    {"positive": 1036071, "negative": 15953, "total": 1052024,
     "positive_pct": 98.5, "negative_pct": 1.5, "score_desc": "Overwhelmingly Positive"}
    ถ้าดึงไม่ได้คืน None"""
    for attempt in range(3):
        try:
            r = requests.get(
                SUMMARY_URL.format(appid=int(appid)),
                params={"json": 1, "language": language, "filter": "all",
                        "review_type": "all", "purchase_type": "all", "num_per_page": 0},
                timeout=30)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            q = r.json().get("query_summary", {})
            pos, neg = int(q.get("total_positive", 0)), int(q.get("total_negative", 0))
            if pos + neg == 0:
                return None
            total = pos + neg
            return {"positive": pos, "negative": neg, "total": total,
                    "positive_pct": round(pos / total * 100, 1),
                    "negative_pct": round(neg / total * 100, 1),
                    "score_desc": q.get("review_score_desc", "")}
        except (requests.RequestException, ValueError):
            time.sleep(5)
    print("  ดึงสถิติรีวิวรวมจาก Steam ไม่ได้ ใช้แบ่งครึ่งแทน")
    return None


def split_counts(total_n, summary):
    """แบ่ง total_n เป็น (บวก, ลบ) ตามสัดส่วนจริง แต่ไม่ต่ำกว่า MIN_PER_SIDE ต่อฝั่ง"""
    if not summary:
        return total_n - total_n // 2, total_n // 2
    neg = round(total_n * summary["negative"] / summary["total"])
    neg = min(max(neg, MIN_PER_SIDE), total_n - MIN_PER_SIDE)
    return total_n - neg, neg


# ---------- 2. ดึงรีวิว ----------
def _call_api(query):
    for attempt in range(6):
        try:
            r = requests.get(REVIEWS_URL, params={"input_json": json.dumps(query)}, timeout=30)
        except requests.RequestException as e:
            print(f"  เน็ตมีปัญหา ({e.__class__.__name__}) รอ 10 วินาที...")
            time.sleep(10)
            continue
        if r.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"  โดนจำกัดความถี่ (429) รอ {wait} วินาที...")
            time.sleep(wait)
            continue
        if r.status_code != 200:
            print(f"  Steam ตอบ status {r.status_code}: {r.text[:200]!r}")
            return None
        return r.json().get("response", {})
    print("  ลองหลายรอบแล้วยังไม่ได้ หยุดดึง")
    return None


def _collect(appid, game_name, pool, seen, top_n, review_type):
    cursor = "*"
    while len(pool) < MAX_SCAN:
        data = _call_api({
            "appid": int(appid),
            "filter": 0,                   # Helpful = เรียงตามความมีประโยชน์
            "day_range": 0,                # 0 = ทุกช่วงเวลา (ไม่ใช่แค่ 30 วันล่าสุด)
            "use_review_quality": False,   # เรียงตามโหวตตรง ๆ
            "languages": [LANGUAGE],
            "review_type": review_type,    # 1 = เฉพาะบวก, 2 = เฉพาะลบ
            "purchase_type": 1,            # ทุกช่องทางการซื้อ
            "num_per_page": 100,
            "cursor": cursor,
        })
        if data is None:
            break
        batch = data.get("reviews", [])
        new = 0
        for rv in batch:
            if rv["recommendationid"] in seen:
                continue
            seen.add(rv["recommendationid"])
            new += 1
            raw = rv.get("review", "").strip()
            pool.append({
                "appid": appid,
                "game_name": game_name,
                "review_id": rv["recommendationid"],
                "text": raw.replace("\n", " "),
                "_raw": raw,                                  # เก็บไว้เช็ค ASCII art (ลบทิ้งก่อนเซฟ)
                "voted_up": int(rv.get("voted_up", False)),   # 1 = แนะนำ, 0 = ไม่แนะนำ
                "votes_helpful": rv.get("votes_up", 0),
                "playtime_hours": round(rv.get("author", {}).get("playtime_forever", 0) / 60, 1),
                "timestamp": rv.get("timestamp_created", 0),
            })
        clean, _ = filter_reviews(pool, game_name)

        # 1) ความคืบหน้าการดึงรีวิวดิบ เทียบกับเพดาน MAX_SCAN
        fetch_percent = min(len(pool) / MAX_SCAN * 100, 100) if MAX_SCAN else 100

        # 2) สัดส่วนรีวิวที่ผ่านตัวกรอง เทียบกับจำนวนที่ต้องการ top_n
        filter_percent = min(len(clean) / top_n * 100, 100) if top_n else 100

        bar_width = 20

        # Progress bar สำหรับการดึงรีวิว
        fetch_filled = int(bar_width * fetch_percent / 100)
        fetch_bar = "█" * fetch_filled + "░" * (bar_width - fetch_filled)

        # Progress bar สำหรับรีวิวที่ผ่านตัวกรอง
        filter_filled = int(bar_width * filter_percent / 100)
        filter_bar = "█" * filter_filled + "░" * (bar_width - filter_filled)

        print(
            f"  ความคืบหน้าการดึง: [{fetch_bar}] {fetch_percent:5.1f}% "
            f"| ดึงแล้ว {len(pool):,}/{MAX_SCAN:,} รายการ"
        )
        print(
            f"  ผ่านตัวกรอง:       [{filter_bar}] {filter_percent:5.1f}% "
            f"| ผ่าน {len(clean):,}/{top_n:,} รายการ"
        )
        if len(clean) >= top_n:            # ได้รีวิวที่ผ่านตัวกรองครบแล้ว ไม่ต้องดึงต่อ
            break
        if new == 0:                       # ไม่มีรีวิวใหม่แล้ว = หมด
            break
        next_cursor = data.get("cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.5)


def _fetch_side(appid, game_name, review_type, top_n, rejected):
    pool, seen = [], set()
    try:
        _collect(appid, game_name, pool, seen, top_n, review_type)
    except KeyboardInterrupt:
        print("\nหยุดกลางคัน (Ctrl+C) ใช้เท่าที่เก็บได้")
    clean, rej = filter_reviews(pool, game_name)
    if rejected is not None:
        rejected.extend(rej)
    clean.sort(key=lambda r: r["votes_helpful"], reverse=True)
    top = clean[:top_n]
    for i, row in enumerate(top, 1):      # rank = อันดับภายในฝั่งตัวเอง (บวก/ลบ)
        row["rank"] = i
        row.pop("_raw", None)
        row.pop("_r", None)
    return top


def fetch_top_reviews(appid, game_name, top_n=None, top_pos=None, top_neg=None,
                      rejected=None, mode=MODE):
    """ดึงรีวิวที่ยอด helpful สูงสุด (หลังกรองมีมแล้ว) ตามโหมด:
      mixed        : รวมบวก+ลบ เรียงตามไลค์ เอา top_n อันดับแรก (ค่าเริ่มต้น เร็วสุด)
      proportional : แบ่งบวก/ลบตามสัดส่วนจริงของ Steam แล้วเอาไลค์สูงสุดของแต่ละฝั่ง
      balanced     : แบ่งบวก/ลบอย่างละครึ่ง
    top_pos / top_neg: ระบุจำนวนแต่ละฝั่งเอง (ข้ามโหมดข้างบน)
    rejected: ถ้าส่ง list มา จะเอารีวิวที่ถูกกรองทิ้งใส่ให้เป็น (เหตุผล, แถว)"""
    total = min(top_n or TOP_N, MAX_REVIEWS)
    if top_pos is not None and top_neg is not None and top_pos + top_neg > MAX_REVIEWS:
        top_pos = round(MAX_REVIEWS * top_pos / (top_pos + top_neg))
        top_neg = MAX_REVIEWS - top_pos
    if top_pos is None or top_neg is None:
        if mode == "mixed":
            return _fetch_side(appid, game_name, 0, total, rejected)   # review_type 0 = ทั้งบวกและลบ
        summary = get_review_summary(appid) if mode == "proportional" else None
        top_pos, top_neg = split_counts(total, summary)
        if summary:
            print(f"  Steam จริง: บวก {summary['positive_pct']}% / ลบ {summary['negative_pct']}% "
                  f"({summary['total']:,} รีวิว, {summary['score_desc']})")
        print(f"  เก็บ: บวก {top_pos} + ลบ {top_neg} = {top_pos + top_neg}")
    print(" -- ฝั่งบวก --")
    pos = _fetch_side(appid, game_name, 1, top_pos, rejected)
    print(" -- ฝั่งลบ --")
    neg = _fetch_side(appid, game_name, 2, top_neg, rejected)
    return pos + neg


# ---------- 2.5 สรุปสัดส่วนบวก/ลบจากรีวิวที่ดึงมา ----------
def summarize_rows(rows):
    """สรุปจากป้าย voted_up จริงของ Steam (ปุ่ม Recommended ที่คนเขียนกดเอง ไม่ใช่ผลโมเดล)
    - positive_pct          : นับทุกรีวิวเท่ากัน
    - weighted_positive_pct : ถ่วงน้ำหนักด้วยยอดไลค์ (รีวิวที่คนเห็นด้วยเยอะมีเสียงมากกว่า)"""
    n = len(rows)
    pos = sum(r["voted_up"] for r in rows)
    likes_all = sum(r["votes_helpful"] for r in rows)
    likes_pos = sum(r["votes_helpful"] for r in rows if r["voted_up"])
    return {
        "count": n, "positive": pos, "negative": n - pos,
        "positive_pct": round(pos / n * 100, 1) if n else 0.0,
        "weighted_positive_pct": round(likes_pos / likes_all * 100, 1) if likes_all else 0.0,
        "weighted_negative_pct": round((likes_all - likes_pos) / likes_all * 100, 1) if likes_all else 0.0,
    }


# ---------- 3. บันทึกไฟล์ ----------
def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_rejected(rejected, path):
    """เซฟรีวิวที่ถูกกรองทิ้งพร้อมเหตุผล ไว้ตรวจว่าตัดผิดไหม"""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["reason", "game_name", "votes_helpful", "voted_up", "text"])
        for reason, r in rejected:
            w.writerow([reason, r["game_name"], r["votes_helpful"], r["voted_up"], r.get("text_full", r["text"])])


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() or "game"


# ---------- ใช้งานจาก command line ----------
def pick_game(term):
    games = search_games(term)
    if not games:
        print(f"ไม่เจอเกมชื่อ '{term}' ข้ามไป")
        return None
    print(f"\nผลการค้นหา '{term}':")
    for i, g in enumerate(games, 1):
        print(f"  {i}. {g['name']}  (appid {g['appid']})")
    if len(games) == 1:
        return games[0]
    pick = input(f"เลือกเกม 1-{len(games)} [Enter = 1]: ").strip() or "1"
    return games[int(pick) - 1]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # รับหลายเกมคั่นด้วย , เช่น "stardew valley, elden ring, hades"
    raw = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        input("พิมพ์ชื่อเกม (ภาษาอังกฤษ, หลายเกมคั่นด้วย ,): ")
    terms = [t.strip() for t in raw.split(",") if t.strip()]
    if not terms:
        sys.exit("ไม่ได้ใส่ชื่อเกม")

    # เลือกเกมให้ครบก่อน แล้วค่อยดึงรวดเดียว
    chosen = [g for g in (pick_game(t) for t in terms) if g]

    all_rows, done, skipped = [], [], []
    for game in chosen:
        print(f"\n=== {game['name']} : {TOP_N} รีวิวยอดโหวตสูงสุด (กรองมีมออก) ===")
        rejected = []
        rows = fetch_top_reviews(game["appid"], game["name"], rejected=rejected)
        slug = safe_filename(game["name"])
        if rejected:
            save_rejected(rejected, os.path.join(OUT_DIR, f"rejected_{slug}.csv"))
            reasons = Counter(r for r, _ in rejected)
            print(f"  กรองทิ้ง {len(rejected)} รีวิว -> rejected_{slug}.csv  {dict(reasons)}")
        pos = sum(r["voted_up"] for r in rows)
        neg = len(rows) - pos
        if not rows:
            print("  ⚠ ไม่ได้รีวิวเลย (ไม่เจอหรือโดนกรองหมด) ไม่บันทึกเกมนี้")
            skipped.append((game["name"], 0, 0))
            continue
        if len(rows) < TOP_N:
            print(f"  ℹ ได้ {len(rows)} รีวิว (น้อยกว่า {TOP_N} เพราะเกมนี้มีรีวิวที่ผ่านตัวกรองไม่พอ) บันทึกเท่าที่ได้")
        out = os.path.join(OUT_DIR, f"reviews_{slug}.csv")
        save_csv(rows, out)
        all_rows.extend(rows)
        done.append((game["name"], out, pos, neg))
        print(f"  ✓ {len(rows)} รีวิว -> {out}  (positive {pos} / negative {neg})")

    if len(done) > 1:
        save_csv(all_rows, os.path.join(OUT_DIR, "reviews_all_games.csv"))

    print("\n===== สรุป =====")
    for name, out, pos, neg in done:
        print(f"  ✓ {name}: positive {pos} / negative {neg} -> {out}")
    for name, pos, neg in skipped:
        print(f"  ✗ {name}: ไม่ได้รีวิวเลย (ไม่ได้บันทึก)")
    if len(done) > 1:
        print(f"  รวมทุกเกม {len(all_rows):,} รีวิว -> reviews_all_games.csv")


if __name__ == "__main__":
    main()
