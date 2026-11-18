"""
ค้นหาเกมจากชื่อ -> ดึงรีวิวภาษาอังกฤษที่มียอดโหวต helpful สูงสุด 1000 อันดับ -> CSV

ได้เกมละ 1000 รีวิวพอดี (เกมที่มีรีวิวไม่ถึง 1000 จะไม่บันทึก)

ใช้แบบถาม-ตอบ:   python steam_reviews.py
ใส่ชื่อเกมเลย:    python steam_reviews.py "stardew valley"
หลายเกม:         python steam_reviews.py "stardew valley, elden ring, hades"

เรียกจากโค้ดส่วนอื่นของทีม (pipeline):
    from steam_reviews import search_games, fetch_top_reviews, save_csv
    games = search_games("hades")
    rows = fetch_top_reviews(games[0]["appid"], games[0]["name"])
    save_csv(rows, "hades.csv")
"""
import csv
import json
import re
import sys
import time

import requests

SEARCH_URL = "https://store.steampowered.com/api/storesearch/"
REVIEWS_URL = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"
LANGUAGE = "english"   # โปรเจคนี้เอาเฉพาะภาษาอังกฤษ
TOP_N = 300           # จำนวนรีวิวต่อเกม (ฟิกไว้)
POOL_SIZE = TOP_N      # ดึงแค่ 1000 พอดี ไม่ดึงเผื่อ
FIELDS = ["rank", "appid", "game_name", "review_id", "text", "voted_up",
          "votes_helpful", "playtime_hours", "timestamp"]


# ---------- 1. ค้นหาเกมจากชื่อ ----------
def search_games(term, limit=5):
    """คืนรายการเกมที่ชื่อตรงกับคำค้น: [{"appid": 413150, "name": "Stardew Valley"}, ...]"""
    r = requests.get(SEARCH_URL, params={"term": term, "l": "english", "cc": "US"}, timeout=30)
    r.raise_for_status()
    items = r.json().get("items", [])
    return [{"appid": it["id"], "name": it["name"]} for it in items[:limit]]


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


def _collect(appid, game_name, pool, seen):
    cursor = "*"
    while len(pool) < POOL_SIZE:
        data = _call_api({
            "appid": int(appid),
            "filter": 0,                   # Helpful = เรียงตามความมีประโยชน์
            "day_range": 0,                # 0 = ทุกช่วงเวลา (ไม่ใช่แค่ 30 วันล่าสุด)
            "use_review_quality": False,   # เรียงตามโหวตตรง ๆ
            "languages": [LANGUAGE],
            "review_type": 0,              # ทั้งบวกและลบ
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
            pool.append({
                "appid": appid,
                "game_name": game_name,
                "review_id": rv["recommendationid"],
                "text": rv.get("review", "").replace("\n", " ").strip(),
                "voted_up": int(rv.get("voted_up", False)),   # 1 = แนะนำ, 0 = ไม่แนะนำ
                "votes_helpful": rv.get("votes_up", 0),
                "playtime_hours": round(rv.get("author", {}).get("playtime_forever", 0) / 60, 1),
                "timestamp": rv.get("timestamp_created", 0),
            })
        print(f"  เก็บแล้ว {len(pool):,} รีวิว")
        if new == 0:                       # ไม่มีรีวิวใหม่แล้ว = หมด
            break
        next_cursor = data.get("cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.5)


def fetch_top_reviews(appid, game_name, top_n=TOP_N):
    """ดึงรีวิวภาษาอังกฤษที่ยอดโหวต helpful สูงสุด top_n อันดับ"""
    pool, seen = [], set()
    try:
        _collect(appid, game_name, pool, seen)
    except KeyboardInterrupt:
        print("\nหยุดกลางคัน (Ctrl+C) ใช้เท่าที่เก็บได้")
    pool.sort(key=lambda r: r["votes_helpful"], reverse=True)
    top = pool[:top_n]
    for i, row in enumerate(top, 1):
        row["rank"] = i
    return top


# ---------- 3. บันทึกไฟล์ ----------
def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


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
        print(f"\n=== {game['name']} : ดึงรีวิวยอดโหวตสูงสุด {TOP_N} อันดับ ===")
        rows = fetch_top_reviews(game["appid"], game["name"])
        if len(rows) < TOP_N:
            print(f"  ⚠ ได้แค่ {len(rows):,} รีวิว (ไม่ถึง {TOP_N}) ไม่บันทึกเกมนี้")
            skipped.append((game["name"], len(rows)))
            continue
        out = f"reviews_{safe_filename(game['name'])}.csv"
        save_csv(rows, out)
        all_rows.extend(rows)
        pos = sum(r["voted_up"] for r in rows)
        done.append((game["name"], out, pos))
        print(f"  ✓ {TOP_N} รีวิว -> {out}  (positive {pos} / negative {TOP_N - pos})")

    if len(done) > 1:
        save_csv(all_rows, "reviews_all_games.csv")

    print("\n===== สรุป =====")
    for name, out, pos in done:
        print(f"  ✓ {name}: {TOP_N} รีวิว (positive {pos} / negative {TOP_N - pos}) -> {out}")
    for name, n in skipped:
        print(f"  ✗ {name}: มีแค่ {n:,} รีวิว ไม่ถึง {TOP_N} (ไม่ได้บันทึก)")
    if len(done) > 1:
        print(f"  รวมทุกเกม {len(all_rows):,} รีวิว -> reviews_all_games.csv")


if __name__ == "__main__":
    main()