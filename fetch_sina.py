"""
fetch_sina.py  ——  新浪K线接口批量下载A股日K数据
- 每只股票一次请求，获取最近1000根日K（约4年）
- 断点续传，已有≥300根K线的跳过
- 速度：约 5-8 stocks/s，5000只约15-20分钟
"""

import sqlite3, time, sys, threading
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── 配置 ──
BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "data" / "stock_db.sqlite"
DATALEN    = 1000    # 最多取1000根K线（约4年）
SKIP_BARS  = 300
MAX_RETRY  = 3
WORKERS    = 4       # 新浪接口比较友好，可用4线程
TIMEOUT    = 8

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer":    "https://finance.sina.com.cn/",
}

# ── K线下载 ──
def fetch_kline_sina(code: str, market: str) -> pd.DataFrame | None:
    prefix = "sh" if market == "SH" else "sz"
    sym    = f"{prefix}{code}"
    url    = (
        f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php"
        f"/CN_MarketData.getKLineData"
        f"?symbol={sym}&scale=240&ma=no&datalen={DATALEN}"
    )
    for attempt in range(MAX_RETRY):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            items = resp.json()
            if not items or len(items) < 20:
                return None
            rows = []
            for it in items:
                try:
                    rows.append({
                        "date":     str(it["day"]),
                        "open":     float(it["open"]),
                        "close":    float(it["close"]),
                        "high":     float(it["high"]),
                        "low":      float(it["low"]),
                        "volume":   float(it["volume"]),
                        "amount":   0.0,
                        "turnover": 0.0,
                    })
                except (KeyError, ValueError):
                    continue
            return pd.DataFrame(rows) if len(rows) >= 20 else None
        except Exception:
            if attempt < MAX_RETRY - 1:
                time.sleep(0.5 * (attempt + 1))
    return None

# ── 数据库写入 ──
db_lock = threading.Lock()

def write_bars(conn, code: str, df: pd.DataFrame):
    data = [
        (code, r["date"], r["open"], r["high"], r["low"],
         r["close"], r["volume"], r["amount"], r["turnover"])
        for _, r in df.iterrows()
    ]
    with db_lock:
        conn.executemany(
            "INSERT OR REPLACE INTO daily_bar "
            "(code,date,open,high,low,close,volume,amount,turnover) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            data
        )
        if len(data) > 0:
            conn.commit()

# ── Worker ──
def worker(task, conn):
    code, market, skip = task
    if skip:
        return "skip"
    df = fetch_kline_sina(code, market)
    if df is not None:
        write_bars(conn, code, df)
        return "ok"
    return "fail"

# ── 主流程 ──
def main():
    t0 = time.time()
    print("=" * 65)
    print("  新浪K线下载器 v1.0")
    print(f"  数据库: {DB_PATH}")
    print("=" * 65)

    conn = sqlite3.connect(str(DB_PATH), timeout=60, check_same_thread=False)

    df_list = pd.read_sql("SELECT code, market FROM stock_list", conn)
    print(f"股票列表: {len(df_list)} 只")

    existing = {
        r[0]: r[1]
        for r in conn.execute("SELECT code, COUNT(*) FROM daily_bar GROUP BY code")
    }
    skip_set = {c for c, n in existing.items() if n >= SKIP_BARS}
    print(f"已有: {len(existing)} 只，可跳过: {len(skip_set)} 只")

    tasks = [
        (r["code"], r["market"], r["code"] in skip_set)
        for _, r in df_list.iterrows()
    ]
    need = sum(1 for t in tasks if not t[2])
    print(f"需下载: {need} 只\n")

    ok = fail = skip_cnt = done = 0
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(worker, t, conn): t for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            if res == "ok":    ok += 1
            elif res == "fail": fail += 1
            else:              skip_cnt += 1
            done += 1

            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                speed   = (ok + fail) / max(elapsed, 1)
                remain  = need - ok - fail
                eta     = remain / max(speed, 0.01)
                pct     = done / total * 100
                bw      = 28
                bar     = "#" * int(pct/100*bw) + "-" * (bw - int(pct/100*bw))
                print(
                    f"\r  [{bar}] {pct:5.1f}%  "
                    f"ok:{ok} fail:{fail} skip:{skip_cnt}  "
                    f"{speed:.1f}/s  ETA:{int(eta)}s   ",
                    end="", flush=True
                )

    print()

    # 统计
    total_bars  = conn.execute("SELECT COUNT(*) FROM daily_bar").fetchone()[0]
    total_codes = conn.execute("SELECT COUNT(DISTINCT code) FROM daily_bar").fetchone()[0]
    elapsed = time.time() - t0

    print(f"\n{'='*65}")
    print(f"  完成！股票: {total_codes:,}  K线: {total_bars:,}")
    print(f"  成功:{ok}  失败:{fail}  跳过:{skip_cnt}")
    print(f"  耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  速度: {(ok+fail)/max(elapsed,1):.1f} stocks/s")
    print(f"  DB: {DB_PATH} ({DB_PATH.stat().st_size/1024/1024:.1f} MB)")
    print(f"{'='*65}")

    conn.close()
    print("Done! 可以开始回测了。")


if __name__ == "__main__":
    main()
