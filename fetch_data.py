"""
A股历史数据完整获取系统 v2.0
- 股票列表：新浪 hs_a（分页，覆盖沪深全A+北交所）
- 日K线：东方财富 历史K线接口（前复权）
- 存储：SQLite
- 时间范围：2023-01-01 ~ 今天
"""

import sqlite3, time, random, sys, json, os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ───────────────────── 配置 ─────────────────────
BASE_DIR    = Path(__file__).parent
DB_PATH     = BASE_DIR / "data" / "stock_db.sqlite"
START_DATE  = "20230101"
END_DATE    = datetime.today().strftime("%Y%m%d")
SLEEP_MIN   = 0.06
SLEEP_MAX   = 0.13
MAX_RETRY   = 3
SKIP_BARS   = 300   # 已有>=300根K线则跳过

DB_PATH.parent.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "http://data.eastmoney.com/",
}

# ─────────────────── 数据库 ───────────────────
def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS stock_list (
            code TEXT PRIMARY KEY, name TEXT, market TEXT, updated TEXT
        );
        CREATE TABLE IF NOT EXISTS daily_bar (
            code TEXT, date TEXT, open REAL, high REAL, low REAL,
            close REAL, volume REAL, amount REAL, turnover REAL,
            PRIMARY KEY (code, date)
        );
        CREATE INDEX IF NOT EXISTS idx_bar_date ON daily_bar(date);
        CREATE INDEX IF NOT EXISTS idx_bar_code ON daily_bar(code);
    """)
    conn.commit()

# ─────────────────── 股票列表 ───────────────────
def fetch_stock_list_sina() -> pd.DataFrame:
    """新浪 hs_a 节点（沪深全A+北交所），支持翻页"""
    print("[Step 1] 获取股票列表（新浪 hs_a 分页）...")
    NUM_PER_PAGE = 100   # 新浪单页最大返回100条
    nodes = [
        ("hs_a",  ),  # 沪深全部A股
        ("bj_a",  ),  # 北交所
    ]
    all_rows = []
    seen = set()

    for (node,) in nodes:
        page = 1
        consecutive_empty = 0
        while True:
            url = (
                "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php"
                f"/Market_Center.getHQNodeData"
                f"?page={page}&num={NUM_PER_PAGE}&sort=symbol&asc=1&node={node}&_s_r_a=page"
            )
            try:
                resp = requests.get(url, headers=HEADERS, timeout=12)
                resp.raise_for_status()
                items = resp.json()
                if not items:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        break
                    page += 1
                    continue
                consecutive_empty = 0
                added = 0
                for it in items:
                    sym = str(it.get("symbol", ""))
                    if sym in seen:
                        continue
                    seen.add(sym)
                    if sym.startswith("sh"):
                        mkt, code = "SH", sym[2:]
                    elif sym.startswith("sz"):
                        mkt, code = "SZ", sym[2:]
                    elif sym.startswith("bj"):
                        mkt, code = "BJ", sym[2:]
                    else:
                        continue
                    all_rows.append({
                        "code":   code.zfill(6),
                        "name":   it.get("name", ""),
                        "market": mkt,
                    })
                    added += 1
                print(f"  {node} page {page}: +{added} (累计: {len(all_rows)})")
                # 如果返回数量小于每页大小，说明到最后一页了
                if len(items) < NUM_PER_PAGE:
                    break
                page += 1
                time.sleep(0.15)
            except Exception as e:
                print(f"  {node} page {page} error: {e}")
                time.sleep(1)
                page += 1
                if page > 100:  # 防止无限循环
                    break

    df = pd.DataFrame(all_rows)
    print(f"  => 共 {len(df)} 只股票")
    return df

# ─────────────────── K线 ───────────────────
def fetch_kline_em(code: str, market: str) -> pd.DataFrame | None:
    """东方财富 前复权日K线"""
    if market in ("SH",):
        secid = f"1.{code}"
    elif market in ("SZ",):
        secid = f"0.{code}"
    else:  # BJ
        secid = f"0.{code}"

    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?secid={secid}"
        "&fields1=f1,f2,f3,f4,f5,f6"
        "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
        f"&klt=101&fqt=1&beg={START_DATE}&end={END_DATE}"
        "&_=1710000000000"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        body = resp.json().get("data") or {}
        klines = body.get("klines") or []
        if len(klines) < 20:
            return None
        rows = []
        for k in klines:
            p = k.split(",")
            if len(p) < 7:
                continue
            try:
                rows.append({
                    "date":     p[0],
                    "open":     float(p[1]),
                    "close":    float(p[2]),
                    "high":     float(p[3]),
                    "low":      float(p[4]),
                    "volume":   float(p[5]),
                    "amount":   float(p[6]),
                    "turnover": float(p[10]) if len(p) > 10 else 0.0,
                })
            except (ValueError, IndexError):
                continue
        return pd.DataFrame(rows) if len(rows) >= 20 else None
    except Exception:
        return None

# ─────────────────── 批量写入 ───────────────────
def upsert_bars(conn, code: str, df: pd.DataFrame):
    df = df.copy()
    df["code"] = code
    data = df[["code","date","open","high","low","close",
               "volume","amount","turnover"]].values.tolist()
    conn.executemany(
        "INSERT OR REPLACE INTO daily_bar "
        "(code,date,open,high,low,close,volume,amount,turnover) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        data
    )

# ─────────────────── 主流程 ───────────────────
def main():
    t0 = time.time()
    print("=" * 65)
    print("  A股历史数据下载系统 v2.0")
    print(f"  时间范围: {START_DATE} ~ {END_DATE}")
    print(f"  数据库:   {DB_PATH}")
    print("=" * 65)

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    init_db(conn)

    # ── Step 1: 股票列表 ──
    df_list = fetch_stock_list_sina()
    if df_list.empty:
        print("ERROR: 无法获取股票列表")
        conn.close()
        return

    df_list["updated"] = datetime.today().strftime("%Y-%m-%d")
    for _, r in df_list.iterrows():
        conn.execute(
            "INSERT OR REPLACE INTO stock_list (code,name,market,updated) VALUES (?,?,?,?)",
            (r["code"], r["name"], r["market"], r["updated"])
        )
    conn.commit()
    print(f"  已写入股票列表: {len(df_list)} 只\n")

    # ── Step 2: K线下载 ──
    # 查已有
    existing = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT code, COUNT(*) FROM daily_bar GROUP BY code"
        )
    }

    # 支持 --test 参数，只下载前N只
    test_mode = "--test" in sys.argv
    if test_mode:
        try:
            n = int(sys.argv[sys.argv.index("--test") + 1])
        except (IndexError, ValueError):
            n = 200
        df_list = df_list.head(n)
        print(f"  [TEST MODE] 只下载前 {n} 只")

    print(f"[Step 2] 下载K线（{len(df_list)} 只, 已有{len(existing)}只数据）...")

    total   = len(df_list)
    success = 0
    fail    = 0
    skip    = 0

    for idx, row in df_list.iterrows():
        code   = row["code"]
        market = row["market"]

        if existing.get(code, 0) >= SKIP_BARS:
            skip += 1
            continue

        df_bar = None
        for attempt in range(MAX_RETRY):
            df_bar = fetch_kline_em(code, market)
            if df_bar is not None:
                break
            time.sleep(0.4 * (attempt + 1))

        if df_bar is not None:
            upsert_bars(conn, code, df_bar)
            success += 1
        else:
            fail += 1

        done = success + fail + skip
        if done % 100 == 0:
            conn.commit()

        if done % 250 == 0 or done == total:
            elapsed = time.time() - t0
            remain  = total - done
            eta     = (elapsed / max(done - skip, 1)) * remain if done > skip else 0
            pct     = done / total * 100
            bw      = 28
            filled  = int(pct / 100 * bw)
            bar     = "#" * filled + "-" * (bw - filled)
            print(
                f"  [{bar}] {pct:5.1f}%  "
                f"done:{done}/{total}  ok:{success} fail:{fail} skip:{skip}  "
                f"elapsed:{elapsed:.0f}s eta:{eta:.0f}s"
            )
            sys.stdout.flush()

        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    conn.commit()

    # ── Step 3: 汇报 ──
    total_bars  = conn.execute("SELECT COUNT(*) FROM daily_bar").fetchone()[0]
    total_codes = conn.execute(
        "SELECT COUNT(DISTINCT code) FROM daily_bar"
    ).fetchone()[0]
    elapsed = time.time() - t0

    print(f"\n[Step 3] 完成!")
    print(f"  股票数量:  {total_codes:,}")
    print(f"  K线条数:   {total_bars:,}")
    print(f"  总耗时:    {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  数据库:    {DB_PATH}")

    sample = pd.read_sql(
        """SELECT s.name, s.market, d.code, MIN(d.date) start, MAX(d.date) end, COUNT(*) bars
           FROM daily_bar d JOIN stock_list s ON d.code=s.code
           GROUP BY d.code ORDER BY RANDOM() LIMIT 10""",
        conn
    )
    print("\n  随机抽样10只：")
    print(sample.to_string(index=False))

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
