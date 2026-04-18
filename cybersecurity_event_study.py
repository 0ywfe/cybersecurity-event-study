## event study on SEC 8-K cybersecurity disclosures (Item 1.05)
## replicating arXiv:2512.06144v1

import requests
import pandas as pd
import numpy as np
import yfinance as yf
import json, time
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

SMALL_CAP_THRESHOLD = 2_000_000_000
EVENT_WINDOW_DAYS = 7
ESTIMATION_WINDOW = 120
EDGAR_RATE_LIMIT = 0.15
START_DATE = "2023-01-01"

OUTPUT_DIR = Path(".")
CACHE_FILE = Path("cache/filings_cache.json")
TICKER_CACHE_FILE = Path("cache/ticker_cache.json")
RESULTS_FILE = Path("results/results.json")
EVENTS_FILE = Path("results/events.csv")

EDGAR_HEADERS = {
    "User-Agent": "Academic Research Event Study",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}


def fetch_edgar_filings(start_date, end_date=None, use_cache=True):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if use_cache and CACHE_FILE.exists():
        print("Loading filings from cache...")
        with open(CACHE_FILE) as f:
            cached = json.load(f)
        print(f"  Loaded {len(cached)} cached filings")
        return cached

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Fetching Item 1.05 filings from EDGAR ({start_date} to {end_date})...")

    filings = []
    from_idx = 0
    batch = 100
    total = None

    while True:
        url = (
            "https://efts.sec.gov/LATEST/search-index?"
            f"q=%221.05%22&forms=8-K"
            f"&dateRange=custom&startdt={start_date}&enddt={end_date}"
            f"&from={from_idx}&size={batch}"
        )

        try:
            resp = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"  EDGAR search error at offset {from_idx}: {exc}")
            break

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break

        if total is None:
            total = data.get("hits", {}).get("total", {}).get("value", 0)
            print(f"  Total filings found: {total}")

        for hit in hits:
            src = hit.get("_source", {})
            filing_date = src.get("period_of_report") or src.get("file_date", "")
            company = src.get("entity_name", "Unknown")
            if isinstance(company, list):
                company = company[0] if company else "Unknown"
            accession = hit.get("_id", "")

            cik = ""
            if accession:
                cik = accession.split("-")[0].lstrip("0")

            if filing_date and len(filing_date) >= 10:
                filings.append({
                    "filing_date": filing_date[:10],
                    "company_name": str(company),
                    "cik": cik,
                    "accession_number": accession,
                })

        from_idx += batch
        print(f"  Fetched {min(from_idx, total or from_idx)} / {total or '?'}...")

        if total and from_idx >= total:
            break

        time.sleep(EDGAR_RATE_LIMIT)

    ## dedup by accession number
    seen, unique = set(), []
    for f in filings:
        k = f["accession_number"]
        if k not in seen:
            seen.add(k)
            unique.append(f)

    print(f"Unique filings after dedup: {len(unique)}")

    with open(CACHE_FILE, "w") as f:
        json.dump(unique, f, indent=2)

    return unique


def cik_to_ticker(cik, ticker_cache):

    if not cik:
        return None

    if cik in ticker_cache:
        return ticker_cache[cik]

    cik_padded = cik.zfill(10)
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        tickers = data.get("tickers", [])
        ticker = tickers[0].upper() if tickers else None
    except Exception:
        ticker = None

    ticker_cache[cik] = ticker
    return ticker


def resolve_tickers(filings):

    if TICKER_CACHE_FILE.exists():
        with open(TICKER_CACHE_FILE) as f:
            ticker_cache = json.load(f)
    else:
        ticker_cache = {}

    print(f"\nResolving tickers for {len(filings)} filings...")
    resolved = []

    for i, filing in enumerate(filings):
        cik = filing.get("cik", "")
        ticker = cik_to_ticker(cik, ticker_cache)

        if ticker:
            f = dict(filing)
            f["ticker"] = ticker
            resolved.append(f)

        time.sleep(EDGAR_RATE_LIMIT)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(filings)} processed, {len(resolved)} tickers found")
            with open(TICKER_CACHE_FILE, "w") as fh:
                json.dump(ticker_cache, fh)

    with open(TICKER_CACHE_FILE, "w") as fh:
        json.dump(ticker_cache, fh)

    print(f"Resolved {len(resolved)} / {len(filings)} filings to tickers")
    return resolved


def get_market_cap_at_date(ticker, filing_date):
    ## price * shares outstanding at filing date
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        if not shares or shares <= 0:
            return None

        dt = datetime.strptime(filing_date, "%Y-%m-%d")
        start = (dt - timedelta(days=5)).strftime("%Y-%m-%d")
        end = (dt + timedelta(days=2)).strftime("%Y-%m-%d")

        hist = stock.history(start=start, end=end, auto_adjust=True)
        if hist.empty:
            return None

        price = float(hist["Close"].iloc[-1])
        return shares * price

    except Exception:
        return None


def get_event_returns(ticker, filing_date, window):
    ## market-adjusted returns over the event window

    try:
        dt = datetime.strptime(filing_date, "%Y-%m-%d")
        fetch_start = (dt - timedelta(days=10)).strftime("%Y-%m-%d")
        fetch_end = (dt + timedelta(days=window + 15)).strftime("%Y-%m-%d")

        raw = yf.download(
            [ticker, "SPY"],
            start=fetch_start,
            end=fetch_end,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            return None

        try:
            stock_px = raw["Close"][ticker].dropna()
            market_px = raw["Close"]["SPY"].dropna()
        except KeyError:
            return None

        stock_rets = stock_px.pct_change().dropna()
        market_rets = market_px.pct_change().dropna()

        filing_ts = pd.Timestamp(filing_date)

        ev_stock = stock_rets[stock_rets.index > filing_ts].head(window)
        ev_market = market_rets[market_rets.index > filing_ts].head(window)

        common = ev_stock.index.intersection(ev_market.index)
        if len(common) < 3:
            return None

        ev_stock = ev_stock[common]
        ev_market = ev_market[common]

        raw_ret = float((1 + ev_stock).prod() - 1)
        mkt_ret = float((1 + ev_market).prod() - 1)
        abnormal = raw_ret - mkt_ret

        return {
            "raw_return": raw_ret,
            "market_return": mkt_ret,
            "abnormal_return": abnormal,
            "n_days": len(common),
        }

    except Exception:
        return None


def run_event_study(filings):

    print(f"\nRunning event study on {len(filings)} filings...")
    print("This takes several minutes due to price data downloads.\n")

    rows, failed = [], 0

    for i, filing in enumerate(filings):

        ticker = filing["ticker"]
        date = filing["filing_date"]
        company = filing["company_name"]

        mktcap = get_market_cap_at_date(ticker, date)
        if mktcap is None:
            failed += 1
            continue

        rets = get_event_returns(ticker, date, EVENT_WINDOW_DAYS)
        if rets is None:
            failed += 1
            continue

        rows.append({
            "ticker": ticker,
            "company_name": company,
            "filing_date": date,
            "market_cap": mktcap,
            "market_cap_bn": round(mktcap / 1e9, 2),
            "is_small_cap": mktcap < SMALL_CAP_THRESHOLD,
            "raw_return": rets["raw_return"],
            "market_return": rets["market_return"],
            "abnormal_return": rets["abnormal_return"],
            "n_days": rets["n_days"],
        })

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(filings)} | success: {len(rows)} | failed: {failed}")

        time.sleep(0.1)

    df = pd.DataFrame(rows)
    print(f"\nEvent study done: {len(df)} events, {failed} failed")
    return df


def analyse(df):

    if df.empty:
        print("No data to analyse.")
        return {}

    small = df[df["is_small_cap"] == True]["abnormal_return"]
    large = df[df["is_small_cap"] == False]["abnormal_return"]
    all_r = df["abnormal_return"]

    print(f"\nFull sample (n={len(all_r)})")
    t_all, p_all = stats.ttest_1samp(all_r, 0)
    print(f"  Mean CAR: {all_r.mean()*100:.2f}%")
    print(f"  t={t_all:.2f}, p={p_all:.4f}")

    t_s, p_s = (0.0, 1.0)
    if len(small) > 1:
        t_s, p_s = stats.ttest_1samp(small, 0)
        match = abs(small.mean() - (-0.0749)) < 0.05
        print(f"\nSmall cap <$2B (n={len(small)})")
        print(f"  Mean CAR: {small.mean()*100:.2f}%")
        print(f"  t={t_s:.2f}, p={p_s:.4f}")
        print(f"  Paper target: -7.49%, t=-3.13, p=0.0019")
        print(f"  Replication: {'CLOSE (within 5pp)' if match else 'DIVERGES'}")

    t_l, p_l = (0.0, 1.0)
    if len(large) > 1:
        t_l, p_l = stats.ttest_1samp(large, 0)
        print(f"\nLarge cap >=$2B (n={len(large)})")
        print(f"  Mean CAR: {large.mean()*100:.2f}%")
        print(f"  t={t_l:.2f}, p={p_l:.4f}")
        print(f"  Paper target: +0.43%")

    if len(small) > 1 and len(large) > 1:
        t_diff, p_diff = stats.ttest_ind(small, large)
        print(f"\nSmall vs Large  t={t_diff:.2f}, p={p_diff:.4f}")

    ## short strategy on small caps
    print(f"\nStrategy: short small caps on disclosure, cover after {EVENT_WINDOW_DAYS}d")

    strategy_rets = -small

    if len(strategy_rets) > 0:
        mean_t = strategy_rets.mean()
        win_rate = (strategy_rets > 0).mean()
        gross = strategy_rets.sum()
        sharpe = (mean_t / strategy_rets.std() * np.sqrt(len(strategy_rets))
                  if strategy_rets.std() > 0 else 0.0)

        print(f"  Trades: {len(strategy_rets)}")
        print(f"  Mean/trade: {mean_t*100:.2f}%")
        print(f"  Win rate: {win_rate*100:.1f}%")
        print(f"  Gross return: {gross*100:.2f}% (sum, not compounded)")
        print(f"  Sharpe: {sharpe:.2f}")
    else:
        mean_t = win_rate = gross = sharpe = 0.0

    return {
        "full_sample": {
            "n": len(all_r),
            "mean_car": float(all_r.mean()),
            "t": float(t_all),
            "p": float(p_all),
        },
        "small_cap": {
            "n": len(small),
            "mean_car": float(small.mean()) if len(small) > 0 else None,
            "t": float(t_s),
            "p": float(p_s),
            "paper_target": -0.0749,
        },
        "large_cap": {
            "n": len(large),
            "mean_car": float(large.mean()) if len(large) > 0 else None,
            "t": float(t_l),
            "p": float(p_l),
            "paper_target": 0.0043,
        },
        "strategy": {
            "trades": len(strategy_rets),
            "mean_trade": float(mean_t),
            "win_rate": float(win_rate),
            "gross_return": float(gross),
            "sharpe": float(sharpe),
        },
    }


def main():

    print("Cybersecurity disclosure event study")
    print(f"Replicating arXiv:2512.06144v1\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    end_date = datetime.now().strftime("%Y-%m-%d")

    filings = fetch_edgar_filings(START_DATE, end_date, use_cache=True)
    if not filings:
        print("No filings found.")
        return

    filings_with_tickers = resolve_tickers(filings)
    if not filings_with_tickers:
        print("Could not resolve any tickers.")
        return

    df = run_event_study(filings_with_tickers)
    if df.empty:
        print("Event study returned no results.")
        return

    df.to_csv(EVENTS_FILE, index=False)
    print(f"\nEvents CSV: {EVENTS_FILE}")

    results = analyse(df)

    results["metadata"] = {
        "run_date": datetime.now().isoformat(),
        "start_date": START_DATE,
        "end_date": end_date,
        "filings_total": len(filings),
        "resolved": len(filings_with_tickers),
        "analysed": len(df),
        "paper": "arXiv:2512.06144v1",
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results JSON: {RESULTS_FILE}")
    print("\nDone.")


if __name__ == "__main__":
    Path("cache").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    main()
