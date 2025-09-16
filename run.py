# sp500_ev_ebitda_model.py
import time
import math
import requests
import pandas as pd
from tqdm import tqdm
import yfinance as yf

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_sp500_tickers():
    r = requests.get(WIKI_SP500_URL, headers=HEADERS, timeout=20)
    df_list = pd.read_html(r.text)
    # first table on the page is the constituents
    df = df_list[0]
    df = df.rename(columns=lambda s: s.strip())
    # columns include 'Symbol', 'Security', 'GICS Sector' (or similar)
    tickers = df["Symbol"].tolist()
    sectors = df[["Symbol", "GICS Sector"]].set_index("Symbol")["GICS Sector"].to_dict()
    return tickers, sectors


def safe_get(info, key):
    try:
        v = info.get(key, None)
        if v is None:
            return None
        # sometimes yfinance returns numpy types
        return float(v)
    except Exception:
        return None


def fetch_ticker_data(ticker, sector_map):
    t = yf.Ticker(ticker)
    info = t.info if hasattr(t, "info") else {}
    # yfinance sometimes has 'ebitda' key on info
    ebitda = safe_get(info, "ebitda")
    market_cap = safe_get(info, "marketCap")
    total_debt = safe_get(info, "totalDebt") or safe_get(info, "totalDebt")
    cash = (
        safe_get(info, "totalCash")
        or safe_get(info, "cash")
        or safe_get(info, "totalCash")
    )
    shares = safe_get(info, "sharesOutstanding")
    price = safe_get(info, "regularMarketPrice") or (
        t.history(period="1d")["Close"][-1] if len(t.history(period="1d")) > 0 else None
    )

    # fallback: try to compute EBITDA from financials if info missing
    if ebitda is None:
        try:
            # yfinance Ticker.quarterly_financials/financials sometimes contain 'EBITDA' index; try to read
            fin = t.financials
            if "Ebitda" in fin.index or "EBITDA" in fin.index:
                idx = "Ebitda" if "Ebitda" in fin.index else "EBITDA"
                ebitda = float(fin.loc[idx].iloc[0])
        except Exception:
            ebitda = None

    # net debt
    net_debt = None
    if (total_debt is not None) and (cash is not None):
        net_debt = total_debt - cash
    elif total_debt is not None:
        net_debt = total_debt

    sector = sector_map.get(ticker, info.get("sector", None))

    return {
        "ticker": ticker,
        "sector": sector,
        "ebitda": ebitda,
        "market_cap": market_cap,
        "total_debt": total_debt,
        "cash": cash,
        "net_debt": net_debt,
        "shares_outstanding": shares,
        "price": price,
    }


def compute_ev(row):
    if pd.isna(row.market_cap) and pd.isna(row.total_debt):
        return None
    mc = row.market_cap or 0.0
    td = row.total_debt or 0.0
    ca = row.cash or 0.0
    return mc + td - ca


def main(out_csv="results.csv", pause=0.2):
    print("Fetching S&P 500 tickers from Wikipedia...")
    tickers, sector_map = get_sp500_tickers()
    print(
        f"Found {len(tickers)} tickers. Fetching data (this may take several minutes)..."
    )

    rows = []
    # iterate with progress bar and polite pause
    for ticker in tqdm(tickers):
        try:
            data = fetch_ticker_data(ticker, sector_map)
            rows.append(data)
        except Exception as e:
            print("Error fetching", ticker, e)
        time.sleep(pause)  # politeness/rate-limit

    df = pd.DataFrame(rows)
    # compute EV and EV/EBITDA where possible
    df["ev"] = df.apply(lambda r: compute_ev(r), axis=1)
    df["ev_to_ebitda"] = df.apply(
        lambda r: (
            (r["ev"] / r["ebitda"])
            if (r["ev"] is not None and r["ebitda"] not in (None, 0))
            else None
        ),
        axis=1,
    )

    # compute sector median EV/EBITDA (ignore outliers and NaNs)
    # filter for reasonable EV/EBITDA values (e.g., 0 < x < 50) to avoid crazy outliers
    df["ev_to_ebitda_clipped"] = df["ev_to_ebitda"].apply(
        lambda x: x if (x is not None and x > 0 and x < 50) else None
    )
    sector_medians = df.groupby("sector")["ev_to_ebitda_clipped"].median().to_dict()

    # fallback overall median if sector median missing
    overall_median = df["ev_to_ebitda_clipped"].median()
    print("Overall median EV/EBITDA (used as fallback):", overall_median)

    # apply sector median to compute implied price
    implied_prices = []
    implied_ev_list = []
    pct_diff_list = []
    for _, r in df.iterrows():
        sector = r["sector"]
        ebitda = r["ebitda"]
        shares = r["shares_outstanding"]
        net_debt = r["net_debt"]
        market_price = r["price"]

        multiple = sector_medians.get(sector, None)
        if multiple is None or math.isnan(multiple):
            multiple = overall_median

        if (
            ebitda in (None, 0)
            or shares in (None, 0)
            or multiple is None
            or pd.isna(multiple)
        ):
            implied_price = None
            implied_ev = None
            pct_diff = None
        else:
            implied_ev = ebitda * multiple
            implied_equity = implied_ev - (net_debt or 0.0)
            implied_price = implied_equity / shares
            if market_price is not None and market_price > 0:
                pct_diff = (implied_price - market_price) / market_price * 100.0
            else:
                pct_diff = None

        implied_prices.append(implied_price)
        implied_ev_list.append(implied_ev)
        pct_diff_list.append(pct_diff)

    df["sector_median_ev_to_ebitda"] = df["sector"].apply(
        lambda s: sector_medians.get(s, overall_median)
    )
    df["implied_ev"] = implied_ev_list
    df["implied_price"] = implied_prices
    df["pct_diff"] = pct_diff_list

    # classify undervalued/overvalued
    df["valuation_signal"] = df["pct_diff"].apply(
        lambda x: (
            "undervalued"
            if (x is not None and x > 0)
            else ("overvalued" if (x is not None and x < 0) else "unknown")
        )
    )

    # Save results
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print("Saved results to", out_csv)

    # Show top undervalued and overvalued by percent difference (filter unknowns)
    good = df.dropna(subset=["pct_diff"])
    top_undervalued = good.sort_values("pct_diff", ascending=False).head(20)
    top_overvalued = good.sort_values("pct_diff", ascending=True).head(20)

    print("\nTop 10 undervalued (implied > market):")
    display_cols = [
        "ticker",
        "sector",
        "price",
        "implied_price",
        "pct_diff",
        "ebitda",
        "market_cap",
        "net_debt",
        "sector_median_ev_to_ebitda",
    ]
    print(top_undervalued[display_cols].head(10).to_string(index=False))

    print("\nTop 10 overvalued (implied < market):")
    print(top_overvalued[display_cols].head(10).to_string(index=False))

    return df


if __name__ == "__main__":
    main()
