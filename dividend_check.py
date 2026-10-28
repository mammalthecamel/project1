"""Ex-dividend check.  python dividend_check.py   (needs internet for yfinance)
Answers two questions:
  1. Are the Kaggle closes adjusted for dividends?  Compare them with Yahoo's raw Close and Adj Close.
  2. Does any event window contain an ex-dividend date?  (ex-dividend price drops look like bad news)
"""
import numpy as np
import pandas as pd
import yfinance as yf
import calculations as c
from event_study import load_data
from company import load_company_events

YAHOO = {"HSBA": "HSBA.L", "SHEL": "SHEL.L", "AZN": "AZN.L", "RR.": "RR.L", "ULVR": "ULVR.L",
         "RIO": "RIO.L", "BP.": "BP.L", "JD.": "JD.L", "AUTO": "AUTO.L", "BRBY": "BRBY.L"}
WINDOW = (-5, 5)        # widest window we use: PRE (-5,-1) to CAR(0,5)


# ---- 1. ex-dividend dates from Yahoo ------------------------------------------------------
def ex_dividend_dates(start, end):
    rows = []
    for t, y in YAHOO.items():
        d = yf.Ticker(y).dividends
        d.index = d.index.tz_localize(None).normalize()
        d = d[(d.index >= start) & (d.index <= end)]
        rows += [dict(ticker=t, ex_date=k, dividend=v) for k, v in d.items()]
    return pd.DataFrame(rows)


# ---- 2. are the Kaggle closes raw or adjusted? ------------------------------------------------
def adjustment_check(px, start, end):
    """If Kaggle = raw prices, Kaggle/YahooClose is constant over time; if adjusted,
    Kaggle/YahooAdjClose is. Report the spread (max/min - 1) of each ratio per stock."""
    out = []
    for t, y in YAHOO.items():
        yd = yf.download(y, start=start, end=end + pd.Timedelta(days=1), auto_adjust=False,
                         progress=False)
        yd.columns = yd.columns.get_level_values(0)
        k = px["close"][t].dropna()
        both = pd.concat([k, yd["Close"], yd["Adj Close"]], axis=1, join="inner").dropna()
        both.columns = ["kaggle", "raw", "adj"]
        r_raw, r_adj = both.kaggle / both.raw, both.kaggle / both.adj
        out.append(dict(ticker=t, days=len(both),
                        spread_vs_raw=r_raw.max() / r_raw.min() - 1,
                        spread_vs_adj=r_adj.max() / r_adj.min() - 1))
    o = pd.DataFrame(out).set_index("ticker")
    return o


# ---- 3. which events have an ex-date inside their window? --------------------------------------
def overlaps(ev, exd, dates):
    rows = []
    for _, e in ev.iterrows():
        pos = c.event_position(dates, e["date"], bool(e.get("after_close", False)))
        if pos is None:
            continue
        lo, hi = dates[max(pos + WINDOW[0], 0)], dates[min(pos + WINDOW[1], len(dates) - 1)]
        hit = exd[(exd.ticker == e["ticker"]) & (exd.ex_date >= lo) & (exd.ex_date <= hi)]
        for _, h in hit.iterrows():
            rows.append(dict(ticker=e["ticker"], event_date=dates[pos].date(), label=e["label"],
                             ex_date=h.ex_date.date(),
                             day_in_window=int(dates.searchsorted(h.ex_date) - pos),
                             dividend=h.dividend, pct_of_price=h.dividend / e["close0"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    px, ret, mkt = load_data()
    dates = ret.index
    start, end = dates[0], dates[-1]

    exd = ex_dividend_dates(start, end)
    print(f"{len(exd)} ex-dividend dates found, per stock:")
    print(exd.groupby("ticker").size().to_string())

    print("\nAre the Kaggle closes adjusted for dividends? (smaller spread = the match)")
    print(adjustment_check(px, start, end).round(4))

    ev = load_company_events()
    ev["close0"] = [px["close"][t].asof(d) for t, d in zip(ev.ticker, ev.date)]
    ov = overlaps(ev, exd, dates)
    print(f"\nEvents with an ex-dividend date inside days {WINDOW}: {len(ov)}")
    if len(ov):
        print(ov.round(4).to_string(index=False))
        print("Only a problem when day_in_window falls inside the CAR you report "
              "(0 for CAR(0,0), -1..1 for CAR(-1,1), 0..5 for CAR(0,5)).")
    # decisive test: raw prices fall by ~the dividend on the ex-date, adjusted prices do not
    exd["kaggle_ret"] = [ret[t].get(d, np.nan) - mkt.get(d, np.nan) for t, d in zip(exd.ticker, exd.ex_date)]
    exd["div_yield"] = [v / px["close"][t].asof(d - pd.Timedelta(days=1)) for t, d, v in
                        zip(exd.ticker, exd.ex_date, exd.dividend)]
    print("\nOn ex-dates: mean market-adjusted Kaggle return vs mean dividend yield")
    print(exd.groupby("ticker")[["kaggle_ret", "div_yield"]].mean().round(4))
    print(f"ALL: return {exd.kaggle_ret.mean():.4f}  vs  yield {exd.div_yield.mean():.4f}")
    exd[["ticker", "ex_date", "dividend"]].to_csv("ex_dividend_dates.csv", index=False)
    print("saved ex_dividend_dates.csv (event_study.py uses it for total returns)")
