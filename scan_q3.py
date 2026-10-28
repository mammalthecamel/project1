"""Price first: find abnormal days, then ask whether announcements explain them.  python scan.py
Needs in the same folder: calculations.py, event_study.py, company.py, macro.py and their CSVs.

Steps
  1. For every stock and every day, fit the market model on the 120 days before it and compute
     SAR = AR / sigma  ([MK97] eqs (3)-(7)).  Abnormal RETURN day: |SAR| > 2.
     Abnormal VOLATILITY day: log(High/Low) more than 2x its estimation-window mean.
  2. Match each abnormal day to news: a company announcement for that stock on day -1, 0 or +1,
     or a macro release (CPI, unemployment, Bank Rate) on day 0.
  3. Compare with chance: what share of ALL days have news nearby? If abnormal days have news
     far more often than ordinary days, announcements explain abnormal returns.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import calculations as c
from event_study import load_data
from company import load_company_events
from macro import load_macro

THRESHOLD = 2.0          # |SAR| above this = abnormal return day
VOL_RATIO = 2.0          # range above this multiple of normal = abnormal volatility day
COMPANY_WINDOW = (-1, 1) # news the day before (leak) to the day after (late reaction)
COLOURS = {"company": "#D62728", "macro": "#2CA02C", "both": "#9467BD", "none": "#BBBBBB"}


# ---- 1. SAR and range ratio for every stock-day --------------------------------------------
def daily_sar(px, ret, mkt, tickers):
    dates, x = ret.index, mkt.values
    rows = []
    for t in tickers:
        y = ret[t].values
        rng = np.log(px["high"][t] / px["low"][t]).reindex(dates).values
        for pos in range(len(dates)):
            sl = c.estimation_slice(pos)
            if sl.start < 0 or np.isnan(y[pos]):
                continue
            fit = c.fit_market_model(y[sl], x[sl])
            ar = y[pos] - fit["alpha"] - fit["beta"] * x[pos]
            rows.append(dict(ticker=t, date=dates[pos], pos=pos, ar=ar, sar=ar / fit["sigma"],
                             range_ratio=rng[pos] / np.nanmean(rng[sl])))
    return pd.DataFrame(rows)


# ---- 2. attach news to every stock-day ------------------------------------------------------
def attach_news(d, comp, macro_pos, dates):
    """company = announcement for that stock within COMPANY_WINDOW; macro = release on the day.
    Each stock is only scanned between its first and last logged announcement, because outside
    that period the research list is empty and every big day would look 'unexplained'."""
    near = set()
    for _, e in comp.iterrows():
        p = c.event_position(dates, e["date"], bool(e.get("after_close", False)))
        if p is not None:
            near |= {(e["ticker"], p + k) for k in range(COMPANY_WINDOW[0], COMPANY_WINDOW[1] + 1)}
    d["company"] = [(t, p) in near for t, p in zip(d.ticker, d.pos)]
    d["macro"] = d.pos.isin(macro_pos)
    d["news"] = np.select([d.company & d.macro, d.company, d.macro], ["both", "company", "macro"], "none")
    span = comp.groupby("ticker").date.agg(["min", "max"])
    keep = [span.at[t, "min"] - pd.Timedelta(days=1) <= dt <= span.at[t, "max"] + pd.Timedelta(days=1)
            for t, dt in zip(d.ticker, d.date)]
    return d[keep].reset_index(drop=True)


# ---- 3. abnormal days vs chance ---------------------------------------------------------------
def explain(d, flag, name):
    """Share of abnormal days with news vs share of normal days with news.
    Fisher exact test on the 2x2 table (abnormal yes/no x news yes/no). Preliminary: R does the formal tests."""
    news = d.news != "none"
    tab = pd.crosstab(flag, news)
    odds, p = stats.fisher_exact(tab.values)
    print(f"\n{name}: {flag.sum()} abnormal stock-days out of {len(d)}")
    print(f"  share with news nearby:  abnormal days {news[flag].mean():.1%}   normal days {news[~flag].mean():.1%}")
    print(f"  odds ratio {odds:.2f}, Fisher exact p = {p:.2g}")
    print("  abnormal days by explanation:", d.loc[flag, "news"].value_counts().to_dict())
    return d.loc[flag, "news"].value_counts(normalize=True)


# ---- 4. plots -------------------------------------------------------------------------------
def plot_explained(d, flag, title):
    t = pd.crosstab(d.loc[flag, "ticker"], d.loc[flag, "news"]).reindex(columns=list(COLOURS), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(t))
    for k, col in COLOURS.items():
        ax.bar(t.index, t[k], bottom=bottom, color=col, label=k)
        bottom += t[k].values
    ax.set_ylabel("number of abnormal days"); ax.set_title(title)
    ax.legend(title="news nearby", frameon=False); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); plt.show()


def plot_timeline(d, ticker):
    s = d[d.ticker == ticker]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(s.date, s.sar, color="#999999", lw=0.7)
    for k, col in COLOURS.items():
        if k == "none":
            continue
        m = s[(s.news == k) & (s.sar.abs() > THRESHOLD)]
        ax.scatter(m.date, m.sar, color=col, s=28, zorder=3, label=f"abnormal, {k} news")
    m = s[(s.news == "none") & (s.sar.abs() > THRESHOLD)]
    ax.scatter(m.date, m.sar, facecolors="none", edgecolors="black", s=28, zorder=3, label="abnormal, no news")
    for h in (THRESHOLD, -THRESHOLD):
        ax.axhline(h, color="#D62728", ls="--", lw=0.8)
    ax.set_ylabel("SAR"); ax.set_title(f"{ticker}: daily standardised abnormal return")
    ax.legend(frameon=False, fontsize=8, ncol=4); ax.grid(alpha=0.25); fig.tight_layout(); plt.show()


# ---- 5. run -------------------------------------------------------------------------------
if __name__ == "__main__":
    px, ret, mkt = load_data()
    comp = load_company_events()
    macro_pos = set(load_macro(ret.index)["pos"])
    tickers = sorted(comp.ticker.unique())               # only stocks with a news list so far

    d = attach_news(daily_sar(px, ret, mkt, tickers), comp, macro_pos, ret.index)
    print(f"scanned {len(d)} stock-days for {tickers}")

    r = explain(d, d.sar.abs() > THRESHOLD, f"Abnormal RETURNS (|SAR| > {THRESHOLD})")
    v = explain(d, d.range_ratio > VOL_RATIO, f"Abnormal VOLATILITY (range > {VOL_RATIO}x normal)")

    big = d[(d.sar.abs() > THRESHOLD) & (d.news == "none")].sort_values("sar", key=abs, ascending=False)
    print("\nBiggest UNEXPLAINED abnormal days (send to the research team to check for missed news):")
    print(big[["ticker", "date", "ar", "sar"]].head(15).assign(date=big.date.dt.date).round(3).to_string(index=False))

    plot_explained(d, d.sar.abs() > THRESHOLD, f"Abnormal return days (|SAR| > {THRESHOLD}) by news nearby")
    plot_explained(d, d.range_ratio > VOL_RATIO, f"Abnormal volatility days (range > {VOL_RATIO}x) by news nearby")
    plot_timeline(d, tickers[0])

    # d.to_csv("scan_results.csv", index=False)   # uncomment to hand to R
