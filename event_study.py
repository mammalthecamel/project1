"""Run the event study.  python event_study.py   (from the Project 1 folder)
Needs clean_ftse100_subset.csv and clean_benchmark_ftse100.csv beside it, and calculations.py.
Read it top to bottom: load -> beta table -> events -> per-event model -> results -> plots.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calculations as c

DIV_FILE = "ex_dividend_dates.csv"     # written by dividend_check.py


# ---- 1. load ----------------------------------------------------------------
def load_data():
    stocks = pd.read_csv("clean_ftse100_subset.csv", parse_dates=["date"])
    ftse = (pd.read_csv("clean_benchmark_ftse100.csv", parse_dates=["date"])
              .set_index("date")["close"].sort_index())
    # everything on the FTSE calendar: a missing stock day is NaN, never a 2-day return
    px = {f: stocks.pivot(index="date", columns="ticker", values=f).reindex(ftse.index)
          for f in ("open", "high", "low", "close")}
    ret = c.log_returns(px["close"])
    # Kaggle closes are raw prices: add the dividend back on ex-dates -> total returns [FFJR p.3]
    if os.path.exists(DIV_FILE):
        divs = pd.read_csv(DIV_FILE, parse_dates=["ex_date"])
        ret = c.add_dividends(ret, px["close"], divs)
    else:
        print(f"note: {DIV_FILE} not found, using price returns without dividends")
    return px, ret, c.log_returns(ftse)


# ---- 2. one event at a time ---------------------------------------------------
def run_events(px, ret, mkt, events):
    dates = ret.index
    rows, paths = [], {}
    for k, ev in events.reset_index(drop=True).iterrows():
        t = ev["ticker"]
        pos = c.event_position(dates, ev["date"], bool(ev.get("after_close", False)))
        if pos is None:
            continue
        sl = c.estimation_slice(pos)
        need = max(b for _, b in c.WINDOWS + [c.PRE] + [(0, c.SHOW)])
        if sl.start < 0 or pos + need >= len(dates):
            print(f"skip {t} {ev['date']}: not enough data around it"); continue

        y, x = ret[t].values, mkt.values
        fit = c.fit_market_model(y[sl], x[sl])          # estimation window only
        ar = c.abnormal_returns(y, x, fit)              # AR for every day

        row = dict(event=k, ticker=t, date=dates[pos], type=ev.get("type", ""),
                   alpha=fit["alpha"], beta=fit["beta"], sigma=fit["sigma"], r2=fit["r2"],
                   ar0=ar[pos], sar0=ar[pos] / fit["sigma"],
                   flag_2sigma=abs(ar[pos]) > 2 * fit["sigma"])
        for a, b in c.WINDOWS + [c.PRE]:
            cv = c.car(ar, pos, a, b)
            row[f"car_{a}_{b}"] = cv
            row[f"t_{a}_{b}"] = c.t_simple(cv, fit["sigma"], a, b)
            row[f"tfull_{a}_{b}"] = c.t_full(cv, x, pos, a, b, fit)
        row.update(c.volatility_measures(px["open"][t].reindex(dates), px["high"][t].reindex(dates),
                                         px["low"][t].reindex(dates), px["close"][t].reindex(dates), pos, sl))
        rows.append(row)
        paths[k] = pd.Series(ar[pos - c.SHOW: pos + c.SHOW + 1], index=range(-c.SHOW, c.SHOW + 1))
    return pd.DataFrame(rows), paths


# ---- 3. plots -----------------------------------------------------------------
def plot_ar_path(paths, k, title=""):
    s = paths[k]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(s.index, s.values * 100, color=["#D62728" if i == 0 else "#4C78A8" for i in s.index])
    ax.plot(s.index, s.cumsum().values * 100, color="black", lw=1.8, label="cumulative AR (from day -10)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Day relative to announcement"); ax.set_ylabel("Abnormal return (%)")
    ax.set_title(title or f"Event {k}"); ax.legend(frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout(); plt.show()


def plot_average_car(paths):
    mean, lo, hi = c.average_car_path(paths)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(mean.index, mean * 100, color="#D62728", lw=2.2, label=f"average CAR ({len(paths)} events)")
    ax.fill_between(mean.index, lo * 100, hi * 100, color="#D62728", alpha=0.15)
    ax.axhline(0, color="grey", lw=0.8); ax.axvline(0, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("Day relative to announcement"); ax.set_ylabel("Cumulative abnormal return (%)")
    ax.legend(frameon=False); ax.grid(alpha=0.25); fig.tight_layout(); plt.show()


# ---- 4. run -------------------------------------------------------------------
if __name__ == "__main__":
    px, ret, mkt = load_data()
    print("\nFull-sample market model per stock (check: HSBA beta ~1.42, BP. ~0.62):")
    print(c.beta_table(ret, mkt).round(4))

    # PLACEHOLDER events: the biggest one-day moves, picked BECAUSE they moved, so only for
    # testing the code. Never test hypotheses on events chosen by their return.
    EVENTS = pd.DataFrame([
        ("BRBY", "2024-11-14", "unlabelled", False),
        ("JD.",  "2024-11-21", "unlabelled", False),
        ("RR.",  "2025-02-27", "unlabelled", False),
        ("BRBY", "2025-05-14", "unlabelled", False),
        ("AUTO", "2025-05-29", "unlabelled", False),
        ("AZN",  "2025-10-01", "unlabelled", False),
        ("JD.",  "2026-08-20", "unlabelled", False),
    ], columns=["ticker", "date", "type", "after_close"])
    EVENTS["date"] = pd.to_datetime(EVENTS["date"])

    res, paths = run_events(px, ret, mkt, EVENTS)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
    print("\nEvent results:")
    print(res.assign(date=res["date"].dt.date).round(4).to_string())

    plot_ar_path(paths, 0, "BRBY 2024-11-14")
    plot_average_car(paths)
    # res.to_csv("event_study_results.csv", index=False)   # uncomment to hand the table to R
