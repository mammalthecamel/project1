"""Company announcements: load, clean, run the event study, compare labels.  python company.py
Needs in the same folder: the company events CSV, clean_ftse100_subset.csv,
clean_benchmark_ftse100.csv, calculations.py, event_study.py.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import calculations as c
from event_study import load_data, run_events

COMPANY_FILE = "company data 5 companies.csv"
TICKERS = {"AstraZeneca": "AZN", "HSBC": "HSBA", "Shell": "SHEL", "Rolls-Royce": "RR.",
           "Unilever": "ULVR", "Rio Tinto": "RIO", "BP": "BP.", "JD Sports": "JD.",
           "Auto Trader": "AUTO", "Burberry": "BRBY"}
EXCLUDE = [("ULVR", "2025-12-08")]       # ice-cream demerger: mechanical price drop, not a reaction
LEAKS = [("AZN", "2025-10-13"), ("SHEL", "2025-06-26"), ("ULVR", "2026-03-20"), ("RR.", "2026-02-26")]
LABELS = {"significant": "#D62728", "moderate": "#FF7F0E", "not": "#4C78A8"}


# ---- 1. load and clean ---------------------------------------------------------------
def load_company_events(path=COMPANY_FILE):
    d = pd.read_csv(path)
    if "HEADLINE" not in d.columns:      # first batch: header row is one column short
        d = d.iloc[:, :9]
        d.columns = ["COMPANY", "DATE", "RNS_CATEGORY", "EVENT_TYPE", "LABEL", "HEADLINE",
                     "SOURCE_URL", "COLLECTOR", "NOTES"]
    d.columns = d.columns.str.lower()
    d["ticker"] = d["company"].map(TICKERS)
    d["date"] = pd.to_datetime(d["date"], dayfirst=True)          # UK format 12/11/2024
    d["after_close"] = False                                      # RNS at 07:00 unless told otherwise
    key = list(zip(d["ticker"], d["date"].dt.strftime("%Y-%m-%d")))
    d["leak"] = [k in LEAKS for k in key]
    d = d[[k not in EXCLUDE for k in key]]
    missing = d["ticker"].isna()
    if missing.any():
        print("unknown company names:", d.loc[missing, "company"].unique())
    return d[~missing].reset_index(drop=True)


# ---- 2. run the event study and attach the labels ------------------------------------
def company_results(px, ret, mkt, ev):
    res, paths = run_events(px, ret, mkt, ev.rename(columns={"event_type": "type"}))
    res = res.merge(ev[["label", "leak", "headline"]], left_on="event", right_index=True)
    sar = {k: p / res.loc[res.event == k, "sigma"].iloc[0] for k, p in paths.items()}
    return res, sar


# ---- 3. first look at the hypothesis: does the size of the reaction depend on the label? ----
def compare_labels(res):
    """|SAR| on day 0 by label. Under no reaction |SAR| has mean sqrt(2/pi) = 0.80.
    Kruskal-Wallis = do the three labels differ? (rank test, no normality assumed).
    Preliminary only: the formal tests are done in R."""
    res = res.assign(abs_sar0=res.sar0.abs())
    table = res.groupby("label").agg(n=("abs_sar0", "size"), mean_abs_sar0=("abs_sar0", "mean"),
                                     median_abs_sar0=("abs_sar0", "median"),
                                     share_2sigma=("flag_2sigma", "mean"), mean_av0=("av0", "mean"))
    groups = [g.values for _, g in res.groupby("label").abs_sar0 if len(g) > 1]
    kw = stats.kruskal(*groups) if len(groups) > 1 else None
    return table.reindex([l for l in LABELS if l in table.index]), kw


# ---- 4. plots ------------------------------------------------------------------------------
def plot_abs_sar_by_day(res, sar):
    """Mean |SAR| per event day for each label: a spike at day 0 = reaction; a rise before 0 =
    anticipation or leak. Grey line = the level expected on a normal day (0.80)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for lab, col in LABELS.items():
        ks = res.loc[res.label == lab, "event"]
        if len(ks) == 0:
            continue
        m = pd.DataFrame({k: sar[k] for k in ks}).abs().mean(axis=1)
        ax.plot(m.index, m.values, marker="o", ms=3, color=col, label=f"{lab} (n={len(ks)})")
    ax.axhline(np.sqrt(2 / np.pi), color="grey", ls="--", lw=1, label="normal day (0.80)")
    ax.axvline(0, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("Day relative to announcement"); ax.set_ylabel("mean |SAR|")
    ax.set_title("Size of the abnormal return around company announcements, by label")
    ax.legend(frameon=False); ax.grid(alpha=0.25); fig.tight_layout(); plt.show()


def plot_box_by_label(res):
    labs = [l for l in LABELS if l in set(res.label)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([res.loc[res.label == l, "sar0"].abs() for l in labs])
    ax.set_xticks(range(1, len(labs) + 1), [f"{l}\n(n={(res.label == l).sum()})" for l in labs])
    ax.axhline(2, color="#D62728", ls="--", lw=1, label="2 sigma")
    ax.set_ylabel("|SAR| on announcement day"); ax.set_title("Day-0 reaction by label")
    ax.legend(frameon=False); ax.grid(alpha=0.25, axis="y"); fig.tight_layout(); plt.show()


# ---- 5. run ---------------------------------------------------------------------------------
if __name__ == "__main__":
    px, ret, mkt = load_data()
    ev = load_company_events()
    res, sar = company_results(px, ret, mkt, ev)
    print(f"{len(res)} of {len(ev)} events have enough data around them")

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
    cols = ["ticker", "date", "type", "label", "leak", "beta", "ar0", "sar0", "car_-1_1",
            "t_-1_1", "car_0_5", "car_-5_-1", "av0", "gap0_rel"]
    print(res[cols].assign(date=res.date.dt.date).round(3).to_string())

    table, kw = compare_labels(res)
    print("\nDay-0 reaction by label (|SAR| is 0.80 on an average normal day):")
    print(table.round(3))
    if kw is not None:
        print(f"Kruskal-Wallis across labels: H = {kw.statistic:.2f}, p = {kw.pvalue:.4f}  (preliminary)")

    plot_abs_sar_by_day(res, sar)
    plot_box_by_label(res)

    # res.to_csv("company_results.csv", index=False)   # uncomment to hand the table to R
