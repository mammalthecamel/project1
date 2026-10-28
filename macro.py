"""Macro announcements (CPI, unemployment, Bank Rate) and the FTSE 100.  python macro.py
Needs: macro_events_audited.csv, clean_ftse100_subset.csv, clean_benchmark_ftse100.csv,
calculations.py and event_study.py in the same folder.

Two questions (macro news moves the whole market, so the market model alone would hide it):
  Q1  Did the MARKET react?  FTSE return on announcement days vs its normal level
      (constant-mean model, [MK97] sec 4A, eq (1)), and FTSE return against the surprise.
  Q2  Did some STOCKS react more than the market?  Market-model AR per stock ([MK97] eq (7)),
      averaged over the 10 stocks with the cross-correlation correction of
      Kolari & Pynnonen (2010) eq (11), because all 10 stocks share the same event day.
Window: day 0 only. Macro events are often 1 day apart (CPI then BoE), so longer windows overlap.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calculations as c
from event_study import load_data

MACRO_FILE = "macro_events_audited.csv"
COLOURS = {"cpi": "#D62728", "unemployment": "#4C78A8", "bank_rate": "#2CA02C"}


# ---- 1. load the events --------------------------------------------------------
def load_macro(dates):
    ev = pd.read_csv(MACRO_FILE, parse_dates=["announcement_date"])
    ev["pos"] = [c.event_position(dates, d) for d in ev["announcement_date"]]   # 07:00/12:00 -> same day
    ev = ev.dropna(subset=["pos"]).astype({"pos": int})
    # two releases on the same day cannot be told apart: keep them but flag them
    ev["same_day"] = ev.duplicated("pos", keep=False)
    # MPC vote split -> share of members AGAINST the decision (0 = unanimous, 0.44 = 5-4)
    v = ev["vote_split"].str.split("-", expand=True).astype(float)
    ev["dissent"] = v[1] / (v[0] + v[1])
    return ev.reset_index(drop=True)


# ---- 2. Q1: the market as a whole --------------------------------------------------
def market_reaction(mkt, ev):
    """Constant-mean model for the FTSE: R_m = mu + e, mu and sigma from the 120-day estimation
    window ([MK97] sec 4A, eq (1)). z = (R_m,0 - mu) / sigma. |z| measures size whatever the direction."""
    m = mkt.values
    rows = []
    for _, e in ev.iterrows():
        sl = c.estimation_slice(e["pos"])
        if sl.start < 0:
            continue
        mu, sd = m[sl].mean(), m[sl].std(ddof=1)
        rows.append(dict(date=mkt.index[e["pos"]], event_type=e["event_type"], same_day=e["same_day"],
                         surprise_pp=e["surprise_pp"], dissent=e["dissent"], action=e["action"],
                         ftse_ret=m[e["pos"]], ftse_z=(m[e["pos"]] - mu) / sd))
    return pd.DataFrame(rows)


def non_event_z(mkt, ev):
    """The same z on every NON-announcement day: the comparison group for Q1."""
    m, out = mkt.values, []
    event_days = set(ev["pos"])
    for p in range(len(m)):
        sl = c.estimation_slice(p)
        if sl.start < 0 or p in event_days:
            continue
        out.append((m[p] - m[sl].mean()) / m[sl].std(ddof=1))
    return np.array(out)


# ---- 3. Q2: the 10 stocks relative to the market ----------------------------------------
def stock_reaction(ret, mkt, ev):
    """Market-model AR and SAR on day 0 for each stock and event, plus the average over stocks
    with the Kolari-Pynnonen correction: t_adj = t * sqrt((1 - r) / (1 + (n - 1) r)),
    r = mean pairwise correlation of estimation-window residuals (KP 2010 eq (11))."""
    y_all, x = ret.values, mkt.values
    rows, summary = [], []
    for _, e in ev.iterrows():
        pos, sl = e["pos"], c.estimation_slice(e["pos"])
        if sl.start < 0:
            continue
        sars, resid = [], []
        for j, t in enumerate(ret.columns):
            fit = c.fit_market_model(y_all[sl, j], x[sl])
            ar = c.abnormal_returns(y_all[:, j], x, fit)
            sars.append(ar[pos] / fit["sigma"])
            resid.append(ar[sl] / fit["sigma"])
            rows.append(dict(date=ret.index[pos], event_type=e["event_type"], ticker=t,
                             beta=fit["beta"], ar0=ar[pos], sar0=ar[pos] / fit["sigma"]))
        n = len(sars)
        R = np.corrcoef(np.array(resid))
        r_bar = (R.sum() - n) / (n * (n - 1))                 # mean off-diagonal correlation
        # our 10 stocks are a big part of the FTSE itself, so their market-model residuals are
        # slightly NEGATIVELY correlated by construction; the correction is only applied when r > 0
        r_bar = max(r_bar, 0.0)
        t_plain = np.mean(sars) * np.sqrt(n) / np.std(sars, ddof=1)   # BMP-style cross-sectional t
        summary.append(dict(date=ret.index[pos], event_type=e["event_type"], mean_sar=np.mean(sars),
                            r_bar=r_bar, t_bmp=t_plain,
                            t_kp=t_plain * np.sqrt((1 - r_bar) / (1 + (n - 1) * r_bar))))
    return pd.DataFrame(rows), pd.DataFrame(summary)


# ---- 4. plots ---------------------------------------------------------------------------------
def plot_market_boxplot(mr, z_normal):
    groups = [np.abs(z_normal)] + [mr.loc[mr.event_type == k, "ftse_z"].abs().values for k in COLOURS]
    labels = ["no announcement"] + [f"{k} (n={len(g)})" for k, g in zip(COLOURS, groups[1:])]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(groups, showfliers=True)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("|FTSE return| in standard deviations")
    ax.set_title("Is the FTSE 100 more volatile on macro announcement days?")
    ax.grid(alpha=0.25, axis="y"); fig.tight_layout(); plt.show()


def plot_surprise(mr, kind):
    d = mr[(mr.event_type == kind) & mr.surprise_pp.notna()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(d.surprise_pp, d.ftse_ret * 100, color=COLOURS[kind])
    if d.surprise_pp.nunique() > 1:
        b, a = np.polyfit(d.surprise_pp, d.ftse_ret * 100, 1)
        xs = np.linspace(d.surprise_pp.min(), d.surprise_pp.max(), 2)
        ax.plot(xs, a + b * xs, color="black", lw=1.2, label=f"slope {b:.2f}% per pp")
        ax.legend(frameon=False)
    ax.axhline(0, color="grey", lw=0.8); ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel(f"{kind} surprise (actual - consensus, percentage points)")
    ax.set_ylabel("FTSE 100 return on the day (%)")
    ax.set_title(f"FTSE 100 reaction vs {kind} surprise (n={len(d)})")
    ax.grid(alpha=0.25); fig.tight_layout(); plt.show()


def plot_stock_sensitivity(sr):
    t = sr.groupby(["ticker", "event_type"]).sar0.apply(lambda s: s.abs().mean()).unstack()
    t = t[list(COLOURS)].sort_values("bank_rate")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.27
    for i, k in enumerate(COLOURS):
        ax.bar(np.arange(len(t)) + (i - 1) * w, t[k], w, color=COLOURS[k], label=k)
    ax.axhline(np.sqrt(2 / np.pi), color="grey", ls="--", lw=1, label="normal day (|z| mean = 0.80)")
    ax.set_xticks(range(len(t)), t.index)
    ax.set_ylabel("mean |SAR| on announcement day")
    ax.set_title("Which stocks react beyond the market on macro days?")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25, axis="y"); fig.tight_layout(); plt.show()


# ---- 5. run ---------------------------------------------------------------------------------
if __name__ == "__main__":
    px, ret, mkt = load_data()
    ev = load_macro(ret.index)
    print(f"{len(ev)} macro events in the price data; same-day clashes: "
          f"{sorted(set(ev.loc[ev.same_day, 'announcement_date'].dt.date))}")

    mr = market_reaction(mkt, ev)
    z0 = non_event_z(mkt, ev)
    print("\nQ1  mean |z| of FTSE return (normal days should be about 0.80):")
    print(f"   no announcement  {np.abs(z0).mean():.3f}  (n={len(z0)})")
    print(mr.groupby("event_type").ftse_z.agg(mean_abs_z=lambda s: s.abs().mean(), n="size").round(3))

    for kind in ("cpi", "unemployment"):
        d = mr[(mr.event_type == kind) & mr.surprise_pp.notna()]
        print(f"   corr(FTSE return, {kind} surprise) = {d.ftse_ret.corr(d.surprise_pp):.3f}  (n={len(d)})")
    b = mr[mr.event_type == "bank_rate"]
    print(f"   corr(|FTSE z|, MPC dissent share) = {b.ftse_z.abs().corr(b.dissent):.3f}  (n={len(b)})")

    sr, ss = stock_reaction(ret, mkt, ev)
    print("\nQ2  average over the 10 stocks, day 0 (t_kp = Kolari-Pynnonen corrected):")
    print(ss.assign(date=ss.date.dt.date).round(3).to_string())

    plot_market_boxplot(mr, z0)
    plot_surprise(mr, "cpi")
    plot_surprise(mr, "unemployment")
    plot_stock_sensitivity(sr)

    # mr.to_csv("macro_market_results.csv", index=False)   # uncomment to hand to R
    # sr.to_csv("macro_stock_results.csv", index=False)
