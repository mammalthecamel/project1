"""Event-study calculations. Pure functions: numbers in, numbers out. No file reading, no plotting.

Sources (page/equation numbers checked against the PDFs):
  [MK97]  MacKinlay (1997), Event Studies in Economics and Finance, J. Economic Literature 35(1)
  [FFJR]  Fama, Fisher, Jensen, Roll (1969), International Economic Review 10(1)
  [BW85]  Brown & Warner (1985), Using daily stock returns, J. Financial Economics 14
  [B98]   Binder (1998), The Event Study Methodology Since 1969, RQFA 11(2)
  [CLM97] Campbell, Lo & MacKinlay (1997), The Econometrics of Financial Markets, ch.4
"""
import numpy as np
import pandas as pd

# ---- settings (change here, nowhere else) ---------------------------------
EST_LEN = 120                         # L1: estimation window length, trading days
GAP = 5                               # days between estimation window and event window
WINDOWS = [(-1, 1), (0, 0), (0, 5)]   # event windows (tau1, tau2) for CAR
PRE = (-5, -1)                        # pre-announcement window (anticipation)
SHOW = 10                             # days either side kept for plots


# ---- 1. returns -------------------------------------------------------------
def log_returns(prices):
    """R_t = ln(P_t) - ln(P_t-1).  Log form: [FFJR] p.4 eq (1) (log returns are more
    symmetric and are continuously compounded). [BW85] find similar results with simple returns."""
    return np.log(prices).diff().iloc[1:]


def add_dividends(ret, close, divs):
    """Total return on ex-dividend days: R_t = ln((P_t + D_t) / P_t-1).
    [FFJR] p.3: R_jt = (P_jt + D_jt) / P_j,t-1, with the dividend D taken on the EX-dividend
    date, not the payment date. Every other day is unchanged (D_t = 0).
    ret: log returns, close: prices (same index), divs: columns ticker, ex_date, dividend."""
    out = ret.copy()
    prev = close.shift(1)
    for _, d in divs.iterrows():
        if d["ticker"] in out.columns and d["ex_date"] in out.index:
            p0, p1 = prev.at[d["ex_date"], d["ticker"]], close.at[d["ex_date"], d["ticker"]]
            out.at[d["ex_date"], d["ticker"]] = np.log((p1 + d["dividend"]) / p0)
    return out


# ---- 2. market model --------------------------------------------------------
def fit_market_model(y, x):
    """Fit R_i = alpha + beta*R_m + e on the ESTIMATION window.
    [MK97] eq (3): model, E(e)=0, var(e)=sigma^2
    [MK97] eq (4): beta  = sum((y-ybar)(x-xbar)) / sum((x-xbar)^2)   (= Cov/Var)
    [MK97] eq (5): alpha = ybar - beta*xbar
    [MK97] eq (6): sigma^2 = sum(e^2) / (L1 - 2)   (-2: two parameters estimated)
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    y, x = y[ok], x[ok]
    n = len(y)
    sxx = ((x - x.mean()) ** 2).sum()
    beta = ((y - y.mean()) * (x - x.mean())).sum() / sxx
    alpha = y.mean() - beta * x.mean()
    resid = y - alpha - beta * x
    sigma = np.sqrt(resid @ resid / (n - 2))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return dict(alpha=alpha, beta=beta, sigma=sigma, r2=r2, n=n, xbar=x.mean(), sxx=sxx)


def beta_table(ret, mkt):
    """Full-sample beta/alpha/sigma/R^2 per stock (descriptive table for the report)."""
    out = pd.DataFrame({t: fit_market_model(ret[t], mkt) for t in ret.columns}).T
    out = out[["beta", "alpha", "sigma", "r2", "n"]]
    out["mean_ret"], out["vol_ret"] = ret.mean(), ret.std()
    return out.sort_values("beta", ascending=False)


# ---- 3. abnormal returns ----------------------------------------------------
def abnormal_returns(y, x, fit):
    """AR_t = R_it - alpha - beta*R_mt   [MK97] eq (7)   (computed for every day)."""
    return np.asarray(y, float) - (fit["alpha"] + fit["beta"] * np.asarray(x, float))


def car(ar, pos, a, b):
    """CAR(a,b) = sum of AR from day a to day b around event day pos.   [MK97] eq (10)."""
    return ar[pos + a: pos + b + 1].sum()


def t_simple(car_value, sigma, a, b):
    """t = CAR / (sigma*sqrt(L2)), from var(CAR) = L2*sigma^2.   [MK97] eq (11), eq (20) (large-L1 form).
    [BW85] fn.4 report that the prediction-error adjustment made no detectable difference with daily data."""
    return car_value / (sigma * np.sqrt(b - a + 1))


def t_full(car_value, x, pos, a, b, fit):
    """t using the variance INCLUDING estimation error in alpha and beta.
    var(CAR) = sigma^2 * [ L2 + L2^2/L1 + (sum(Rm over window) - L2*xbar)^2 / Sxx ]
    Closed form of gamma' V gamma with V = I*sigma^2 + X*(X'X)^-1 X*' sigma^2:
    [CLM97] Campbell, Lo & MacKinlay (1997) ch.4 eqs (4.4.9) and (4.4.12) (checked numerically).
    One-day case is [MK97] eq (8) (as printed there the sigma^2 factor is missing).
    Matters for short estimation windows: Corrado (1989) fn.4; [B98] p.115 (Patell 1976)."""
    L2 = b - a + 1
    xs = np.asarray(x, float)[pos + a: pos + b + 1].sum() - L2 * fit["xbar"]
    var = fit["sigma"] ** 2 * (L2 + L2 ** 2 / fit["n"] + xs ** 2 / fit["sxx"])
    return car_value / np.sqrt(var)


# ---- 4. aggregation across events --------------------------------------------
def average_car_path(paths):
    """Mean cumulative AR in event time with a 95% band.   [MK97] eqs (13),(15),(17)-(19);
    [B98] eqs (2)-(4) (AAR, CAAR, sd(CAAR)). Assumes events do not cluster in calendar time:
    macro days break that (see [BW85] Table 8)."""
    M = pd.DataFrame(paths).cumsum()
    mean = M.mean(axis=1)
    se = M.std(axis=1, ddof=1) / np.sqrt(M.shape[1])
    return mean, mean - 1.96 * se, mean + 1.96 * se


# ---- 5. abnormal volatility ---------------------------------------------------
def volatility_measures(open_, high, low, close, pos, sl):
    """Day-0 log range and overnight gap, each relative to the estimation-window mean.
    range = ln(High/Low)  (Parkinson 1980: sigma^2 = (h-l)^2/(4 ln2)); gap = |ln(Open_t/Close_t-1)|
    (UK RNS comes out at 07:00 so the reaction is an opening gap). Our own ratio construction."""
    rng = np.log(high / low)
    gap = np.abs(np.log(open_ / close.shift(1)))
    rng, gap = rng.values, gap.values
    return dict(av0=rng[pos] / np.nanmean(rng[sl]),
                gap0=gap[pos], gap0_rel=gap[pos] / np.nanmean(gap[sl]))


# ---- 6. event dates and windows -----------------------------------------------
def event_position(dates, event_date, after_close=False):
    """Index of the first trading day the market can react. Weekends roll forward;
    after-close announcements react next trading day."""
    ev = pd.Timestamp(event_date)
    pos = int(dates.searchsorted(ev, side="left"))
    if after_close and pos < len(dates) and dates[pos] == ev:
        pos += 1
    return pos if pos < len(dates) else None


def estimation_slice(pos):
    """Estimation window = EST_LEN days ending GAP days before the event window opens.
    Never overlaps the event window: [MK97] p.20, [B98] p.113 (bias if event is in the estimation window)."""
    tau1 = min(min(a for a, _ in WINDOWS), PRE[0])
    end = pos + tau1 - GAP
    return slice(end - EST_LEN, end)
