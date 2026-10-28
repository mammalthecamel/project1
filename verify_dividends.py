"""Check the dividend fix.  python verify_dividends.py   (run after dividend_check.py)"""
import numpy as np
import pandas as pd
import calculations as c
from event_study import load_data, DIV_FILE

px, ret_total, mkt = load_data()                      # with dividends (total returns)
ret_price = c.log_returns(px["close"])                # without dividends (raw price returns)
divs = pd.read_csv(DIV_FILE, parse_dates=["ex_date"])

# 1. only ex-dividend days should differ
diff = (ret_total - ret_price).stack()
changed = diff[diff.abs() > 1e-12]
print(f"1. dividends in file: {len(divs)}   returns changed: {len(changed)}  (should match, minus any off-calendar dates)")

# 2. one example by hand: ln((P_t + D_t) / P_t-1)
d = divs.iloc[0]
p = px["close"][d.ticker]
i = p.index.get_loc(d.ex_date)
by_hand = np.log((p.iloc[i] + d.dividend) / p.iloc[i - 1])
print(f"2. {d.ticker} {d.ex_date.date()}: by hand {by_hand:.6f}   code {ret_total.at[d.ex_date, d.ticker]:.6f}")

# 3. the real proof: market-adjusted return on ex-dates should move from about -1.2% to about 0
on = [(t, x) for t, x in zip(divs.ticker, divs.ex_date) if x in ret_price.index]
before = np.nanmean([ret_price.at[x, t] - mkt[x] for t, x in on])
after = np.nanmean([ret_total.at[x, t] - mkt[x] for t, x in on])
print(f"3. mean ex-date return vs FTSE:  price returns {before:.4f}   total returns {after:.4f}")
