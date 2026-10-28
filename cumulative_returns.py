import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. LOAD (run from the 'Project 1' folder)
stocks = pd.read_csv("clean_ftse100_subset.csv", parse_dates=["date"])
ftse = (pd.read_csv("clean_benchmark_ftse100.csv", parse_dates=["date"])
          .set_index("date")["close"])

# 2. WIDE prices: one column per stock
prices = stocks.pivot(index="date", columns="ticker", values="close")

# 3. LOG RETURNS on the FTSE's calendar (a missing day becomes NaN, never a 2-day return)
returns = np.log(prices.reindex(ftse.index)).diff().iloc[1:]
ftse_ret = np.log(ftse).diff().iloc[1:]

# 4. CUMULATIVE log returns = running sums
cum_stocks = returns.cumsum()
cum_ftse = ftse_ret.cumsum()

# 5. PLOT: stocks thin and muted, FTSE bold red so it stands out
fig, ax = plt.subplots(figsize=(11, 6))
for t in cum_stocks.columns:
    ax.plot(cum_stocks.index, cum_stocks[t], lw=1.1, alpha=0.75, label=t)
ax.plot(cum_ftse.index, cum_ftse, color="#D62728", lw=3.2, label="FTSE 100")
ax.axhline(0, color="grey", lw=0.8)
ax.set_title("Cumulative log returns: 10 stocks vs FTSE 100")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative log return")
ax.grid(alpha=0.25)
ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
plt.show()
