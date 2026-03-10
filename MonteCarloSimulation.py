import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
from scipy.stats import t as t_dist
import time

# CONFIGURATION

TICKER = "EQNR.OL"
YEARS_HISTORY = "5y"

SIMULATIONS = 10000
DAYS = 252

PLOT_PATHS = 100
T_DIST_DOF = 5           # Degrees of freedom for fat-tail t-distribution (lower = fatter tails)
RISK_FREE_RATE = 0.02    # Annual risk-free rate (used for Sharpe/Sortino)

np.random.seed(42)

print("Downloading historical data...")

# DATA COLLECTION

data = yf.download(TICKER, period=YEARS_HISTORY)

prices = data["Close"][TICKER]
returns = prices.pct_change().dropna()

print("Data downloaded.")
print(f"Number of price points: {len(prices)}")

# MODEL PARAMETERS

mu = returns.mean()
sigma = returns.std()
S0 = float(prices.iloc[-1])

print("\nModel parameters:")
print("Current price:", round(S0, 2))
print("Mean daily return:", round(float(mu), 6))
print("Daily volatility:", round(float(sigma), 6))

# Annualized metrics
annual_return = mu * 252
annual_volatility = sigma * np.sqrt(252)

print("\nAnnualized metrics:")
print("Expected return:", f"{float(annual_return):.2%}")
print("Volatility:", f"{float(annual_volatility):.2%}")

# MONTE CARLO SIMULATION

print("\nStarting Monte Carlo simulation...\n")

start_time = time.time()

# Geometric Brownian Motion with fat-tailed t-distribution shocks
# Shocks are rescaled so variance = 1, consistent with GBM assumptions
raw_shocks = t_dist.rvs(df=T_DIST_DOF, size=(SIMULATIONS, DAYS))
random_shocks = raw_shocks / np.sqrt(T_DIST_DOF / (T_DIST_DOF - 2))

drift = float(mu) - 0.5 * float(sigma) ** 2
diffusion = float(sigma) * random_shocks

daily_returns = np.exp(drift + diffusion)

# Vectorized path construction via cumulative product
price_paths = np.empty((SIMULATIONS, DAYS + 1))
price_paths[:, 0] = S0
price_paths[:, 1:] = S0 * np.cumprod(daily_returns, axis=1)

results = price_paths[:, -1]

elapsed = time.time() - start_time
print(f"Simulation finished in {elapsed:.2f} seconds")

# RESULTS

expected_price = np.mean(results)

prob_up   = np.mean(results > S0)
prob_down = np.mean(results < S0)

log_returns_sim = np.log(results / S0)
volatility  = np.std(log_returns_sim)
mean_return = np.mean(log_returns_sim)

print("\nResults")
print("Current price:", round(S0, 2))
print("Expected price:", round(expected_price, 2))

print("\nProbabilities:")
print("Price goes UP:  ", round(prob_up   * 100, 2), "%")
print("Price goes DOWN:", round(prob_down * 100, 2), "%")

print("\nLog return:", f"{mean_return:.2%}")
print("Volatility:", f"{volatility:.2%}")

print("\nDistribution stats:")
print("Min price:   ", round(results.min(), 2))
print("Max price:   ", round(results.max(), 2))
print("Median price:", round(float(np.median(results)), 2))

# RISK METRICS

ci_low  = np.percentile(results, 2.5)
ci_high = np.percentile(results, 97.5)

var_95_price = np.percentile(results, 5)
var_95       = S0 - var_95_price

# CVaR: average loss in the worst 5% of outcomes
cvar_95 = S0 - results[results <= var_95_price].mean()

# Max drawdown: peak-to-trough drop inside each simulated path
peak             = np.maximum.accumulate(price_paths, axis=1)
drawdowns        = (price_paths - peak) / peak
avg_max_drawdown = drawdowns.min(axis=1).mean()

print("\n95% Confidence Interval:")
print("Lower:", round(ci_low,  2))
print("Upper:", round(ci_high, 2))

print("\nValue at Risk (95%):", round(var_95, 2))
print("CVaR / Expected Shortfall (95%):", round(cvar_95, 2))
print("Avg Max Drawdown across paths:", f"{avg_max_drawdown:.2%}")

# Sharpe ratio
daily_rf = RISK_FREE_RATE / 252
sharpe   = (float(mu) - daily_rf) / float(sigma)
print("\nSharpe Ratio:", round(sharpe, 3))

# Sortino ratio: penalizes only downside volatility
downside_returns = returns[returns < 0]
sortino = (float(mu) - daily_rf) / float(downside_returns.std())
print("Sortino Ratio:", round(sortino, 3))

# SAVE RESULTS

df = pd.DataFrame({"FinalPrice": results})
df.to_csv("monte_carlo_results.csv", index=False)

summary = {
    "Ticker": TICKER,
    "CurrentPrice": round(S0, 2),
    "ExpectedPrice": round(expected_price, 2),
    "ProbUp_%": round(prob_up * 100, 2),
    "ProbDown_%": round(prob_down * 100, 2),
    "VaR_95": round(var_95, 2),
    "CVaR_95": round(cvar_95, 2),
    "AvgMaxDrawdown_%": round(avg_max_drawdown * 100, 2),
    "SharpeRatio": round(sharpe, 3),
    "SortinoRatio": round(sortino, 3),
}
pd.DataFrame([summary]).to_csv("monte_carlo_summary.csv", index=False)
print("\nResults saved to monte_carlo_results.csv and monte_carlo_summary.csv")

# VISUALIZATION

print("\nPlotting simulation paths...")

fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

# Simulation paths + fan chart
for i in range(PLOT_PATHS):
    ax1.plot(price_paths[i], alpha=0.15, linewidth=0.7, color="steelblue")

mean_path = np.mean(price_paths, axis=0)
p10 = np.percentile(price_paths, 10, axis=0)
p25 = np.percentile(price_paths, 25, axis=0)
p75 = np.percentile(price_paths, 75, axis=0)
p90 = np.percentile(price_paths, 90, axis=0)

x = range(len(mean_path))
ax1.fill_between(x, p10, p90, alpha=0.15, color="steelblue", label="10–90% range")
ax1.fill_between(x, p25, p75, alpha=0.25, color="steelblue", label="25–75% range")
ax1.plot(mean_path, linewidth=2.5, color="navy",   label="Mean path")
ax1.axhline(S0,     linewidth=1.5, color="orange", linestyle="--", label=f"Current price ({round(S0,2)})")

ax1.set_title(f"Monte Carlo Simulation — {TICKER}  ({SIMULATIONS:,} paths, {DAYS} days)", fontsize=13)
ax1.set_xlabel("Trading Days")
ax1.set_ylabel("Price")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Final price distribution with KDE
sns.histplot(results, bins=80, kde=True, ax=ax2, color="steelblue", alpha=0.6)
ax2.axvline(S0,             color="orange", linewidth=2,               label=f"Current ({round(S0,2)})")
ax2.axvline(expected_price, color="navy",   linewidth=2,  linestyle="--", label=f"Expected ({round(expected_price,2)})")
ax2.axvline(var_95_price,   color="red",    linewidth=1.5, linestyle=":",  label=f"VaR 95% ({round(var_95_price,2)})")
ax2.axvline(ci_low,         color="green",  linewidth=1.5, linestyle=":",  label=f"CI 2.5% ({round(ci_low,2)})")
ax2.axvline(ci_high,        color="green",  linewidth=1.5, linestyle=":",  label=f"CI 97.5% ({round(ci_high,2)})")

ax2.set_title("Distribution of Final Prices", fontsize=12)
ax2.set_xlabel("Final Price")
ax2.set_ylabel("Frequency")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Max drawdown distribution
max_drawdowns = drawdowns.min(axis=1) * 100
sns.histplot(max_drawdowns, bins=60, kde=True, ax=ax3, color="tomato", alpha=0.6)
ax3.axvline(avg_max_drawdown * 100, color="darkred", linewidth=2,
            label=f"Avg max drawdown ({avg_max_drawdown:.1%})")

ax3.set_title("Distribution of Max Drawdowns per Path", fontsize=12)
ax3.set_xlabel("Max Drawdown (%)")
ax3.set_ylabel("Frequency")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.suptitle(f"{TICKER} — Monte Carlo Risk Analysis", fontsize=15, fontweight="bold", y=1.01)
plt.savefig("monte_carlo_improved.png", dpi=150, bbox_inches="tight")
plt.show()

print("Plot saved to monte_carlo_improved.png")