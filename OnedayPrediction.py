import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import time



TICKER = "EQNR.OL"
YEARS_HISTORY = "5y"
SIMULATIONS = 1000
DAYS = 1   # Predicting tommorwos prices
PROGRESS_INTERVAL = 100  # print update every N simulations

print("Downloading historical data...")

data = yf.download(TICKER, period=YEARS_HISTORY)

prices = data["Close"][TICKER]

returns = prices.pct_change().dropna()

print("Data downloaded.")
print(f"Number of price points: {len(prices)}")

mu = returns.mean()
sigma = returns.std()
S0 = prices.iloc[-1]

print("\nModel parameters:")
print("Current price:", round(float(S0), 2))
print("Mean daily return:", round(float(mu), 6))
print("Daily volatility:", round(float(sigma), 6))


print("\nStarting Monte Carlo simulation...\n")

start_time = time.time()

results = []
paths = []

for i in range(SIMULATIONS):

    price = S0
    path = [price]

    for d in range(DAYS):

        shock = np.random.normal(mu, sigma)
        price = price * (1 + shock)
        path.append(price)

    results.append(price)

    if i < 100:   # save first 100 paths for plotting
        paths.append(path)

    # Progress update
    if (i + 1) % PROGRESS_INTERVAL == 0:

        elapsed = time.time() - start_time
        progress = (i + 1) / SIMULATIONS * 100

        print(
            f"Progress: {i+1}/{SIMULATIONS} "
            f"({progress:.1f}%) | "
            f"Elapsed: {elapsed:.1f}s"
        )

#RESULTS
results = np.array(results, dtype=float)

prob_up = np.mean(results > S0)
prob_down = np.mean(results < S0)

expected_price = np.mean(results)

print("\nSimulation finished.\n")

print("Results:")
print("----------------------------")
print("Current price:", round(S0, 2))
print("Expected price:", round(expected_price, 2))
print("Probability price goes UP:", round(prob_up * 100, 2), "%")
print("Probability price goes DOWN:", round(prob_down * 100, 2), "%")

print("\nDistribution stats:")
print("Min price:", round(results.min(), 2))
print("Max price:", round(results.max(), 2))
print("Median price:", round(np.median(results), 2))

# PLOTTING

print("\nPlotting simulation paths...")

for path in paths:
    plt.plot(path)

plt.title(f"Monte Carlo Simulation for {TICKER}")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()



plt.hist(results, bins=50)
plt.title("Distribution of Final Prices")
plt.xlabel("Final Price")
plt.ylabel("Frequency")
plt.show()