import numpy as np
import matplotlib.pyplot as plt
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta
from greeks import mc_delta

S_range = np.linspace(50, 150, 50)
deltas = [mc_delta(S, 100, 1, 0.05, 0.2) for S in S_range]

plt.plot(S_range, deltas)
plt.title("Monte Carlo Delta")
plt.xlabel("Spot Price")
plt.ylabel("Delta")
plt.show()


# # Parameters
# K = 100
# T = 30/365
# r = 0.01
# sigma = 0.2

# spots = np.linspace(50, 150, 100)

# # Compute Greeks
# deltas = [delta('c', S, K, T, r, sigma) for S in spots]
# gammas = [gamma('c', S, K, T, r, sigma) for S in spots]

# # Vega vs Volatility
# vols = np.linspace(0.05, 1.0, 100)
# vegas = [vega('c', 100, K, T, r, v) for v in vols]


# # Theta vs Time to Expiry
# times = np.linspace(1/365, 365/365, 100)
# thetas = [theta('c', 100, K, t, r, sigma) for t in times]

# # Plot Delta
# plt.figure()
# plt.plot(spots, deltas)
# plt.xlabel("Spot Price")
# plt.ylabel("Delta")
# plt.title("Call Option Delta vs Spot Price")
# plt.show()

# # Plot Gamma
# plt.figure()
# plt.plot(spots, gammas)
# plt.xlabel("Spot Price")
# plt.ylabel("Gamma")
# plt.title("Call Option Gamma vs Spot Price")
# plt.show()


# plt.figure()
# plt.plot(vols, vegas)
# plt.xlabel("Volatility")
# plt.ylabel("Vega")
# plt.title("Vega vs Volatility")
# plt.show()


# plt.figure()
# plt.plot(times, thetas)
# plt.xlabel("Time to Expiry (years)")
# plt.ylabel("Theta")
# plt.title("Theta vs Time to Expiry")
# plt.show()

