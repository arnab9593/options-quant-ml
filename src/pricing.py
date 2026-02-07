from py_vollib.black_scholes import black_scholes
import numpy as np
from scipy.stats import norm


def price_option(flag, S, K, T, r, sigma):
    return black_scholes(flag, S, K, T, r, sigma)


# if __name__ == "__main__":
#     S = 100     
#     K = 100     
#     T = 30/365  
#     r = 0.01    
#     sigma = 0.2 

#     call_price = price_option('c', S, K, T, r, sigma)
#     put_price  = price_option('p', S, K, T, r, sigma)

#     print(f"Call Option Price: {call_price:.4f}")
#     print(f"Put Option Price : {put_price:.4f}")

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def monte_carlo_call(S0, K, T, r, sigma, n_sim=100000):
    Z = np.random.standard_normal(n_sim)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r*T) * np.mean(payoff)
