from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho
from pricing import monte_carlo_call

def compute_greeks(flag, S, K, T, r, sigma):
    return {
        "Delta": delta(flag, S, K, T, r, sigma),
        "Gamma": gamma(flag, S, K, T, r, sigma),
        "Vega":  vega(flag, S, K, T, r, sigma),
        "Theta": theta(flag, S, K, T, r, sigma),
        "Rho":   rho(flag, S, K, T, r, sigma)
    }


# if __name__ == "__main__":
#     S = 100
#     K = 100
#     T = 30/365
#     r = 0.01
#     sigma = 0.2

#     greeks_call = compute_greeks('c', S, K, T, r, sigma)
#     greeks_put  = compute_greeks('p', S, K, T, r, sigma)

#     print("CALL OPTION GREEKS")
#     for k, v in greeks_call.items():
#         print(f"{k}: {v:.6f}")

#     print("\nPUT OPTION GREEKS")
#     for k, v in greeks_put.items():
#         print(f"{k}: {v:.6f}")


def mc_delta(S, K, T, r, sigma, eps=0.01):
    price_up = monte_carlo_call(S+eps, K, T, r, sigma)
    price_down = monte_carlo_call(S-eps, K, T, r, sigma)
    return (price_up - price_down) / (2*eps)
