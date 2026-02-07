from pricing import black_scholes_call, monte_carlo_call

S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

bs = black_scholes_call(S,K,T,r,sigma)
mc = monte_carlo_call(S,K,T,r,sigma)

print("BS Price:", bs)
print("MC Price:", mc)
