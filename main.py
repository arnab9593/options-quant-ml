import numpy as np
import matplotlib.pyplot as plt
import logging
import yfinance as yf
from scipy.stats import norm
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class OptionsQuantEngine:
    def __init__(self, risk_free_rate=0.05):
        self.r = risk_free_rate

    def fetch_live_data(self, ticker_symbol):
        try:
            ticker = yf.Ticker(ticker_symbol)
            history = ticker.history(period="1d")
            if history.empty:
                raise ValueError(f"No data found for {ticker_symbol}")
            
            S = history['Close'].iloc[-1]
            expirations = ticker.options
            opt_chain = ticker.option_chain(expirations[0])
            calls = opt_chain.calls

            atm_idx = (calls['strike'] - S).abs().idxmin()
            K = calls.loc[atm_idx, 'strike']
            sigma = calls.loc[atm_idx, 'impliedVolatility']

            logger.info(f"Loaded {ticker_symbol}: Spot={S:.2f}, Strike={K}, IV={sigma:.2%}")
            return S, K, sigma

        except Exception as e:
            logger.error(f"Error fetching data for {ticker_symbol}: {e}")
            return None, None, None

    def calculate_metrics(self, flag, S, K, T, sigma):
        """ black scholes price and all greeks"""
        price = black_scholes(flag, S, K, T, self.r, sigma)
        greeks = {
            "Delta": delta(flag, S, K, T, self.r, sigma),
            "Gamma": gamma(flag, S, K, T, self.r, sigma),
            "Vega":  vega(flag, S, K, T, self.r, sigma),
            "Theta": theta(flag, S, K, T, self.r, sigma),
            "Rho":   rho(flag, S, K, T, self.r, sigma)
        }
        return price, greeks

    def simulate_gbm_paths(self, S0, T, sigma, steps=252, n_sim=10000):
        """geometric brownian motion paths"""
        dt = T / steps
        paths = np.zeros((n_sim, steps + 1))
        paths[:, 0] = S0
        for t in range(1, steps + 1):
            Z = np.random.standard_normal(n_sim)
            paths[:, t] = paths[:, t-1] * np.exp((self.r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        return paths

    def asian_call_price(self, S0, K, T, sigma):
        """asian call using simulated paths"""
        paths = self.simulate_gbm_paths(S0, T, sigma)
        avg_prices = np.mean(paths, axis=1)
        payoffs = np.maximum(avg_prices - K, 0)
        return np.exp(-self.r * T) * np.mean(payoffs)

    def plot_delta_analysis(self, ticker, S, K, sigma, T, opt_price, greeks):
        """s-curve with a results summary overlay"""
        S_range = np.linspace(S * 0.8, S * 1.2, 100)
        deltas = [delta('c', s, K, T, self.r, sigma) for s in S_range]
        plt.figure(figsize=(12, 7))
        plt.plot(S_range, deltas, color='tab:green', lw=2, label='Call Delta')
        plt.axvline(S, color='red', linestyle='--', label=f'Current Spot (${S:.2f})')

        stats_text = (
            f"RESULTS FOR {ticker}\n"
            f"------------------\n"
            f"Spot: ${S:.2f}\n"
            f"Strike: ${K}\n"
            f"IV: {sigma:.2%}\n\n"
            f"Call Price: ${opt_price:.2f}\n"
            f"Delta: {greeks['Delta']:.4f}\n"
            f"Gamma: {greeks['Gamma']:.4f}\n"
            f"Vega: {greeks['Vega']:.4f}\n"
            f"Theta: {greeks['Theta']:.4f}"
        )

        plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                       fontsize=10, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.title(f"{ticker} Sensitivity Analysis: Delta vs Spot Price")
        plt.xlabel("Stock Price ($)")
        plt.ylabel("Delta (Δ)")
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        logger.info("Displaying plot with results overlay...")
        plt.show()


if __name__ == "__main__":

    TICKER = input("Enter TICKER: ") 
    engine = OptionsQuantEngine(risk_free_rate=0.045) 
    # Pipeline: Data -> Pricing -> Greeks -> Visualization
    s_price, k_strike, iv = engine.fetch_live_data(TICKER)

    if s_price:
        expiry_years = 30 / 365
        opt_price, greeks = engine.calculate_metrics('c', s_price, k_strike, expiry_years, iv)
        asian_price = engine.asian_call_price(s_price, k_strike, expiry_years, iv)

        print("-" * 30)
        print(f"RESULTS FOR {TICKER}")
        print("-" * 30)
        print(f"Market Spot Price: ${s_price:.2f}")
        print(f"Standard Call Price: ${opt_price:.4f}")
        print(f"Asian Call Price:    ${asian_price:.4f}")

        for k, v in greeks.items():
            print(f"{k:8}: {v:.4f}")
        print("-" * 30)
        engine.plot_delta_analysis(TICKER, s_price, k_strike, iv, expiry_years, opt_price, greeks) 