"""
Options pricing: Black-Scholes, Greeks, implied volatility, payoff diagrams.
"""

import numpy as np
from scipy.stats import norm


def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price.

    Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
    """
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho).

    Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
    """
    if T <= 0:
        if option_type == 'call':
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Vega (same for calls and puts) — per 1% vol change
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # Theta — per calendar day
    if option_type == 'call':
        theta = ((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365)
    else:
        theta = ((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                  + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365)

    # Rho — per 1% rate change
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}


def implied_volatility(option_price, S, K, T, r, option_type='call',
                        max_iterations=100, tolerance=1e-5):
    """Calculate implied volatility using Newton-Raphson method."""
    sigma = 0.3

    for i in range(max_iterations):
        price = black_scholes(S, K, T, r, sigma, option_type)
        vega = calculate_greeks(S, K, T, r, sigma, option_type)['vega'] * 100

        diff = option_price - price

        if abs(diff) < tolerance:
            return sigma

        if vega != 0:
            sigma = sigma + diff / vega
        else:
            return None

        sigma = max(0.01, min(sigma, 5.0))

    return None


def option_payoff(S_range, K, premium, option_type='call', position='long'):
    """Calculate option payoff diagram."""
    if option_type == 'call':
        intrinsic = np.maximum(S_range - K, 0)
    else:
        intrinsic = np.maximum(K - S_range, 0)

    if position == 'long':
        return intrinsic - premium
    else:
        return premium - intrinsic
