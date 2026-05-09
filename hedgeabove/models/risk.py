"""
Risk analytics: VaR, Expected Shortfall, copula analysis, portfolio metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats


def calculate_var(returns, confidence=0.95, method='historical'):
    """Calculate Value at Risk using historical, parametric, or Monte Carlo method."""
    if method == 'historical':
        return np.percentile(returns, (1 - confidence) * 100)
    elif method == 'parametric':
        mu = returns.mean()
        sigma = returns.std()
        return stats.norm.ppf(1 - confidence, mu, sigma)
    elif method == 'monte_carlo':
        mu = returns.mean()
        sigma = returns.std()
        simulations = np.random.normal(mu, sigma, 10000)
        return np.percentile(simulations, (1 - confidence) * 100)


def calculate_es(returns, confidence=0.95):
    """Calculate Expected Shortfall (CVaR)."""
    var = calculate_var(returns, confidence, method='historical')
    return returns[returns <= var].mean()


def calculate_portfolio_metrics(returns):
    """Calculate comprehensive portfolio metrics."""
    metrics = {}
    metrics['Total Return'] = (1 + returns).prod() - 1
    metrics['Annualized Return'] = returns.mean() * 252
    metrics['Annualized Volatility'] = returns.std() * np.sqrt(252)
    metrics['Sharpe Ratio'] = (
        (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        if returns.std() != 0 else 0
    )
    metrics['Sortino Ratio'] = (
        (returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252))
        if len(returns[returns < 0]) > 0 else 0
    )
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    metrics['Max Drawdown'] = drawdown.min()
    return metrics


def fit_copula(returns1, returns2, copula_type='gaussian'):
    """Fit copula to bivariate returns data."""
    try:
        from copulas.bivariate import Gaussian, Clayton, Gumbel, Frank
        from scipy.stats import rankdata

        n = len(returns1)
        u1 = rankdata(returns1) / (n + 1)
        u2 = rankdata(returns2) / (n + 1)

        data = pd.DataFrame({'u1': u1, 'u2': u2})

        copula_map = {
            'gaussian': Gaussian,
            'clayton': Clayton,
            'gumbel': Gumbel,
            'frank': Frank,
        }
        copula_cls = copula_map.get(copula_type.lower(), Gaussian)
        copula = copula_cls()
        copula.fit(data)

        return {
            'copula': copula,
            'type': copula_type,
            'u1': u1,
            'u2': u2,
            'params': copula.to_dict(),
        }
    except Exception as e:
        return None


def calculate_tail_dependence(returns1, returns2, copula_type='gaussian'):
    """Calculate upper and lower tail dependence coefficients."""
    try:
        from scipy.stats import spearmanr, kendalltau

        rho, _ = spearmanr(returns1, returns2)
        tau, _ = kendalltau(returns1, returns2)

        if copula_type.lower() == 'gaussian':
            upper_tail = 0
            lower_tail = 0
        elif copula_type.lower() == 't':
            upper_tail = tau
            lower_tail = tau
        elif copula_type.lower() == 'clayton':
            theta = 2 * tau / (1 - tau) if tau < 1 else 1
            lower_tail = 2 ** (-1 / theta) if theta > 0 else 0
            upper_tail = 0
        elif copula_type.lower() == 'gumbel':
            theta = 1 / (1 - tau) if tau < 1 else 1
            upper_tail = 2 - 2 ** (1 / theta)
            lower_tail = 0
        else:
            upper_tail = 0
            lower_tail = 0

        return {
            'spearman_rho': rho,
            'kendall_tau': tau,
            'upper_tail': upper_tail,
            'lower_tail': lower_tail,
        }
    except Exception as e:
        return {'error': str(e)}


def copula_var(portfolio_returns, returns_matrix, copula_type='gaussian',
               confidence=0.95, simulations=10000):
    """Calculate portfolio VaR using copula simulation."""
    try:
        from copulas.multivariate import GaussianMultivariate

        copula = GaussianMultivariate()
        copula.fit(returns_matrix)

        samples = copula.sample(simulations)

        weights = np.ones(returns_matrix.shape[1]) / returns_matrix.shape[1]
        simulated_returns = (samples * weights).sum(axis=1)

        return np.percentile(simulated_returns, (1 - confidence) * 100)
    except Exception:
        return None


def simulate_copula(copula, n_samples=1000):
    """Generate samples from fitted copula."""
    try:
        return copula.sample(n_samples)
    except Exception:
        return None
