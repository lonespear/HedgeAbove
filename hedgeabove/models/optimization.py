"""
Portfolio optimization: Modern Portfolio Theory, Efficient Frontier.
"""

import numpy as np
from scipy.optimize import minimize


def optimize_portfolio(returns, method='max_sharpe', target_return=None):
    """Portfolio optimization using Modern Portfolio Theory."""
    n_assets = returns.shape[1]

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    def portfolio_return(weights):
        return np.dot(weights, mean_returns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def neg_sharpe(weights):
        ret = portfolio_return(weights)
        vol = portfolio_volatility(weights)
        return -ret / vol if vol != 0 else 0

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x: portfolio_return(x) - target_return,
        })

    bounds = tuple((0, 1) for _ in range(n_assets))
    init_guess = np.array([1 / n_assets] * n_assets)

    if method == 'max_sharpe':
        result = minimize(neg_sharpe, init_guess, method='SLSQP',
                          bounds=bounds, constraints=constraints)
    elif method == 'min_vol':
        result = minimize(portfolio_volatility, init_guess, method='SLSQP',
                          bounds=bounds, constraints=constraints)
    elif method == 'target_return':
        result = minimize(portfolio_volatility, init_guess, method='SLSQP',
                          bounds=bounds, constraints=constraints)
    elif method == 'risk_parity':
        def risk_parity_objective(weights):
            portfolio_vol = portfolio_volatility(weights)
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol
            target_risk = portfolio_vol / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)
        result = minimize(risk_parity_objective, init_guess, method='SLSQP',
                          bounds=bounds, constraints=constraints)

    return result.x if result.success else init_guess


def generate_efficient_frontier(returns, num_portfolios=50):
    """Generate efficient frontier portfolios."""
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    target_returns = np.linspace(min_ret, max_ret, num_portfolios)

    frontier_vols = []
    frontier_rets = []
    frontier_weights = []

    for target in target_returns:
        try:
            weights = optimize_portfolio(returns, method='target_return',
                                         target_return=target)
            ret = np.dot(weights, mean_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            frontier_rets.append(ret)
            frontier_vols.append(vol)
            frontier_weights.append(weights)
        except Exception:
            continue

    return frontier_rets, frontier_vols, frontier_weights
