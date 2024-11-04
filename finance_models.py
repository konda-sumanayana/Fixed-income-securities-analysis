# finance_models.py
import numpy as np
from scipy.optimize import minimize

# Sample data for companies (replace with real data when available)
companies = {
    'Google': {
        'maturities': np.array([1, 2, 3, 5, 7, 10]),
        'yields': np.array([1.5, 1.7, 1.9, 2.1, 2.3, 2.5])
    },
    'Amazon': {
        'maturities': np.array([1, 2, 3, 5, 7, 10]),
        'yields': np.array([1.6, 1.8, 2.0, 2.2, 2.4, 2.6])
    },
    'Nvidia': {
        'maturities': np.array([1, 2, 3, 5, 7, 10]),
        'yields': np.array([1.7, 1.9, 2.1, 2.3, 2.5, 2.7])
    }
}

def nss_model(t, params):
    beta0, beta1, beta2, beta3, tau1, tau2 = params
    
    # Prevent division by zero by setting a small value
    epsilon = 1e-10
    tau1 = np.where(tau1 == 0, epsilon, tau1)
    tau2 = np.where(tau2 == 0, epsilon, tau2)
    
    # Avoid division by zero for t=0
    t_adjusted = np.where(t == 0, epsilon, t)
    
    term1 = np.where(t == 0, 1, (1 - np.exp(-t_adjusted / tau1)) / (t_adjusted / tau1))
    term2 = np.where(t == 0, 0, term1 - np.exp(-t_adjusted / tau1))
    term3 = np.where(t == 0, 1, (1 - np.exp(-t_adjusted / tau2)) / (t_adjusted / tau2))
    term4 = np.where(t == 0, 0, term3 - np.exp(-t_adjusted / tau2))
    
    return (
        beta0 
        + beta1 * term1
        + beta2 * term2
        + beta3 * term4
    )

def fit_yield_curve(maturities, yields):
    def objective(params):
        return np.sum((yields - nss_model(maturities, params))**2)
    
    initial_params = [3, -1, -1, -1, 1, 1]
    bounds = [(0, 15), (-15, 15), (-30, 30), (-30, 30), (0, 30), (0, 30)]
    result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)
    return result.x

def get_yield_curves():
    maturities = np.linspace(0, 10, 100)
    fitted_yields = {}
    for company, data in companies.items():
        params = fit_yield_curve(data['maturities'], data['yields'])
        fitted_yields[company] = nss_model(maturities, params)
    
    return {
        'maturities': maturities.tolist(),
        'yields': {company: yields.tolist() for company, yields in fitted_yields.items()}
    }

def calculate_var(returns, confidence_level=0.95):
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_cvar(returns, confidence_level=0.95):
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def get_risk_metrics():
    # Simulate returns for demonstration
    np.random.seed(42)
    returns = {company: np.random.normal(0.001, 0.02, 1000) for company in companies}
    
    risk_metrics = {}
    for company, company_returns in returns.items():
        risk_metrics[company] = {
            'var': calculate_var(company_returns),
            'cvar': calculate_cvar(company_returns)
        }
    
    return risk_metrics

def negative_sharpe_ratio(weights, returns, risk_free_rate):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns.T) * 252, weights)))
    return -(portfolio_return - risk_free_rate) / portfolio_std

def get_optimal_portfolio():
    np.random.seed(42)
    returns = {company: np.random.normal(0.001, 0.02, 1000) for company in companies}
    
    n_assets = len(companies)
    args = (np.array(list(returns.values())).T, 0.02 / 252)  # Assuming 2% annual risk-free rate
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(negative_sharpe_ratio, 
                      [1/n_assets] * n_assets, 
                      args=args, 
                      method='SLSQP', 
                      bounds=bounds, 
                      constraints=constraints)
    
    return dict(zip(companies.keys(), result.x))

def get_macro_impact():
    # Simulated macroeconomic factors and their impact on yields
    macro_factors = ['GDP Growth', 'Inflation', 'Unemployment', 'Fed Funds Rate']
    impacts = np.random.uniform(-0.5, 0.5, len(macro_factors))
    return dict(zip(macro_factors, impacts))

# You can test the functions here
if __name__ == "__main__":
    print("Yield Curves:", get_yield_curves())
    print("Risk Metrics:", get_risk_metrics())
    print("Optimal Portfolio:", get_optimal_portfolio())
    print("Macroeconomic Impact:", get_macro_impact())
