import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import requests
from datetime import datetime, timedelta
import certifi
from io import StringIO

# Fetch real economic indicators from FRED API
def fetch_economic_indicators(api_key):
    base_url = 'https://api.stlouisfed.org/fred/series/observations'
    series = {
        'GDP': 'GDP',
        'Inflation': 'CPIAUCSL',
        'Unemployment': 'UNRATE',
        'Fed_Funds_Rate': 'FEDFUNDS',
        'Industrial_Production': 'INDPRO',
        'Consumer_Confidence': 'UMCSENT',
        'Retail_Sales': 'RSAFS',
        'Housing_Starts': 'HOUST'
    }
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    
    data = {}
    for name, series_id in series.items():
        params = {
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            observations = response.json()['observations']
            data[name] = pd.DataFrame(observations)[['date', 'value']].set_index('date')
            data[name]['value'] = pd.to_numeric(data[name]['value'], errors='coerce')
        else:
            print(f"Failed to fetch data for {name}")
    
    return pd.concat(data.values(), axis=1, keys=data.keys())

# Fetch real yield curve data from U.S. Treasury
def fetch_yield_curve_data():
    url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2023/all?type=daily_treasury_yield_curve&field_tdr_date_value=2023&page&_format=csv"
    response = requests.get(url)
    df = pd.read_csv(StringIO(response.text))
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # Ensure we're only using the yield columns
    yield_columns = ['1 Mo', '2 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
    df = df[yield_columns]
    return df


# Nelson-Siegel-Svensson model
def nss_model(t, params):
    beta0, beta1, beta2, beta3, tau1, tau2 = params
    
    epsilon = 1e-10
    tau1 = np.where(tau1 == 0, epsilon, tau1)
    tau2 = np.where(tau2 == 0, epsilon, tau2)
    
    t_adjusted = np.where(t == 0, epsilon, t)
    
    term1 = np.where(t == 0, 1, (1 - np.exp(-t_adjusted / tau1)) / (t_adjusted / tau1))
    term2 = np.where(t == 0, 0, term1 - np.exp(-t_adjusted / tau1))
    term3 = np.where(t == 0, 1, (1 - np.exp(-t_adjusted / tau2)) / (t_adjusted / tau2))
    term4 = np.where(t == 0, 0, term3 - np.exp(-t_adjusted / tau2))
    
    return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term4

def fit_yield_curve(maturities, yields):
    # Ensure maturities and yields have the same length
    if len(maturities) != len(yields):
        raise ValueError("maturities and yields must have the same length")
    
    def objective(params):
        return np.sum((yields - nss_model(maturities, params))**2)
    
    initial_params = [3, -1, -1, -1, 1, 1]
    bounds = [(0, 15), (-15, 15), (-30, 30), (-30, 30), (0, 30), (0, 30)]
    result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)
    return result.x


def get_yield_curves():
    yield_curve_data = fetch_yield_curve_data()
    maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
    
    fitted_yields = {}
    for date, row in yield_curve_data.iterrows():
        yields = row.values
        # Ensure maturities and yields have the same length
        if len(maturities) != len(yields):
            maturities = maturities[:len(yields)]  # or adjust yields if needed
        params = fit_yield_curve(maturities, yields)
        fitted_yields[date.strftime('%Y-%m-%d')] = nss_model(np.linspace(0, 30, 100), params).tolist()
    
    return {
        'maturities': np.linspace(0, 30, 100).tolist(),
        'yields': fitted_yields
    }


def calculate_var(returns, confidence_level=0.95):
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_cvar(returns, confidence_level=0.95):
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def get_risk_metrics(yield_curve_data):
    returns = yield_curve_data.pct_change().dropna()
    
    risk_metrics = {}
    for column in returns.columns:
        risk_metrics[column] = {
            'var': calculate_var(returns[column]),
            'cvar': calculate_cvar(returns[column])
        }
    
    return risk_metrics

def negative_sharpe_ratio(weights, returns, risk_free_rate):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns.T) * 252, weights)))
    return -(portfolio_return - risk_free_rate) / portfolio_std

def get_optimal_portfolio(yield_curve_data):
    returns = yield_curve_data.pct_change().dropna()
    
    n_assets = len(returns.columns)
    args = (returns, 0.02 / 252)  # Assuming 2% annual risk-free rate
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(negative_sharpe_ratio, 
                      [1/n_assets] * n_assets, 
                      args=args, 
                      method='SLSQP', 
                      bounds=bounds, 
                      constraints=constraints)
    
    return dict(zip(returns.columns, result.x))

def predict_yield_curve(economic_data, yield_data):
    X = economic_data
    y = yield_data.mean(axis=1)  # Using average yield as target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    return predictions, accuracy

def sensitivity_analysis(model, economic_data, yield_data):
    sensitivities = {}
    for indicator in economic_data.columns:
        perturbed_data = economic_data.copy()
        perturbed_data[indicator] *= 1.01  # 1% increase
        perturbed_prediction = model.predict(perturbed_data)
        sensitivity = (perturbed_prediction - yield_data.mean(axis=1)) / (0.01 * economic_data[indicator])
        sensitivities[indicator] = sensitivity.mean()
    return sensitivities

def get_macro_impact(economic_data, yield_data):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(economic_data, yield_data.mean(axis=1))
    
    return sensitivity_analysis(model, economic_data, yield_data)

def main(api_key):
    economic_data = fetch_economic_indicators(api_key)
    yield_curve_data = fetch_yield_curve_data()
    
    yield_curves = get_yield_curves()
    risk_metrics = get_risk_metrics(yield_curve_data)
    optimal_portfolio = get_optimal_portfolio(yield_curve_data)
    predictions, accuracy = predict_yield_curve(economic_data, yield_curve_data)
    macro_impact = get_macro_impact(economic_data, yield_curve_data)
    
    return {
        'yield_curves': yield_curves,
        'risk_metrics': risk_metrics,
        'optimal_portfolio': optimal_portfolio,
        'prediction_accuracy': accuracy,
        'macro_impact': macro_impact
    }

if __name__ == "__main__":
    api_key = 'ec4077324af26cb6d90f2ae75dcd2541'
    results = main(api_key)
    print(results)
