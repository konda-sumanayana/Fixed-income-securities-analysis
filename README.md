# Fixed-Income Securities Analysis and Yield Curve Modeling

## Project Overview

This project implements a sophisticated fixed-income securities analysis tool, focusing on yield curve modeling, risk assessment, and portfolio optimization. It processes data from multiple economic indicators to predict yield curves and assess their impact on bond prices and yields.

## Features

- Yield curve modeling using the Nelson-Siegel-Svensson (NSS) model
- Risk assessment using Value at Risk (VaR) and Conditional Value at Risk (CVaR)
- Portfolio optimization based on the Sharpe ratio
- Yield curve prediction using machine learning (Random Forest)
- Sensitivity analysis of economic indicators' impact on yields
- Real-time data fetching from FRED (Federal Reserve Economic Data) and U.S. Treasury

## Technologies Used

- Python 3.8+
- Flask for web application
- NumPy and Pandas for data manipulation
- SciPy for optimization
- Scikit-learn for machine learning
- Requests for API calls
- Chart.js for data visualization

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/fixed-income-analysis.git
cd fixed-income-analysis
```

2. Install required packages:
```
pip install -r requirements.txt
```

3. Obtain a FRED API key from [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

4. Replace 'your_fred_api_key_here' in `app.py` and `finance_models.py` with your actual FRED API key.

## Usage

1. Run the Flask application:
```
python app.py
```

2. Open a web browser and navigate to `http://localhost:5000`

3. The web interface will display yield curves, risk metrics, optimal portfolio allocation, and macroeconomic impact analysis.

## Project Structure

- `app.py`: Flask application setup and routing
- `finance_models.py`: Core financial models and data processing functions
- `templates/index.html`: Web interface for displaying results
- `requirements.txt`: List of Python package dependencies

## Future Improvements

- Implement real-time updates for continuous data refreshing
- Enhance prediction accuracy to achieve 90% or higher
- Expand the range of economic indicators and their analysis
- Implement more sophisticated portfolio optimization techniques
- Add user authentication and personalized portfolio management features

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.
