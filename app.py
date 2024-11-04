# app.py
from flask import Flask, render_template, jsonify
from finance_models import (get_yield_curves, get_risk_metrics, 
                            get_optimal_portfolio, get_macro_impact)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    yield_curves = get_yield_curves()
    risk_metrics = get_risk_metrics()
    optimal_portfolio = get_optimal_portfolio()
    macro_impact = get_macro_impact()
    
    return jsonify({
        'yield_curves': yield_curves,
        'risk_metrics': risk_metrics,
        'optimal_portfolio': optimal_portfolio,
        'macro_impact': macro_impact
    })

if __name__ == '__main__':
    app.run(debug=True)
