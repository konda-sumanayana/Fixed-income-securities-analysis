from flask import Flask, render_template, jsonify
from finance_models import main

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    api_key = 'your-api-key'
    results = main(api_key)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
