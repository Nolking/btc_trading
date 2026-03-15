from flask import Flask
app = Flask(__name__)
from investiny import historical_data

data = historical_data(investing_id=6408, from_date="09/01/2022", to_date="10/01/2022") # Returns AAPL historical data as JSON (without date)
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


