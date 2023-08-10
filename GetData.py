import requests
import urllib
import json
import pickle
import pandas as pd

symbol = "BSE:INFY"

api_url = "https://www.alphavantage.co/query?"

params = {
    "function": "TIME_SERIES_DAILY",
    "symbol": symbol,
    "outputsize": "full",
    "datatype": "json",
    "apikey": "OPLI8A9BAZPIQNJI",

}

url = api_url + urllib.parse.urlencode(params)
print(url)
req = requests.get(url)
with open(f"{symbol}.json", "w") as f:
    f.write(req.text)

df = pd.read_json(f"{symbol}.json")

print(df)
