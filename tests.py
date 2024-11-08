import requests
from pprint import pprint


def get_pool_coingecko(pool_address="0xa07dbd2f63f78a06c41b2528755f62973e6dab18", usd_deposit=2130):
  url = f"https://api.geckoterminal.com/api/v2/networks/eth/pools/{pool_address}"
  headers = {"accept": "application/json"}

  response = requests.get(url, headers=headers)

  if int(response.status_code) == 200:
    data = response.json()
    data = data["data"]["attributes"]

    reserve_in_usd = float(data["reserve_in_usd"]) / 2

    quote_token_price_usd = float(data["quote_token_price_usd"])
    base_token_price_usd = float(data["base_token_price_usd"])

    base_deposit = usd_deposit / base_token_price_usd

    x = reserve_in_usd / base_token_price_usd  # GOVNO
    y = reserve_in_usd / quote_token_price_usd   # WETH

    
    k = x * y
    x2 = (x + base_deposit)
    y2 = k / x2

    new_base_price = (y2 / x2) * quote_token_price_usd
    
    price_impact = (new_base_price / base_token_price_usd - 1) * 100

    print(f"USD Deposit: {usd_deposit} Price Impact: {price_impact}%")


get_pool_coingecko()