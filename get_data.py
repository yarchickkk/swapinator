import requests
import json


def get_trades_coingecko(pool_address: str, title: str) -> None:
    url = f"https://api.geckoterminal.com/api/v2/networks/eth/pools/{pool_address}/trades?trade_volume_in_usd_greater_than=0"
    headers = {"accept": "application/json"}

    try:
        response = requests.get(url, headers=headers)
        if int(response.status_code) == 200:
            data = response.json()
            title = f"data/{title}_data.json"
            with open(title, "w") as json_file:
                json.dump(data, json_file, indent=4)
    except:
        return None
    

 
18

if __name__ == "__main__":
    # maga_pool = "0x0c3fdf9c70835f9be9db9585ecb6a1ee3f20a6c7"
    # get_trades_coingecko(maga_pool, "maga")
    # get_pool_coingecko(maga_pool, "maga_pool")

    # doge_pool = "0x308c6fbd6a14881af333649f17f2fde9cd75e2a6"
    # get_trades_coingecko(doge_pool, "doge")
    # get_pool_coingecko(doge_pool, "doge_pool")

    groggo_pool = "0xa07dbd2f63f78a06c41b2528755f62973e6dab18"
    get_trades_coingecko(groggo_pool, "groggo")
    # get_pool_coingecko(groggo_pool, "groggo_pool")

    # spx_pool = "0x52c77b0cb827afbad022e6d6caf2c44452edbc39"
    # get_trades_coingecko(spx_pool, "spx")
    # get_pool_coingecko(spx_pool, "spx_pool")