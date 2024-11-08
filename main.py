import sys
import json
import torch
from torch import nn as nn

ERROR = "\033[1;31m(swapinator)\033[0m"
SPACE = " " * 12


class Swapinator(nn.Module):
    
    def __init__(self, pool_data: dict) -> None:
        super().__init__()

        # get pool attributes safely
        self.__validate_and_unpack(pool_data)
        # set initial deposit value
        self.input_usd = torch.tensor(2192.0, dtype=torch.float64)

    def forward(self, percent: float) -> float:
        # convert USD to core tokens
        core_input = self.input_usd / self.core_price_usd
        # apply the uniswap fee of 0.3%
        effective_core_input = core_input * torch.tensor(0.997, dtype=torch.float64)
        # compute paired tokens obtained for core
        paired_output = (self.paired_reserve * effective_core_input) / (self.core_reserve + effective_core_input)
        
        # --- 
        updated_core_reserve = self.core_reserve + effective_core_input
        updated_paired_reserve = self.paired_reserve - paired_output

        flag = self.core_reserve * self.paired_reserve <= updated_core_reserve * updated_paired_reserve
        print("Constant product maintained:", flag.item())
        # ---

        updated_paired_price_usd = (updated_core_reserve / updated_paired_reserve) * self.core_price_usd
        
        # ---
        price_diff = updated_paired_price_usd - self.paired_price_usd
        price_ratio = updated_paired_price_usd / self.paired_price_usd
        print(f"Price difference: {price_diff:.10f}")
        print(f"Price ratio     : {price_ratio:.10f}")
        # ---

        loss = torch.tensor(1.0 + percent / 100, dtype=torch.float64) - price_ratio
        print(f"loss: {loss}")
        # loss.backward()
        pass
        # optimizes start deposit to fit paired token target price change
        # clears gradients, returns required deposit

    def __validate_and_unpack(self, pool_data: dict) -> None:
        # stop execution if pool_data is of wrong format
        try:
            # get actual prices of tokens in the pool and it's liquidity in USD
            attributes = pool_data["data"]["attributes"]
            self.core_price_usd = torch.tensor(float(attributes["quote_token_price_usd"]), dtype=torch.float64)
            self.paired_price_usd = torch.tensor(float(attributes["base_token_price_usd"]), dtype=torch.float64)
            self.pool_liquidity_usd = torch.tensor(float(attributes["reserve_in_usd"]), dtype=torch.float64)

            # calculate actual approximate amount of both tokens in the pool
            self.core_reserve = self.pool_liquidity_usd / (2 * self.core_price_usd)
            self.paired_reserve = self.pool_liquidity_usd / (2 * self.paired_price_usd)
        
        except Exception as e:
            sys.exit(f"{ERROR} Seems like input isn't a dictionary or doesn't have following keys with according values:\n"
                     f"{SPACE} • ['data']['attributes']['quote_token_price_usd']: string convertible to float\n"
                     f"{SPACE} • ['data']['attributes']['base_token_price_usd'] : string convertible to float\n"
                     f"{SPACE} • ['data']['attributes']['reserve_in_usd']       : string convertible to float")
    
    def __repr__(self) -> str:
        return (
            f"Core token price   :{self.core_price_usd:>25.10f} $\n"
            f"Paired token price :{self.paired_price_usd:>25.10f} $\n"
            f"Pool liquidity     :{self.pool_liquidity_usd:>25.10f} $\n"
            f"Core reserve       :{self.core_reserve:>25.10f} tokens\n"
            f"Paired reserve     :{self.paired_reserve:>25.10f} tokens"
        )   

    # optional
    def __swap(self):
        pass
        # performs token swap
    
    def __optimize(self):
        pass
        # runs optimization loop


with open("data/maga_pool_data.json") as f:
    maga_pool_data = json.load(f)

swap = Swapinator(maga_pool_data)
print(swap)
swap(5)







def swap(core_reserve: float, paired_reserve: float, input_amount: float, purchase_kind: str = "buy") -> float:
    """
    Perform a swap in an automated market maker (AMM) pool based on Uniswap's constant product formula.
    """
    input_reserve, output_reserve = (core_reserve, paired_reserve) if purchase_kind == "buy" else (paired_reserve, core_reserve)

    effective_input = input_amount * 0.997  # apply the Uniswap fee of 0.3%
    output_amount = (output_reserve * effective_input) / (input_reserve + effective_input)
    return output_amount

"""


input_token_amount = float(data["data"][0]["attributes"]["from_token_amount"])

output_token_amount = float(data["data"][0]["attributes"]["to_token_amount"])
output_amount = swap(
    core_reserve=core_token_reserve,
    paired_reserve=paired_token_reserve,
    input_amount=input_token_amount,
    purchase_kind=purchase_kind
)

price = paired_token_price_usd if purchase_kind == "buy" else core_token_price_usd
res = abs(output_token_amount - output_amount) * price
print(res)
"""


