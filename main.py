import sys
import json
import torch
from torch import nn as nn


ERROR = "\033[1;31m(swapinator)\033[0m"
SPACE = " " * 12
LEARNING_RATE = 1000.0
INITIAL_DEPOSIT = 2192.0


class Swapinator(nn.Module):
    
    def __init__(self, pool_data: dict) -> None:
        super().__init__()

        # get pool attributes safely
        self.__validate_and_unpack(pool_data)
        # set initial deposit value - the optimized parameter, requires gradient
        self.input_usd = nn.Parameter(torch.tensor(INITIAL_DEPOSIT, dtype=torch.float64, requires_grad=True))

    def forward(self, percent: float) -> float:
        # perhaps adding data dependent lr setting
        optimizer = torch.optim.AdamW(params=[self.input_usd], lr=LEARNING_RATE, weight_decay=1e-4)
        loss = torch.tensor(float("inf"), dtype=torch.float64)
        
        for i in range(100):
            
            print(f"{i} deposit: {self.input_usd.item()} $")
            
            # convert USD to core tokens
            core_input = self.input_usd / self.core_price_usd
            # apply the uniswap fee of 0.3%
            effective_core_input = core_input * torch.tensor(0.997, dtype=torch.float64)
            
            # compute paired tokens obtained for core assets
            paired_output = (self.paired_reserve * effective_core_input) / (self.core_reserve + effective_core_input)
            
            # get the new token ratio in the pool
            updated_core_reserve = self.core_reserve + effective_core_input
            updated_paired_reserve = self.paired_reserve - paired_output
            
            # --- purely optional check ---
            flag = self.core_reserve * self.paired_reserve <= updated_core_reserve * updated_paired_reserve
            print("Constant product maintained:", flag.item())
            # -----------------------------

            # compute paired token price after swap
            updated_paired_price_usd = (updated_core_reserve / updated_paired_reserve) * self.core_price_usd
            
            # measure paired token price growth (price difference is optional)
            paired_price_diff = updated_paired_price_usd - self.paired_price_usd
            paired_price_ratio = updated_paired_price_usd / self.paired_price_usd
            print(f"Price difference: {paired_price_diff:.10f}")
            print(f"Price ratio     : {paired_price_ratio:.10f}")
            
            loss = torch.square(torch.tensor(1.0 + percent / 100.0, dtype=torch.float64) - paired_price_ratio)
            print(f"loss: {loss:.10f}", end="\n\n")

            # clear gradients 
            optimizer.zero_grad(set_to_none=True)
            # get new gradients
            loss.backward()
            # update parameter
            optimizer.step()

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
    
    """ normalization stuff (simlply breaks the float with our numbers)
    def __normalize_attrs(self) -> None:
        with torch.no_grad():
            # filter pool attributes used in forward pass
            pool_attrs = {key: value for key, value in self.__dict__.items() if (key[0] != "_" and key != "training")}

            # get extreme values
            min_attr, max_attr = min(pool_attrs.values()), max(pool_attrs.values())

            # apply min-max normalization
            for key in self.__dict__:
                if key[0] != "_" and key != "training":
                    self.__dict__[key] = (self.__dict__[key] - min_attr) / (max_attr - min_attr)
            
            # save extremes for restoring
            self.min_attr, self.max_attr = min_attr, max_attr

    def restore_attr(self, attr: torch.Tensor) -> torch.Tensor:
        return attr * (self.max_attr - self.min_attr) + self.min_attr
    """

    def __repr__(self) -> str:
        return (
            f"Core token price   :{self.core_price_usd:>25.10f} $\n"
            f"Paired token price :{self.paired_price_usd:>25.10f} $\n"
            f"Pool liquidity     :{self.pool_liquidity_usd:>25.10f} $\n"
            f"Core reserve       :{self.core_reserve:>25.10f} tokens\n"
            f"Paired reserve     :{self.paired_reserve:>25.10f} tokens"
        )   


with open("data/maga_pool_data.json") as f:
    maga_pool_data = json.load(f)


swap = Swapinator(maga_pool_data)
print(swap)
swap(percent=0.3)

