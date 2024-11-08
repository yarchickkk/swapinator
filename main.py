import sys
import json
import torch
from torch import nn as nn


ERROR = "\033[1;31m(swapinator)\033[0m"
SPACE = " " * 12
LEARNING_RATE = 10.0
INITIAL_DEPOSIT = 400.0
ITERATIONS_NUMBER = 1000


class Swapinator(nn.Module):
    
    def __init__(self, pool_data: dict) -> None:
        super().__init__()

        # get pool attributes safely
        self.__validate_and_unpack(pool_data)
        # set initial deposit value - the optimized parameter, requires gradient
        self.input_usd = nn.Parameter(torch.tensor(INITIAL_DEPOSIT, dtype=torch.float64, requires_grad=True))

    def forward(self, percent: float, verbosity: bool = False) -> float:
        # perhaps adding data dependent lr setting
        optimizer = torch.optim.AdamW(params=[self.input_usd], lr=LEARNING_RATE, weight_decay=1e-4)
        loss = torch.tensor(float("inf"), dtype=torch.float64)
        
        for i in range(ITERATIONS_NUMBER):
            # convert USD to core tokens
            core_input = self.input_usd / self.core_price_usd
            # apply the uniswap fee of 0.3%
            effective_core_input = core_input * torch.tensor(0.997, dtype=torch.float64)
            
            # compute paired tokens obtained for core assets
            paired_output = (self.paired_reserve * effective_core_input) / (self.core_reserve + effective_core_input)
            
            # get the new token ratio in the pool
            updated_core_reserve = self.core_reserve + effective_core_input
            updated_paired_reserve = self.paired_reserve - paired_output

            # compute paired token price after swap
            updated_paired_price_usd = (updated_core_reserve / updated_paired_reserve) * self.core_price_usd
            
            # measure paired token price growth (price difference used for logging only)
            paired_price_diff = updated_paired_price_usd - self.paired_price_usd
            paired_price_ratio = updated_paired_price_usd / self.paired_price_usd
            
            # casual mean squared error, increased artifically for better optimizer perfomance
            loss = torch.square(torch.tensor(1.0 + percent / 100.0, dtype=torch.float64) - paired_price_ratio) * 1e6

            # optional logging
            if verbosity is True:
                message = (
                    f"{i} Search Step:\n"
                    f"• Paired Token Price Difference: {paired_price_diff:.10f} $\n"
                    f"• Paired Token Price Ratio     : {paired_price_ratio:.10f}\n"
                    f"• Loss Achieved                : {loss:.10f}\n"
                )
                print(message)

            # clear gradients, obtain new ones and make a step! 
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        self.loss_achieved = loss.item()
        self.percent_achieved = ((paired_price_ratio - 1.0) * 100.0).item()
        # return optimized deposit as float
        return self.input_usd.item()

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
            f"\n{'='*40}\n"
            f"Swapinator Object Summary:\n"
            f"{'='*40}\n"
            f"• Core Token Price  :{self.core_price_usd:>25.10f} $\n"
            f"• Paired Token Price:{self.paired_price_usd:>25.10f} $\n"
            f"• Pool Liquidity    :{self.pool_liquidity_usd:>25.10f} $\n"
            f"• Core Reserve      :{self.core_reserve:>25.10f} tokens\n"
            f"• Paired Reserve    :{self.paired_reserve:>25.10f} tokens\n"
            f"{'='*40}\n"
        )   


with open("data/groggo_pool_data.json") as f:
    maga_pool_data = json.load(f)


swap = Swapinator(maga_pool_data)
print(swap)
deposit = swap(percent=0.69)
print(f"Found Deposit: {deposit:.2f} $")
print(f"Achieved Percent: {swap.percent_achieved:.10f} %")
print(f"Achieved Loss   : {swap.loss_achieved:.10f}")
print()
