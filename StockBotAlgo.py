#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class StockBot:

    def __init__(self, balance=100000, nu_of_stocks=0):
        self.balance = balance
        self.min_bal = balance
        self.nu_of_stocks = nu_of_stocks
        self.profit = []
        self.investment = []
        self.last_day_price = None  

    def stockbot_decision(self, backward_look, forward_look):
        curr_price = backward_look[-1]
        del_i = np.sign(forward_look[0] - curr_price)
        del_i_plusone = np.sign(forward_look[1] - forward_look[0])
        delta = del_i_plusone - del_i
        new_arr = np.concatenate((backward_look, forward_look))
        x_ticks = range(0, len(backward_look) + 2)
        curr_pos = len(backward_look) - 1
        curr_price = new_arr[curr_pos]
        plt.figure(figsize=(10, 8))
        
        self.last_day_price = curr_price

        if delta == 2 and self.nu_of_stocks > 0:
            decision = 'Sell'
            self.balance += curr_price
            self.min_bal = min(self.balance, self.min_bal)
            diff = curr_price - self.investment[-1]
            self.profit.append(diff)
            print(decision)
        elif delta == -2 and self.balance > curr_price:
            decision = 'Buy'
            self.balance -= curr_price
            self.min_bal = min(self.balance, self.min_bal)
            self.nu_of_stocks += 1
            self.investment.append(curr_price)
            print(decision)
        else:
            decision = 'Hold'
            print(decision)

        return self.balance

    def calculate_roi(self):
        if self.last_day_price is not None:
            total_profit = -self.min_bal + 100000 + (self.nu_of_stocks * self.last_day_price)
            total_investment = 100000 - self.min_bal
            roi = ((total_profit * 100) / total_investment)/10
            print("ROI : {:.2f} %".format(roi))
            return roi
        else:
            print("No data available to calculate ROI.")
            return None



# In[4]:


# Load data from CSV
asian = pd.read_csv('ASIANPAINT.csv')
asian_close = asian['Close']

asian_stocks = StockBot(nu_of_stocks=0)

# Iterate through the data
for i in range(3400, len(asian_close) - 52):
    backward = asian_close[i:i+50].tolist()
    forward = asian_close[i+51:i+53].tolist()

    asian_stocks.stockbot_decision(backward, forward)

# Calculate and print ROI
asian_stocks.calculate_roi()
print("Final Balance:", asian_stocks.balance)


# In[ ]:




