#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 21:47:39 2023

@author: tarunvannelli
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import yfinance as yf
from nsetools import Nse
import pandas_market_calendars as mcal
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly


nifty50_stocks = ['TATAMOTORS.NS','HCLTECH.NS','POWERGRID.NS','HDFCLIFE.NS','EICHERMOT.NS',
                       'BAJAJ-AUTO.NS','TECHM.NS','NTPC.NS','SBILIFE.NS','AXISBANK.NS',
                       'LT.NS','KOTAKBANK.NS','BRITANNIA.NS','INFY.NS','WIPRO.NS',
                       'HEROMOTOCO.NS','NESTLEIND.NS','ADANIENT.NS','HINDALCO.NS','TCS.NS',
                       'ICICIBANK.NS','DRREDDY.NS','BHARTIARTL.NS','RELIANCE.NS','JSWSTEEL.NS',
                       'GRASIM.NS','ADANIPORTS.NS','TATASTEEL.NS','HDFCBANK.NS','TATACONSUM.NS',
                       'APOLLOHOSP.NS','HDFC.NS','COALINDIA.NS','ASIANPAINT.NS','TITAN.NS',
                       'CIPLA.NS','ITC.NS','MARUTI.NS','SBIN.NS','UPL.NS',
                       'HINDUNILVR.NS','ULTRACEMCO.NS','ONGC.NS','INDUSINDBK.NS','DIVISLAB.NS',
                       'BPCL.NS','M&M.NS','SUNPHARMA.NS','BAJAJFINSV.NS','BAJFINANCE.NS']



initial_equity = st.number_input('Insert a number',value=1000000)

start_date = st.date_input("Select Start Date",dt.date(2020,10,1))
end_date = st.date_input("Select End Date",min_value=start_date,max_value=dt.datetime.now())
#initial_equity = 1000000
#start_date = dt.datetime(2020, 10, 1)
#end_date = dt.datetime.now()


### BEnchMark

equity_curves_df = pd.DataFrame()

def calculate_equity_curve(stock_open_prices, stock_close_prices, initial_equity):
    num_stocks = len(nifty50_stocks)
    allocation_per_stock = initial_equity / num_stocks
    shares_held = allocation_per_stock / stock_open_prices[0]
    position_values = shares_held * stock_close_prices
    return position_values


for s in nifty50_stocks:
    # Fetch historical stock prices
    stock_prices = yf.download(s, start=start_date, end=end_date)

    # Extract the required data
    stock_dates = stock_prices.index
    stock_open_prices = stock_prices['Open'].values
    print(len(stock_open_prices))
    stock_close_prices = stock_prices['Close'].values
    print(len(stock_close_prices))

    # Calculate equity curve for the stock
    equity_curve = calculate_equity_curve(stock_open_prices, stock_close_prices, initial_equity)
    equity_curves_df[s] = equity_curve



equity_curves_df_copy = equity_curves_df.copy()

equity_curves_df_copy['Date'] = stock_dates

equity_curves_df_copy.set_index(equity_curves_df_copy['Date'],inplace=True)

equity_curves_df_copy = equity_curves_df_copy.drop('Date', axis=1)

#BEC - benchmark equity_curve
equity_curves_df_copy['BEC'] = equity_curves_df_copy.sum(axis=1)




###Sample Strategy

no_of_days = st.number_input('Insert No of Days',value=100)

def calculate_returns(prices):
    returns = prices[-1] / prices[-no_of_days] - 1
    return returns

returns_dict = {}

for s in nifty50_stocks:
    # Fetch historical stock prices
    stock_prices_ss = yf.download(s, start=start_date, end=end_date)
    
    # Extract the required data
    stock_close_prices_ss = stock_prices_ss['Close'].values

    # Calculate returns for the stock
    returns = calculate_returns(stock_close_prices_ss)

    # Store the returns in the dictionary
    returns_dict[s] = returns
    


no_of_stocks = st.number_input('Insert No of Stocks',min_value=5,value=10)

top_stocks = sorted(returns_dict, key=returns_dict.get, reverse=True)[:no_of_stocks]

top_10_stocks = []

for stock in top_stocks:
    top_10_stocks.append(stock)



def calculate_equity_curve(stock_open_prices, stock_close_prices, initial_equity):
    
    num_stocks = len(top_10_stocks)
    
    allocation_per_stock = initial_equity / num_stocks

    shares_held = allocation_per_stock / stock_open_prices[0]
    print(shares_held)
    
    position_values = shares_held * stock_close_prices
    return position_values

equity_curves_ss_df = pd.DataFrame()

for s in top_10_stocks:
    # Fetch historical stock prices
    stock_prices = yf.download(s, start=start_date, end=end_date)
    print(s)

    # Extract the required data
    stock_dates = stock_prices.index
    
    stock_open_prices = stock_prices['Open'].values
    print(len(stock_open_prices))
    
    stock_close_prices = stock_prices['Close'].values
    print(len(stock_close_prices))

    # Calculate equity curve for the stock
    equity_curve_ss = calculate_equity_curve(stock_open_prices, stock_close_prices, initial_equity)
    equity_curves_ss_df[s] = equity_curve_ss
    

equity_curves_ss_df_copy = equity_curves_ss_df.copy()

equity_curves_ss_df_copy['Date'] = stock_dates

equity_curves_ss_df_copy.set_index(equity_curves_ss_df_copy['Date'],drop=True,inplace=True)

equity_curves_ss_df_copy = equity_curves_ss_df_copy.drop('Date', axis=1)

#BEC - benchmark equity_curve
equity_curves_ss_df_copy['BEC_SS'] = equity_curves_ss_df_copy.sum(axis=1)


###Nifty Index Equity Curve

stock_prices_nifty = yf.download('^NSEI', start=start_date, end=end_date)

stock_open_prices_nifty = stock_prices['Open'].values

stock_close_prices_nifty = stock_prices['Close'].values

def calculate_equity_curve_nifty(stock_open_prices, stock_close_prices, initial_equity):

    shares_held = initial_equity / stock_open_prices[0]
    
    position_values_nifty = shares_held * stock_close_prices
    return position_values_nifty


equity_curve_nifty = calculate_equity_curve_nifty(stock_open_prices_nifty, stock_close_prices_nifty, initial_equity)

equity_curves_nifty_df = pd.DataFrame()

equity_curves_nifty_df['Nifty'] = equity_curve_nifty

equity_curves_nifty_df['Date'] = stock_dates

equity_curves_nifty_df.set_index(equity_curves_nifty_df['Date'],drop=True,inplace=True)

equity_curves_nifty_df = equity_curves_nifty_df.drop('Date', axis=1)


st.write("Equity Curves")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=stock_dates, y=equity_curves_nifty_df['Nifty'], name='Nifty'))
fig.add_trace(go.Scatter(
    x=stock_dates, y=equity_curves_ss_df_copy['BEC_SS'], name='Sample_Strategy'))
fig.add_trace(go.Scatter(
    x=stock_dates, y=equity_curves_df_copy['BEC'], name='BenchMark_Strategy'))

fig.update_layout(xaxis_title='Date',
                  yaxis_title='Price (INR)')
st.plotly_chart(fig)



st.write("Top Selected Stocks:",top_10_stocks)


### Return Stocks

equity_curves_nifty_df["Daily_Returns"] = (equity_curves_nifty_df["Nifty"]/equity_curves_nifty_df["Nifty"].shift(1))-1

equity_curves_df_copy["Daily_Returns"] = (equity_curves_df_copy["BEC"]/equity_curves_df_copy["BEC"].shift(1))-1

equity_curves_ss_df_copy["Daily_Returns"] = (equity_curves_ss_df_copy["BEC_SS"]/equity_curves_ss_df_copy["BEC_SS"].shift(1))-1

##CGPR

def calculate_cagr(equity_curve, start_date, end_date):
    num_years = (end_date - start_date).days / 365  # Number of years
    final_value = equity_curve[-1]  # Value on final day
    beginning_value = equity_curve[0]  # Value on beginning day
    
    cagr = (((final_value / beginning_value) ** (1/num_years)) - 1) * 100
    return round(cagr,3)



Metrics_Df = pd.DataFrame()

names = ['BenchMark','Sample','Nifty']

cagr = [calculate_cagr(equity_curves_df_copy['BEC'],start_date, end_date),calculate_cagr(equity_curves_ss_df_copy['BEC_SS'], start_date, end_date) ,calculate_cagr(equity_curves_nifty_df['Nifty'], start_date, end_date),]

volt = [ (np.std(equity_curves_df_copy['Daily_Returns'])**(1/252))*100, (np.std(equity_curves_ss_df_copy['Daily_Returns'])**(1/252))*100  , (np.std(equity_curves_nifty_df['Daily_Returns'])**(1/252))*100]

sr = [(np.mean(equity_curves_df_copy['Daily_Returns'])/np.std(equity_curves_df_copy['Daily_Returns']))**(1/252),
      (np.mean(equity_curves_ss_df_copy['Daily_Returns'])/np.std(equity_curves_ss_df_copy['Daily_Returns']))**(1/252),
      (np.mean(equity_curves_nifty_df['Daily_Returns'])/np.std(equity_curves_nifty_df['Daily_Returns']))**(1/252)
     ]


Metrics_Df['Index'] = names

Metrics_Df['CAGR%'] = cagr

Metrics_Df['Volatility%'] = volt

Metrics_Df['Sharpe'] = sr


st.table(Metrics_Df)

