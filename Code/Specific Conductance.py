# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:09:53 2020

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:53:52 2020

@author: admin
"""
#%%
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
from pandas import datetime
from sklearn.preprocessing import Imputer
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import train_test_split as tts
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pylab as plt #for visualization

#%%

def input_data():
    def parser(x):
        return datetime.strptime(x,'%Y-%m-%d %H:%M')
    dataset = pd.read_csv('Data7.csv',header=0, delimiter=',',index_col=0, parse_dates=[0], date_parser=parser)
    dataset = dataset.fillna(method ='pad') 
    turb = dataset.filter(['Turb(FNU)'], axis=1)
    train_size,test_size = 1000, 1396
    turb_train,turb_test = tts(turb,test_size = test_size, random_state=0, shuffle=False)
    #dataset.fillna(method ='bfill') 

#%%
def check_adfuller(att):
#Perform Augmented Dickey–Fuller test:
    print('Results of Dickey Fuller Test:')
    print("--------For a stationary time series Test statistic is less than critical values-----------")
    dftest = adfuller(att, autolag='AIC')
    
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        
    print(dfoutput)

#%%
def check_mean_std(ts, name):

    rolmean = ts.rolling(window=192).mean()
    rolstd = ts.rolling(window=192).std()
    plt.figure(figsize=(12,8))   
    print(name)
    orig = plt.plot(ts, color='red',label='Original')
    mean = plt.plot(rolmean, color='black', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label = 'Rolling Std')
    plt.xlabel("Date")
    plt.ylabel("Turbidity")
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    plt.show()
#%%
def acf_pacf_plots(dataset):
    ts_diff = dataset - dataset.shift()
    ts_diff.dropna(inplace=True)
    lag_acf = acf(ts_diff, nlags=20)
    lag_pacf = pacf(ts_diff, nlags=20, method='ols')
    
    # ACF
    plt.figure(figsize=(22,10))
    
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    
    # PACF
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
#%%
def arima_model(ts, order):
    # fit model
    model = ARIMA(ts, order=order) # (ARMA) = (p,d,q)
    model_fit = model.fit(disp=0)
    
    # predict
    forecast = model_fit.predict(start=1000, end=2396)
    
    # visualization
    plt.figure(figsize=(12,8))
    plt.plot(turb_test,label = "original")
    plt.figure(figsize=(12,8))
    plt.plot(forecast,label = "predicted")
    plt.title("Turbidity Time Series Forecast")
    plt.xlabel("Date")
    plt.ylabel("Dissolve Oxygen(FNU)")
    plt.legend()
    plt.show()
    
#%%
# Moving average method for Turbidity
def moving_average():
    #turb_logScale = np.log(turb)
    #plt.plot(turb_logScale)

    turb_ma = turb.rolling(window=192).mean() #window size 12 denotes 12 months, giving rolling mean at yearly level
    #sc_movingSTD = sc_logScale.rolling(window=192).std()

#   plt.plot(sc_logScale)
#plt.plot(sc_moving_Average, color='blue')
    
    
    plt.figure(figsize=(12,8))
    plt.plot(turb, color = "red",label = "Original")
    plt.plot(turb_ma, color='black', label = "turb moving_avg_mean")
    plt.title("Turbidity Rolling mean(mg/L) of Potomac River")
    plt.xlabel("Date")
    plt.ylabel("Turb(FNU)")
    plt.legend()
    plt.show()
    
    turb_ma_diff = turb - turb_ma
    #sc_LogScaleMinusMovingAverage.head(100)
    
    turb_ma_diff.dropna(inplace=True)
    #print(sc_rolmean,sc_rolstd)

    check_adfuller(turb_ma_diff['turb(uS)'])
    check_mean_std(turb_ma_diff, 'Turb(FNU)')
#%%
def main():
    input_data()
    check_adfuller(dataset['Turb(FNU)'])
    check_mean_std(dataset['Turb(FNU)'],'\n\nTurbidity')
    acf_pacf_plots(turb)
    arima_model(turb_test,(1,0,1))
    moving_average()
    