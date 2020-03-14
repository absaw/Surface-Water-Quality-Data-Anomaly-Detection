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
from sklearn.metrics import mean_squared_error

#%%

def input_data():
    def parser(x):
        return datetime.strptime(x,'%Y-%m-%d %H:%M')
    dataset = pd.read_csv('Data8.csv',header=0, delimiter=',',index_col=0, parse_dates=[0], date_parser=parser)
    dataset = dataset.fillna(method ='pad') 
    do = dataset.filter(['DO(mg/L)'], axis=1)
    train_size,test_size = 1000, 1396
    do_train,do_test = tts(do,test_size = test_size, random_state=0, shuffle=False)
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
    plt.ylabel("Dissolved Oxygen")
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
    plt.axhline(y=0.4,linestyle='--',color='gray')
    plt.axhline(y=0.3,linestyle='--',color='gray')
    plt.axhline(y=0.2,linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    
    # PACF
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0.4,linestyle='--',color='gray')
    plt.axhline(y=0.3,linestyle='--',color='gray')
    plt.axhline(y=0.2,linestyle='--',color='gray')
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
    plt.plot(do_test,label = "original")
    plt.figure(figsize=(12,8))
    plt.plot(forecast,label = "predicted")
    plt.title("Dissolved Oxygen Time Series Forecast")
    plt.xlabel("Date")
    plt.ylabel("Dissolve Oxygen(FNU)")
    plt.legend()
    plt.show()
    
#%%
# Moving average method for DISSOLVED OXYGEN
def moving_average():
    #do_logScale = np.log(do)
    #plt.plot(do_logScale)

    do_ma = do.rolling(window=192).mean() #window size 12 denotes 12 months, giving rolling mean at yearly level
    #sc_movingSTD = sc_logScale.rolling(window=192).std()

#   plt.plot(sc_logScale)
#plt.plot(sc_moving_Average, color='blue')
    
    
    plt.figure(figsize=(12,8))
    plt.plot(do, color = "red",label = "Original")
    plt.plot(do_ma, color='black', label = "DO moving_avg_mean")
    plt.title("Dissolved Oxygen Rolling mean(mg/L) of Potomac River")
    plt.xlabel("Date")
    plt.ylabel("DO(mg/L)")
    plt.legend()
    plt.show()
    
    do_ma_diff = do - do_ma
    #sc_LogScaleMinusMovingAverage.head(100)
    
    do_ma_diff.dropna(inplace=True)
    #print(sc_rolmean,sc_rolstd)

    check_adfuller(do_ma_diff['DO(mg/L)'])
    check_mean_std(do_ma_diff, 'Dissolved Oxygen(mg/L)')
#%%
#series = read_csv('', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#%%
#X = series.values

diss_train, diss_test = do_train.values, do_test.values
history_do = [x for x in diss_train]
do_predictions = list()
for t in range(len(diss_test)):
	model = ARIMA(history_do, order=(1,0,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	do_predictions.append(yhat)
	obs = diss_test[t]
	history_do.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
#do_test1, do_test2 = tts(do_test,test_size = 298, random_state=0, shuffle=False)
error = mean_squared_error(diss_test, do_predictions)
print('Test MSE: %.3f' % error)
# plot
plt.figure(figsize=(12,8))
pyplot.plot(diss_test, label = "Original")
pyplot.plot(do_predictions, color='red',label='Predicted')
plt.xlabel("Datatime Index ")
plt.ylabel("Dissolved Oxygen Values")
plt.title('Dissolved Oxygen Forecast')
plt.legend()
plt.show()
#%%
def main():
    input_data()
    check_adfuller(dataset['DO(mg/L)'])
    check_mean_std(dataset['DO(mg/L)'],'\n\nDissolved Oxygen')
    acf_pacf_plots(do)
    arima_model(do_test,(1,0,1))
    moving_average()
    