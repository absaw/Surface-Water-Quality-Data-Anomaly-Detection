# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:09:53 2020

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:53:52 2020

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
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
#%%

#def input_data():
def parser(x):
    return datetime.strptime(x,'%Y-%m-%d %H:%M')
dataset = pd.read_csv('./Dataset/Data7.csv',header=0, delimiter=',',index_col=0, parse_dates=[0], date_parser=parser)
dataset = dataset.fillna(method ='pad') 
turb = dataset.filter(['Turb(FNU)'], axis=1)
train_size,test_size = 1920, 3251#in paper given as 3169
#train_size,test_size = 1000, 1396
turb_train,turb_test = tts(turb,train_size = train_size, random_state=0, shuffle=False)
#dataset.fillna(method ='bfill') 

#%%
def check_adfuller(att):

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
    ts = turb_train
    order=(1,0,1)
    model = ARIMA(ts, order=order) # (ARMA) = (p,d,q)
    model_fit = model.fit(disp=0)
    #print summary of fit model
    print(model_fit.summary())
    # predict
    #forecast = model_fit.forecast()[0]
    forecast2 = model_fit.predict(start=1000, end=2396)
    
    # visualization
    plt.figure(figsize=(12,8))
    plt.plot(turb_test,label = "original")
    #plt.figure(figsize=(12,8))
    plt.plot(forecast2,label = "predicted")
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
#X = series.values

train, test = turb_train.values, turb_test.values
turb_history = [x for x in train]
turb_predictions = list()
turb_diff = list()
k = 1921
for t in range(len(test)):
    model = ARIMA(turb_history, order=(1,0,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    turb_predictions.append(yhat)
    obs = test[t]
    turb_history.append(obs)
    diff = obs-yhat
    turb_diff.append(diff)
    print('TurbParameter Index= %d, predicted=%f, expected=%f, difference = %f' % (k, yhat, obs,diff))
    k=k+1
    if(k==3000):
        break
#test1, test2 = tts(test,test_size = 337, random_state=0, shuffle=False)
error = mean_squared_error(test1, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.figure(figsize=(12,8))
pyplot.plot(test1, label = "Original")
pyplot.plot(predictions, color='red',label='Predicted')
plt.xlabel("Datatime Index ")
plt.ylabel("Turbidity Values")
plt.title('Turbidity Forecast')
plt.legend()
plt.show()
#%%
# SARIMA example
from statsmodels.tsa.statespace.sarimax import SARIMAX
from random import random

# fit model
model = SARIMAX(turb_test, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
model_fit = model.fit(disp=False)
# make prediction
print(model_fit.summary().tables[1])
#Prediction plot
model_fit.plot_diagnostics(figsize=(18, 8))
plt.show()
yhat = model_fit.predict(len(turb_test), len(turb_test))
print(yhat)


pred = model_fit.get_prediction(start=1500, dynamic=False)
pred_ci = pred.conf_int()
plt.figure(figsize=(20,10))
ax = turb_test.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 4))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Retail_sold')
plt.legend()
plt.show()

y_forecasted = pred.predicted_mean
y_truth = y['2018-06-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))
#%%
#Isolation Forest Model Prediction
model= IsolationForest(n_estimators=100, max_samples=256)
#model = IsolationForest(behaviour = 'new')
model.fit(turb_train)
turb_pred = model.predict(turb_test)
print("Valid cases accuracy:", list(turb_pred).count(1)/turb_pred.shape[0])
Fraud_pred = model.predict(turb_test)


#%%
plt.figure(figsize=(12,8))
plt.hist(test, normed=True)

plt.xlim([-1, 10])

plt.show()
#%%

isolation_forest = IsolationForest(n_estimators=100)

isolation_forest.fit(train.reshape(-1, 1))

#xx = np.linspace(-6, 6, 100).reshape(-1,1)
xx = test.reshape(-1,1)
anomaly_score = isolation_forest.decision_function(xx)

outlier = isolation_forest.predict(xx)

plt.figure(figsize=(12,8))

plt.plot(xx, anomaly_score, label='anomaly score')

plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 
v
                 where=outlier==-1, color='r', 

                 label='outlier region')

plt.legend()

plt.ylabel('anomaly score')

plt.xlabel('Turbidity Value Frequency')

plt.xlim([-1, 10])

plt.show()
#%%
def main():
    #input_data()
    check_adfuller(dataset['Turb(FNU)'])
    check_mean_std(dataset['Turb(FNU)'],'\n\nTurbidity')
    acf_pacf_plots(turb)
    arima_model(turb_test,(1,0,1))
    moving_average()
    