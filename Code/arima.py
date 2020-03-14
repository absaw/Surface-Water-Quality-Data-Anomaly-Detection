# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:12:59 2020

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
def parser(x):
    return datetime.strptime(x,'%Y-%m-%d %H:%M')
dataset = pd.read_csv('Data7.csv',header=0, delimiter=',',index_col=0, parse_dates=[0], date_parser=parser)
dataset = dataset.fillna(method ='pad') 
#dataset.fillna(method ='bfill') 
#%%
dataset.plot()
dataset.show()
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
check_adfuller(dataset['SC(uS)'])
check_adfuller(dataset['Turb(FNU)'])
check_adfuller(dataset['DO(mg/L)'])

#%%
sc = dataset.filter(['SC(uS)'], axis=1)
turb = dataset.filter(['Turb(FNU)'], axis=1)
do = dataset.filter(['DO(mg/L)'], axis=1)
#%%
def check_mean_std(ts, name):

    rolmean = ts.rolling(window=96).mean()
    rolstd = ts.rolling(window=96).std()
    plt.figure(figsize=(22,10))   
    print(name)
    orig = plt.plot(ts, color='red',label='Original')
    mean = plt.plot(rolmean, color='black', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label = 'Rolling Std')
    plt.xlabel("Date")
    plt.ylabel("Mean Temperature")
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    plt.show()
#%%
def acf_pacf_plots(datset):
    dataset = sc
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
    ts = sc_train
    model = ARIMA(ts, order=(1,0,1)) # (ARMA) = (p,d,q)
    model_fit = model.fit(disp=0)
    
    # predict
    forecast = model_fit.predict(start=1919, end=5171)
    
    # visualization
    plt.figure(figsize=(12,8))
    plt.plot(sc_test,label = "original")
    plt.figure(figsize=(12,8))
    plt.plot(forecast,label = "predicted")
    plt.title("Turbidity Time Series Forecast")
    plt.xlabel("Date")
    plt.ylabel("Turbidity(FNU)")
    plt.legend()
    plt.show()
#%%
#X = series.values

sp_train, sp_test = sc_train.values, sc_test.values
history_sc = [x for x in train]
sc_predictions = list()
for t in range(len(sp_test)):
	model = ARIMA(history_sc, order=(1,0,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	sc_predictions.append(yhat)
	obs = sp_test[t]
	history_sc.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
sp_test1, sp_test2 = tts(sc_test,test_size = 3232, random_state=0, shuffle=False)
error = mean_squared_error(do_test1, do_predictions)
print('Test MSE: %.3f' % error)
# plot
plt.figure(figsize=(12,8))
pyplot.plot(test1, label = "Original")
pyplot.plot(predictions, color='red',label='Predicted')
plt.xlabel("Datatime Index ")
plt.ylabel("Dissolved Oxygen Values")
plt.title('Dissolved Oxygen Forecast')
plt.legend()
plt.show()
#%%
# Moving average method for TURBIDITY
window_size = 192
turb_ma = turb.rolling(window=window_size).mean()
plt.figure(figsize=(22,10))
plt.plot(turb, color = "red",label = "Original")
plt.plot(turb_ma, color='black', label = "moving_avg_mean")
plt.title("Turbidity(FNU) of Potomac River")
plt.xlabel("Date")
plt.ylabel("Turbidity")
plt.legend()
plt.show()
turb_moving_avg_diff = turb - turb_ma
turb_moving_avg_diff.dropna(inplace=True) # first 6 is nan value due to window size

# check stationary: mean, variance(std)and adfuller test
check_mean_std(turb_moving_avg_diff, "Turbidity")
check_adfuller(turb_moving_avg_diff['Turb(FNU)'])

#p and q for turbidity

acf_pacf_plots(turb)

#%%
check_adfuller(sc['SC(uS)'])
check_mean_std(sc['SC(uS)'])
#%%
# Moving average method for Specific Conductance

#sc_logScale = np.log(sc)
#plt.plot(sc_logScale)

sc_ma = sc.rolling(window=192).mean() #window size 12 denotes 12 months, giving rolling mean at yearly level
#sc_movingSTD = sc_logScale.rolling(window=192).std()

#plt.plot(sc_logScale)
#plt.plot(sc_moving_Average, color='blue')
plt.figure(figsize=(22,10))
plt.plot(sc, color = "red",label = "Original")
plt.plot(sc_ma, color='black', label = "moving_avg_mean")
plt.title("Specific Conductance(uS) of Potomac River")
plt.xlabel("Date")
plt.ylabel("Specific Conductance")
plt.legend()
plt.show()

sc_ma_diff = sc - sc_ma
#sc_LogScaleMinusMovingAverage.head(100)

sc_ma_diff.dropna(inplace=True)
#print(sc_rolmean,sc_rolstd)

check_adfuller(sc_ma_diff['SC(uS)'])
check_mean_std(sc_ma_diff, "SC(uS)")

#%%
# Moving average method for DISSOLVED OXYGEN

do_ma = do.rolling(window=192).mean() 
plt.figure(figsize=(22,10))
plt.plot(do, color = "red",label = "Original")
plt.plot(do_ma, color='black', label = "DO moving_avg_mean")
plt.title("Dissolved Oxygen (mg/L) of Potomac River")
plt.xlabel("Date")
plt.ylabel("DO(mg/L)")
plt.legend()
plt.show()

#do_ma_diff = do - do_ma
#sc_LogScaleMinusMovingAverage.head(100)

do_ma_diff.dropna(inplace=True)
#print(sc_rolmean,sc_rolstd)

check_adfuller(do_ma_diff['DO(mg/L)'])
check_mean_std(sc_ma_diff, 'Dissolved Oxygen(mg/L)')
#%%
#Determine rolling statistics
sc_rolmean = sc.rolling(window=96).mean() #window size 12 denotes 12 months
sc_rolstd = sc.rolling(window=96).std()

print(sc_rolmean,sc_rolstd)

orig = plt.plot(sc, color='blue', label='Original')
mean = plt.plot(sc_rolmean, color='red', label='Rolling Mean')
std = plt.plot(sc_rolstd, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)
#%%
#PLotting all individual variables
plt.figure(figsize=(17,8)) 
plt.plot(sc,label="Specific Conductance(uS) of Potomac River",color='red')
plt.plot(turb,label="Turbidity(FNU) of Potomac River",color='black')
plt.plot(do,label="Dissolved Oxygen(mg/L)",color='green')
plt.title("Dataset (mg/L) of Potomac River")
plt.xlabel("Date")
plt.ylabel("DO(mg/L)")
plt.legend()
plt.show()
plt.show()
#%%
#Auto Correlation Plots
autocorrelation_plot(sc)
pyplot.show()

autocorrelation_plot(turb)
pyplot.show()

autocorrelation_plot(do)
pyplot.show()
#%%
#Auto Correlation
from pandas.plotting import lag_plot

lag_plot(sc)
pyplot.show()

lag_plot(turb)
pyplot.show()

lag_plot(do)
pyplot.show()
#%%
#ACF Plots
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(sc)

plot_acf(turb, label="Turbidity")

plot_acf(do)

#%%
'''
Extra knowledge
pd.set_option('display.max_columns', 5) - To set displayed no. of columns

data = pd.read_csv('file1.csv', error_bad_lines=False) - To ignore error ridden lines
'''
#%%

sc_v = sc.values
turb_v = turb.values
do_v = do.values

#%%

from scipy.stats import zscore
sc_z = sc.apply(zscore)
turb_z = turb.apply(zscore)

#%%
#ELIMINATING THE MISSING VALUES

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(sc_v[:,:])
sc_v=imputer.transform(sc_v)

imputer = imputer.fit(turb_v[:,:])
turb_v=imputer.transform(turb_v)

#%%
#Splitting dataset to training and test set
'''
Training Set- from which the model will learn from
Test -with which it will compare itself and check itself


'''
#from sklearn.model_selection import train_test_split as tts
train_size = 1920
test_size = 3252
sc_train,sc_test = tts(sc,test_size = test_size, random_state=0, shuffle=False)
do_train,do_test = tts(do,test_size = test_size, random_state=0, shuffle=False)
turb_train,turb_test = tts(turb,test_size =test_size, random_state=0, shuffle=False)

#sc_train, sc_test = sc_v[0:train_size,:], sc_v[train_size:5171,:]

#%%

#p,d,q  p = periods taken for autoregressive model
#d -> Integrated order, difference
# q periods in moving average model
sc_arima = ARIMA(sc_train,order=(12,0,0))
sc_arima_fit = sc_arima.fit(disp=-1)
plt.plot(sc_train, color = 'blue')
plt.plot(sc_arima_fit.fittedvalues, color = 'red', figsize=(20,20))
print(sc_arima_fit.aic)

#%%
# predict
start_index = parser("2017-02-16 00:00")
end_index = "2017-03-21 23:45"
forecast = sc_arima_fit.predict(start=1919, end=4000)

plt.figure(figsize=(22,10))
plt.plot(sc_test,label = "original")
plt.plot(forecast,label = "predicted", color = 'red')

#%%

predictions = sc_arima_fit.predict(start=1919, end=3000)
print(predictions)

#%%
import itertools
p=d=q=range(0,5)
pdq = list(itertools.product(p,d,q))

#%%
import warnings
warnings.filterwarnings('ignore')
for param in pdq:
    try:
        sc_arima = ARIMA(sc_train,order=param)
        sc_arima_fit = sc_arima.fit()
        print(param, sc_arima_fit.aic)
        '''
        model_arima = ARIMA(train,order=param)
        model_arima_fit = model_arima.fit()
        print(param,model_arima_fit.aic)
        '''
    except:
        continue
    
#%%
mp.plot(sc_test)
mp.plot(forecast,color='red')

#%%
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
sc_ar = AR(sc_train)
sc_ar_fit = sc_ar.fit()

predictions = sc_ar_fit.predict(start=train_size, end=5171)
#%%
import warnings
warnings.filterwarnings('ignore')
for param in pdq:
    try:
        turb_arima = ARIMA(turb_train,order=param)
        turb_arima_fit = turb_arima.fit()
        print(param, turb_arima_fit.aic)
        '''
        model_arima = ARIMA(train,order=param)
        model_arima_fit = model_arima.fit()
        print(param,model_arima_fit.aic)
        '''
    except:
        continue
#%%
import warnings
warnings.filterwarnings('ignore')
for param in pdq:
    try:
        do_arima = ARIMA(do_train,order=param)
        turb_arima_fit = do_arima.fit()
        print(param, turb_arima_fit.aic)
        '''
        model_arima = ARIMA(train,order=param)
        model_arima_fit = model_arima.fit()
        print(param,model_arima_fit.aic)
        '''
    except:
        continue

