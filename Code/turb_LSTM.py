# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:53:44 2020

@author: admin
"""
#%%

import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split as tts
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from math import sqrt
import numpy

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
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# load dataset
def parser(x):
	return datetime.strptime(x,'%Y-%m-%d %H:%M')
#series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
dataset = pd.read_csv('./Dataset/Data7.csv',header=0, delimiter=',',index_col=0, parse_dates=[0], date_parser=parser)  
dataset = dataset.fillna(method ='pad') 
turb = dataset.filter(['Turb(FNU)'], axis=1)
# transform to supervised learning
X = turb.values
supervised = timeseries_to_supervised(X, 1)
print(supervised.head())
#%%

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df
'''
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
'''
# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	#train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	#test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train):
    
	#X, y = train[:, 0:-1], train[:, -1]
	#X = X.reshape(X.shape[0], 1, X.shape[1])
    batch_size = 96 #batch_size:no of entries sampled in one go
    #epoch = 3000
    neurons = 200
    n_input = 12
    n_features = 
    generator = TimeseriesGenerator(train, train, length = n_input, batch_size = batch_size)
    model = Sequential()
	#model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(LSTM(neurons, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dropout(0.15))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    '''
    for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	'''
    model.fit_generator(generator, epochs=180)
    return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# load dataset
def parser(x):
	return datetime.strptime(x,'%Y-%m-%d %H:%M')
#series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
dataset = pd.read_csv('./Dataset/Data7.csv',header=0, delimiter=',',index_col=0, parse_dates=[0], date_parser=parser)  
dataset = dataset.fillna(method ='pad')
turb = dataset.filter(['Turb(FNU)'], axis=1)

# transform to supervised learning
X = turb.values
supervised = timeseries_to_supervised(X, 1)
supervised_values = supervised.values
print(supervised.head())
'''
# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)
'''

# split data into train and test-sets
#train, test = supervised_values[0:-12], supervised_values[-12:]
train_size,test_size = 4000, 1171#in paper given as 3169
#train_size,test_size = 1000, 1396
turb_train,turb_test = tts(turb,train_size = train_size, random_state=0, shuffle=False)
turb_train = turb_train.values
turb_test = turb_test.values
# transform the scale of the data
scaler, train_scaled, test_scaled = scale(turb_train, turb_test)

# fit the model
lstm_model = fit_lstm(train_scaled)#96 entries of one day
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)