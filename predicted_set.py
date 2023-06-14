import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
dataset_train = pd.read_csv("currency_monthly_BTC_CNY.csv")
dataset_train.head()
dataset_train.info()
training_set = dataset_train.iloc[:,1:2].values
print(training_set)
print(training_set.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))
scaled_training_set = scaler.fit_transform(training_set)
scaled_training_set
X_train = []
Y_train = []
for i in range(5,25):
    X_train.append(scaled_training_set[i-5:i,0])
    Y_train.append(scaled_training_set[i,0])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print(X_train.shape)
print(Y_train.shape)

X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))
X_train.shape

#Building the Model

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = "adam", loss = "mean_squared_error")
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)
dataset_test = pd.read_csv("currency_monthly_BTC_CNY.csv")
actual_stock_price = dataset_test.iloc[:,1:2].values

dataset_test_total = pd.concat((dataset_train['openinUSD'], dataset_test['openinUSD']), axis = 0)
inputs = dataset_test_total[len(dataset_test_total) - len(dataset_test)-33:].values

inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(5,33):
    X_test.append(inputs[i-5:i,0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# Implementation and Predicting Values

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


# Plotting the Map

plt.plot(actual_stock_price, color = "red", label = "Daily Adjusted BTC Price")
plt.plot(predicted_stock_price, color = "blue", label = "Predicted BTC Price")
plt.title("Bitcoin Price Prediction")
plt.xlabel("Time")
plt.ylabel("Bitcoin Price")
plt.legend()