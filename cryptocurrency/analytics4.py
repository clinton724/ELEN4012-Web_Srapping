import sys
sys.path.insert(0,'../../')
from db import connection, cursor
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt


cursor.execute("SELECT Date, Market_Cap, Volume, Open_Price, Close_Price from HistoricalData where cryptocurrency = 'Bitcoin (BTC)'")
data = cursor.fetchall()
connection.commit()
columns = ["date", "market_cap", "volume", "open_price", "close_price"]
df = pd.DataFrame(data, columns=columns)

#preprocessing of data
df.index = df.pop('date')
df_ClosePrice = df.filter(['close_price'])
df_ClosePrice = df_ClosePrice[:-1]
Dataset = df_ClosePrice.values
train_len = math.ceil(len(Dataset) * 0.8)
# scale the data
scaler =  MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(Dataset)
train_data = scaled_data[0:train_len,:]
X_train = []
y_train = []
for i in range(6, len(train_data)):
    X_train.append(train_data[i-6:i,0])
    y_train.append(train_data[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)
#Reshape the data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape =(X_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='relu'))
#Compile the model
model.compile(optimizer= 'adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
model.fit(X_train, y_train, batch_size=2, epochs=5)

#Testing the model
test_data = scaled_data[train_len-6:,:]
X_test = []
y_test = Dataset[train_len:,:]
for i in range(6, len(test_data)):
    X_test.append(test_data[i-6:i,0])
#change the X test to numpy array for the model
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Predict the six values using the model
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
#Get errors to quantify the acurracy of the model
#Mean absolute error
mae = np.mean(np.abs(predictions-y_test))
print(mae)
#mean squared error
mse = np.mean(((predictions - y_test)**2))
print(mse)
#plot the data
train = df[:train_len]
df = df[:-1]
valid = df[train_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('RNN with LSTM Model')
plt.plot(train['close_price'])
plt.plot(valid[['close_price','Predictions']])
plt.xlabel('Date', fontsize = 18)
plt.xticks(rotation = 45)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right') 
plt.show()

