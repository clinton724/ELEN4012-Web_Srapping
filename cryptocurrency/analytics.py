import sys
sys.path.insert(0, '../')
from db import connection, cursor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
from sklearn.preprocessing import MinMaxScaler


cursor.execute("select * from HistoricalData where Cryptocurrency = 'Bitcoin (BTC)'")
Data = cursor.fetchall()
connection.commit()
columns = ["cryptocurrency", "date", "market_cap", "Volume", "Open", "Close"]
df = pd.DataFrame(Data, columns=columns)
close_data = df.filter(['Open'])
dataset = close_data.values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
training_data_len = math.ceil(len(dataset)*.7)
train_data = scaled_data[0:training_data_len,:]

x_train_data=[]
y_train_data =[]
for i in range(5,len(train_data)):
    x_train_data=list(x_train_data)
    y_train_data=list(y_train_data)
    x_train_data.append(train_data[i-5:i,0])
    y_train_data.append(train_data[i,0])
 
    # 6. Converting the training x and y values to numpy arrays
    x_train_data1, y_train_data1 = np.array(x_train_data), np.array(y_train_data)
 
    # 7. Reshaping training s and y data to make the calculations easier
    x_train_data2 = np.reshape(x_train_data1, (x_train_data1.shape[0],x_train_data1.shape[1],1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data2.shape[1],1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train_data2, y_train_data1, batch_size=1, epochs=1)

# 1. Creating a dataset for testing
test_data = scaled_data[training_data_len - 5: , : ]
x_test = []
y_test =  dataset[training_data_len : , : ]
for i in range(5,len(test_data)):
    x_test.append(test_data[i-5:i,0])
 
# 2.  Convert the values into arrays for easier computation
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
 
# 3. Making predictions on the testing data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)
train = df[:training_data_len]
valid = df[training_data_len:]
print(valid)
valid['Predictions'] = predictions
print(valid)
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close')
 
plt.plot(train['Open'])
plt.plot(valid[['Open', 'Predictions']])
 
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
 
plt.show()
