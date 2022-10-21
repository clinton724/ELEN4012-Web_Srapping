import sys
sys.path.insert(0, '../')
from db import connection, cursor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow .keras.optimizers import Adam
from tensorflow.keras import layers
from keras import callbacks
from copy import deepcopy

#Function for converting the string to datetime
def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return dt.datetime(year=year, month=month, day=day)

#Windowing function
def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
  first_date = str_to_datetime(first_date_str)
  last_date  = str_to_datetime(last_date_str)

  target_date = first_date
  
  dates = []
  X, Y = [], []
  
  last_time = False
  while True:
    df_subset = dataframe.loc[:target_date].tail(n+1)
    
    if len(df_subset) != n+1:
      print(f'Error: Window of size {n} is too large for date {target_date}')
      return

    values = df_subset['Close'].to_numpy()
    x, y = values[:-1], values[-1]

    dates.append(target_date)
    X.append(x)
    Y.append(y)

    next_week = dataframe.loc[target_date:target_date+dt.timedelta(days=7)]
    next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
    next_date_str = next_datetime_str.split('T')[0]
    year_month_day = next_date_str.split('-')
    year, month, day = year_month_day
    next_date = dt.datetime(day=int(day), month=int(month), year=int(year))
    
    if last_time:
      break
    
    target_date = next_date

    if target_date == last_date:
      last_time = True
    
  ret_df = pd.DataFrame({})
  ret_df['Target Date'] = dates
  
  X = np.array(X)
  for i in range(0, n):
    X[:, i]
    ret_df[f'Target-{n-i}'] = X[:, i]
  
  ret_df['Target'] = Y

  return ret_df

def windowed_df_to_date_X_y(windowed_dataframe):
    #Conversion of dataframe to numpy array
    df_as_np = windowed_dataframe.to_numpy()
    #Getting the dates from the windowed dataframe
    dates = df_as_np[:,0]
    #Getting the middle matrix for x
    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
    Y = df_as_np[:, -1]
    return dates, X.astype(np.float32), Y.astype(np.float32)


#Getting the dataset from the database
cursor.execute("select * from HistoricalData where Cryptocurrency = 'Bitcoin (BTC)'")
data = cursor.fetchall()
connection.commit()
columns = ["cryptocurrency", "date", "market_cap", "Volume", "Open", "Close"]
df = pd.DataFrame(data, columns=columns)

#Extracting the closing value for analysisstart = df.iloc[3]['date']
start = df.iloc[3]['date']
end = df.iloc[30]['date']

future_days = 1
df = df[['date', 'Close']]

#Converting all strings in the 'date' column to datetime datatype
df['date'] = df['date'].apply(str_to_datetime)

#Making the date to be the index of the dataset
df.index = df.pop('date')

#Converting the dataframe to a supervised learning problem becuase we are using the LSTM model
windowed_df = df_to_windowed_df(df, start, end, n=3)
print(windowed_df)
#Converting the windowed dataframe into a numpy array so that we can feed into the tensorflow model
#X is a 3 dimensional vector that consists of previous values to the target
#y is the target
#date is the target date
dates, X, y = windowed_df_to_date_X_y(windowed_df)
print(dates.shape, X.shape, y.shape)

#Split the data into train, validation and testing partitions
#The training will train the model, the validation wil help train the model 
# and the testing will be used to evaluate the performance of the model
#80% is for training
#80%-90% (10%) is for validation
#90%-100% (10%) is for testing

q_80 = int(len(dates)*.8)
q_90 = int(len(dates)*.9)
dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

#Creating and training the model
#Creating a sequential model
model = Sequential([layers.Input((3, 1)), 
                  layers.LSTM(64),
                  layers.Dense(32, activation='relu'), 
                  layers.Dense(32, activation='relu'),
                  layers.Dense(1)])

#Optimize the loss function
model.compile(loss='mse', 
             optimizer=Adam(learning_rate=0.001),
             metrics=['mean_absolute_error'])
earlystopping = callbacks.EarlyStopping(monitor="val_loss", 
                           mode="min", patience=5,
                           restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size = 1,epochs=100, callbacks=[earlystopping])
training_loss  = history.history['loss']
validation_loss = history.history['val_loss']
epoch_count = range(1, len(training_loss)+1)
train_predictions = model.predict(X_train).flatten()
test_predictions = model.predict(X_test).flatten()

val_predictions = model.predict(X_val).flatten()
recursive_predictions = []
recursive_dates = np.concatenate([dates_val, dates_test])
for target_date in recursive_dates:
    last_window = deepcopy(X_train[-1])
    next_prediction = model.predict(np.array([last_window])).flatten()
    recursive_predictions.append(next_prediction)
    new_window = list(last_window[1:])
    new_window.append(next_prediction)
    new_window = np.array(new_window)
    last_window = new_window

#--------------------------------PLOTS----------------------------------

#Plot of Cryptocurrency price vs time
plt.figure('Figure 1')
plt.plot(df.index, df['Close'])
plt.xlabel("Date")
plt.ylabel("Crypto price")
plt.grid()
plt.xticks(rotation=30, horizontalalignment='right')
####

#
plt.figure('Figure 2')
plt.plot(dates_train, y_train)
plt.plot(dates_val, y_val)
plt.plot(dates_test, y_test)
plt.legend(['Train', 'Validation', 'Test'])
plt.xlabel("Date")
plt.ylabel("Crypto price")
plt.grid()
plt.xticks(rotation=25, horizontalalignment='right')
#####

#
plt.figure('Figure 3')
plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val)
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.plot(recursive_dates, recursive_predictions)
plt.legend(['Training Predictions', 
            'Training_Observations',
            'Validation Predictions', 
            'Validation Observations',
            'Testing Predictions', 
            'Testing Observations',
            'Recursive Predictions'])
plt.xlabel("Date")
plt.ylabel("Crypto price")
plt.grid()
plt.xticks(rotation=25, horizontalalignment='right')
#####

#
plt.figure('Figure 4')
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, validation_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
#####

#
plt.figure('Figure 5')
plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.title('Training predictions and Training Observations')
plt.legend(['Training Predictions', 'Training_Observations'])
plt.xlabel("Date")
plt.ylabel("Crypto price")
plt.grid()
plt.xticks(rotation=25, horizontalalignment='right')
#####

#
plt.figure('Figure 6')
val_predictions = model.predict(X_val).flatten()
plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val)
plt.title('Validation predictions and Validation Observations')
plt.legend(['Validation Predictions', 'Validation Observations'])
plt.xlabel("Date")
plt.ylabel("Crypto price")
plt.grid()
plt.xticks(rotation=25, horizontalalignment='right')
#####

#
plt.figure('Figure 7')
test_predictions = model.predict(X_test).flatten()
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.title('Testing predictions and Testing Observations')
plt.legend(['Testing Predictions', 'Testing Observations'])
plt.xlabel("Date")
plt.ylabel("Crypto price")
plt.grid()
plt.xticks(rotation=25, horizontalalignment='right')
plt.show()


