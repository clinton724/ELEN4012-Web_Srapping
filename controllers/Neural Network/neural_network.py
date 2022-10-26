import imp
import sys
sys.path.insert(0,'../../')
from db import connection, cursor
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers

def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)

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

    values = df_subset['close_price'].to_numpy()
    x, y = values[:-1], values[-1]

    dates.append(target_date)
    X.append(x)
    Y.append(y)

    next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
    next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
    next_date_str = next_datetime_str.split('T')[0]
    year_month_day = next_date_str.split('-')
    year, month, day = year_month_day
    next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
    
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
  df_as_np = windowed_dataframe.to_numpy()

  dates = df_as_np[:, 0]

  middle_matrix = df_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)


cursor.execute("select * from HistoricalData where Cryptocurrency = 'Prom (PROM)'")
data = cursor.fetchall()
connection.commit()
columns = ["cryptocurrency", "date", "market_cap", "volume", "open_price", "close_price"]
df = pd.DataFrame(data, columns=columns)
start = df.iloc[3]['date']
length = len(df['date'])
end = df.iloc[length-1]['date']

df['date'] = df['date'].apply(str_to_datetime)
df = df[['date','close_price']]
df.index = df.pop('date')
#lt.plot(df.index, df['close_price'])
#plt.show()


windowed_df = df_to_windowed_df(df, start, end, n=3)


dates, X, y = windowed_df_to_date_X_y(windowed_df)

#print( dates.shape, X.shape, y.shape)

q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

plt.plot(dates_train, y_train)
plt.plot(dates_val, y_val)
plt.plot(dates_test, y_test)

plt.legend(['Train', 'Validation', 'Test'])

model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100,)










'''
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import numpy as np
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

y = np.array(y)
X = np.array(X)


X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size= 0.4)
X_val, X_test, Y_val, Y_test = train_test_split(X_test,Y_test, test_size= 0.5)

model = Sequential()
#model.add(LSTM(10, input_shape=(10,1), activation='relu', return_sequences=False))
#model.add(Dense(256,activation='relu'))
model.add(Dense(12, input_shape=(10,), activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss = 'mean_squared_error', optimizer= 'adam', metrics=['accuracy'])
model.summary()

history = model.fit(X,y, epochs=50, batch_size=10)


#import tensorflow as tf
 
#print(tf.__version__)
#print(tf.reduce_sum(tf.random.normal([1000,1000])))






from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import numpy as np
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

y = np.array(y)
X = np.array(X)


#column_one = [row[0] for row in X]
#plt.scatter(column_one,y)
#plt.show()
#plt.plot(column_one,y)
#plt.show()


X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size= 0.4)
X_val, X_test, Y_val, Y_test = train_test_split(X_test,Y_test, test_size= 0.5)

model = Sequential()
model.add(LSTM(128, input_shape=(10,1), activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='softmax'))
model.compile(loss = 'mean_absolute_error', optimizer= 'adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train,Y_train, epochs=50, validation_data=(X_val,Y_val))
'''