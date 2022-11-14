import sys
sys.path.insert(0, '../')
from db import connection, cursor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

cursor.execute("select * from HistoricalData where Cryptocurrency = 'Bitcoin (BTC)'")
data = cursor.fetchall()
connection.commit()
columns = ["cryptocurrency", "date", "market_cap", "Volume", "Open", "Close"]
df = pd.DataFrame(data, columns=columns)

df = df[['Open']]
future_days = 5

#This will contain our target data, this will try to predict values 5 days in future
df['Prediction'] = df[['Open']].shift(-future_days)
#Create the feature data set (X) and then convert it to a numpy array, then remove the last 'x' rows/days
X = np.array(df.drop(['Prediction'], 1))[:-future_days]
#Create the target data set (y), then convert it to a numpy array 
y = np.array(df['Prediction'])[:-future_days]
#split data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#Create model
#Decision tree regressor model
tree = DecisionTreeRegressor().fit(x_train, y_train)
#linear regression
lr = LinearRegression().fit(x_train, y_train)
#Get the last 'x'/5 days rows of the feature data set
x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)

x_future = np.array(x_future)
print(x_future)
print()
print(x_train)
#Show the model tree prediction
tree_prediction = tree.predict(x_future)
linear_prediction = lr.predict(x_future)
#Visualizing the data
predictions_tree = tree_prediction
valid_tree = df[X.shape[0]:]
valid_tree['Predictions'] = predictions_tree

predictions_linear = linear_prediction
valid_linear = df[X.shape[0]:]
valid_linear['Predictions'] = predictions_linear
print('MSE: ', metrics.mean_squared_error(valid_tree['Open'], valid_tree['Predictions']))
print('MAE: ', metrics.mean_absolute_error(valid_tree['Open'], valid_tree['Predictions']))
#########################plot###############
plt.figure('Figure 1')
plt.title('Prices (USD)', fontsize=14, fontweight='bold')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.plot(df['Open'], linewidth=2)
plt.grid()

plt.figure('Figure 2')
plt.title('DecisionTree Model (Bitcoin)', fontsize=20, fontweight='bold')
plt.xlabel('Days', fontsize=18, fontweight='bold')
plt.ylabel('Price (USD)', fontsize=18, fontweight='bold')
plt.plot(df['Open'], linewidth=3)
plt.plot(valid_tree[['Open', 'Predictions']], linewidth=3)
plt.legend(['Actual', 'Validation', 'Prediction'], fontsize=18)
plt.grid()

plt.figure('Figure 3')
plt.title('Linear Model')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.plot(df['Open'])
plt.plot(valid_linear[['Open', 'Predictions']])
plt.legend(['Actual', 'Validation', 'Prediction'])
plt.grid()
plt.show()

