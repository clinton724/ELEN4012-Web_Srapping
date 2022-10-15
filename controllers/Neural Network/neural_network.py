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

print(X.shape)


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

#history = model.fit(X_train,Y_train, epochs=50, validation_data=(X_val,Y_val))


#import tensorflow as tf
 
#print(tf.__version__)
#print(tf.reduce_sum(tf.random.normal([1000,1000])))





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