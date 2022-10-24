from django.shortcuts import render, redirect
from .forms import CreateUserForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow .keras.optimizers import Adam
from tensorflow.keras import layers
from keras import callbacks
from copy import deepcopy
from .db import connection, cursor
from .utils import get_plot, str_to_datetime, df_to_windowed_df, windowed_df_to_date_X_y

# Create your views here.
def home(request):
    return render(request, 'home.html', {})

def signup(request):
    #cursor.execute("select * from dbo.[User] FOR JSON AUTO")
    #data = cursor.fetchall()
    #print(data)
    if request.user.is_authenticated:
        return redirect('dashboard')
    else:
        if request.method == 'POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                form.save()
                username = form.cleaned_data['username']
                password = form.cleaned_data['password1']
                user = authenticate(username=username, password=password)
                login(request, user)
                return redirect('dashboard')
        else:
            form = CreateUserForm()

        return render(request, 'signup.html', {"form":form,})

def loginPage(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    else:
        if request.method == 'POST':
            username= request.POST.get('username')
            password = request.POST.get('password')
            user = authenticate(request, username = username, password = password)
            print(user)
            if user is not None:
                login(request, user)
                return redirect('dashboard')
            else:
                messages.success(request, ("There was an error"))
                return redirect('login')
        
        else: return render(request, 'login.html', {})

@login_required(login_url='login')
def dashboard(request):
    cursor.execute("select id, Cryptocurrency from dbo.[urlMapping]")
    data = cursor.fetchall()
    columns = ["id", "cryptocurrency"]
    df = pd.DataFrame(data, columns=columns)
    return render(request, 'index.html', {'context': df})

@login_required(login_url='login')
def coins(request):
    cursor.execute("select id, Cryptocurrency from dbo.[urlMapping]")
    data = cursor.fetchall()
    columns = ["id", "cryptocurrency"]
    df = pd.DataFrame(data, columns=columns)
    return render(request, 'coins.html', {'context': df})


@login_required(login_url='login')
def analytics(request, coin):
    #Getting the dataset from the database
    cursor.execute("""select Cryptocurrency from urlMapping where id = '%s'"""% (coin))
    data = cursor.fetchone()
    connection.commit()
    check = data[0]
    cursor.execute("""select * from HistoricalData where Cryptocurrency = '%s'"""% (check))
    data = cursor.fetchall()
    connection.commit()
    columns = ["cryptocurrency", "date", "market_cap", "Volume", "Open", "Close"]
    df = pd.DataFrame(data, columns=columns)

    #Extracting the closing value for analysisstart = df.iloc[3]['date']
    start = df.iloc[3]['date']
    end = df.iloc[30]['date']

    df = df[['date', 'Close']]

    #Converting all strings in the 'date' column to datetime datatype
    df['date'] = df['date'].apply(str_to_datetime)

    #Making the date to be the index of the dataset
    df.index = df.pop('date')

    #Converting the dataframe to a supervised learning problem becuase we are using the LSTM model
    windowed_df = df_to_windowed_df(df, start, end, n=3)
    #Converting the windowed dataframe into a numpy array so that we can feed into the tensorflow model
    #X is a 3 dimensional vector that consists of previous values to the target
    #y is the target
    #date is the target date
    dates, X, y = windowed_df_to_date_X_y(windowed_df)

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
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size = 1,epochs=5, callbacks=[earlystopping])
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
    graphic = [None]*7
    params = {'title': 'Actual Scraped Data',
          'xlabel': 'Date',
          'ylabel': 'Price',
          'legend': []}
    graphic[0] = get_plot({'plot-1': {
                             'x':df.index, 
                             'y':df['Close']}}, params)
    ########################################################
    x = {'plot-1': {
                    'x':dates_train, 
                    'y':y_train},
         'plot-2': {
                    'x': dates_val,
                    'y': y_val},
         'plot-3': {
                    'x': dates_test, 
                    'y': y_test}
        }
    params = {'title': 'Windowed Observation model',
          'xlabel': 'Date',
          'ylabel': 'Price',
          'legend': ['Train (80%)', 'Validation (10%)', 'Test (10%)']}
    graphic[1] = get_plot(x, params)
    ########################################################
    x = {'plot-1': {
                    'x':dates_train, 
                    'y':train_predictions},
         'plot-2': {
                    'x': dates_train,
                    'y': y_train},
         'plot-3': {
                    'x': dates_val, 
                    'y': val_predictions},
         'plot-4': {
                    'x': dates_val, 
                    'y': y_val},
         'plot-5': {
                    'x': dates_test, 
                    'y': test_predictions},
         'plot-6': {
                    'x': dates_test, 
                    'y': y_test},
         'plot-7': {
                    'x': recursive_dates, 
                    'y': recursive_predictions}
        }
    params = {'title': 'Predictions and Observations',
          'xlabel': 'Date',
          'ylabel': 'Price',
          'legend': ['Training Predictions', 
                        'Training_Observations',
                        'Validation Predictions', 
                        'Validation Observations',
                        'Testing Predictions', 
                        'Testing Observations',
                        'Recursive Predictions']}
    graphic[2] = get_plot(x, params)
    ##############################################################
    x = {'plot-1': {
                    'x':epoch_count, 
                    'y':training_loss },
         'plot-2': {
                    'x':epoch_count,
                    'y':validation_loss}
        }
    params = {'title': 'Model Loss',
          'xlabel': 'Epoch',
          'ylabel': 'Loss',
          'legend': ['Training Loss', 'Test Loss']}
    graphic[3] = get_plot(x, params)
    ################################################################
    x = {'plot-1': {
                    'x':dates_train, 
                    'y':train_predictions },
         'plot-2': {
                    'x':dates_train,
                    'y':y_train}
        }
    params = {'title': 'Training dataset (80%)',
          'xlabel': 'Price',
          'ylabel': 'Date',
          'legend': ['Training Predictions', 'Training_Observations']}
    graphic[4] = get_plot(x, params)
    #################################################################
    x = {'plot-1': {
                    'x':dates_val, 
                    'y':val_predictions },
         'plot-2': {
                    'x':dates_val,
                    'y':y_val}
        }
    params = {'title': 'Validation dataset (10%)',
          'xlabel': 'Price',
          'ylabel': 'Date',
          'legend': ['Validation Predictions', 'Validation Observations']}
    graphic[5] = get_plot(x, params)
    ##################################################################
    x = {'plot-1': {
                    'x':dates_test, 
                    'y':test_predictions },
         'plot-2': {
                    'x':dates_test,
                    'y':y_test}
        }
    params = {'title': 'Testing dataset (10%)',
          'xlabel': 'Price',
          'ylabel': 'Date',
          'legend': ['Testing Predictions', 'Testing Observations']}
    graphic[6] = get_plot(x, params)
    return render(request, 'analytics.html', {'graphic': graphic, 'coin': check+" "+'Visuals'})

@login_required(login_url='login')
def logout_user(request):
    logout(request)
    return redirect('home')


