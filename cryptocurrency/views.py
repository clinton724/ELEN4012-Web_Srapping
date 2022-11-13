from django.shortcuts import render, redirect
from .forms import CreateUserForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from .db import connection, cursor
from sklearn import metrics
from .utils import get_plot


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

    #--------------------------------PLOTS----------------------------------
    #Plot of Cryptocurrency price vs time
    ########################################################
    x = {'plot-1': {
                    'x':df['Open']},
         'plot-2': {
                    'x': valid_tree[['Open', 'Predictions']]}
        }
    params = {'title': 'Price Prediction',
          'xlabel': 'Days',
          'ylabel': 'Price (USD)',
          'legend': ['Actual', 'Validation', 'Prediction']}
    graphic = get_plot(x, params)
    
    return render(request, 'analytics.html', {'graphic': graphic, 'coin': check+" "+'Visuals'})

@login_required(login_url='login')
def logout_user(request):
    logout(request)
    return redirect('home')


