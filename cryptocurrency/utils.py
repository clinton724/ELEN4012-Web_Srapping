import matplotlib.pyplot as plt
from io import BytesIO
import base64
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
        
def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def get_plot(plots, params):
    plt.switch_backend('AGG')
    plt.figure(figsize=(10,5))
    for index in plots:
       plt.plot(plots[index]['x'], plots[index]['y'])
    plt.xticks(rotation=25, horizontalalignment='right')
    plt.title(params['title'])
    plt.xlabel(params['xlabel'])
    plt.ylabel(params['ylabel'])
    plt.grid()
    if len(params['legend']) > 0:
       plt.legend(params['legend'])
    plt.tight_layout()
    graph = get_graph()
    return graph