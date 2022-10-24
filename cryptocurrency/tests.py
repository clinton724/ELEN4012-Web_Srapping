from django.test import TestCase
from .utils import str_to_datetime, df_to_windowed_df, windowed_df_to_date_X_y
from datetime import datetime
import random
import pandas as pd

# Create your tests here.
class URLTesting(TestCase):

    def test_testhomepage(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_login(self):
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
    
    def test_signup(self):
        response = self.client.get('/signup')
        self.assertEqual(response.status_code, 200)

class NeuralTesting(TestCase):

    randomlist = []
    for index in range(0,12):
            n = round(random.uniform(10.5, 75.5), 4)
            randomlist.append(n)
    
    data = {
            'date': ['2022-03-01', '2022-03-02', '2022-03-03', '2022-03-04', '2022-03-05', '2022-03-06', '2022-03-07', '2022-03-08', '2022-03-09', '2022-03-10', '2022-03-11', '2022-03-12'],
            'Close': randomlist
        }
    df = pd.DataFrame.from_dict(data)
    start = df.iloc[3]['date']
    end = df.iloc[11]['date']
    df['date'] = df['date'].apply(str_to_datetime)
    df.index = df.pop('date')
    prediction_partition = 3
    windowed_df = df_to_windowed_df(df, start, end, n=prediction_partition)
    dates, X, y = windowed_df_to_date_X_y(windowed_df)

    def test_str_to_date(self):
        _str = '2022-03-14'
        _datetime = str_to_datetime(_str)
        self.assertEqual(isinstance(_str, str), True)
        self.assertEqual(isinstance(_datetime, datetime), True)
    
    def test_windowing_function(self):
        self.assertEqual(len(self.windowed_df), len(self.df)-3)
    
    def test_indowed_df_to_date_X_y(self):
        self.assertEqual(len(self.dates.shape), 1)
        self.assertEqual(len(self.X.shape), self.prediction_partition)
        self.assertEqual(len(self.y.shape), 1)
        

        
        
        

    

