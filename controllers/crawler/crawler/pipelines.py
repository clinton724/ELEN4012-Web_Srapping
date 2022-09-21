# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import pymssql
from dotenv import load_dotenv
import os

load_dotenv()
user_name = os.getenv("userName")
password = os.getenv("password")
serverName = os.getenv("sqlServerName")

class CrawlerPipeline(object):
    def __init__(self):
        self.create_connection()
        
    def create_connection(self):
            self.conn = pymssql.connect(server=serverName, 
                            user=user_name, 
                            password=password, 
                            database='RawData')
            self.curr = self.conn.cursor()      

    def process_item(self, item, spider):
        self.curr.execute("""insert into urlMapping values (%s,%s,%s,%s)""", 
                          ( item['name'], item['rootURL'], item['historicalData'], item['market']))
        self.conn.commit()
        return item

class ScraperPipeline(object):
    def __init__(self):
        self.create_connection()
        
    def create_connection(self):
            self.conn = pymssql.connect(server=serverName, 
                            user=user_name, 
                            password=password, 
                            database='RawData')
            self.curr = self.conn.cursor()      

    def process_item(self, item, spider):
        self.curr.execute("""insert into HistoricalData values (%s,%s,%s,%s,%s,%s)""", 
                          ( item['coin'], item['Date'], item['Market_cap'], item['Volume'], item['Open'], item['Close']))
        self.conn.commit()
        return item
    
        