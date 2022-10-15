import pymssql
from dotenv import load_dotenv
import os

load_dotenv()
user_name = os.getenv("userName")
password = os.getenv("password")
serverName = os.getenv("sqlServerName")

def connectionPool ():
            conn = pymssql.connect(server=serverName, 
                            user=user_name, 
                            password=password, 
                            database='RawData')
            return conn,conn.cursor()

connection, cursor = connectionPool() 