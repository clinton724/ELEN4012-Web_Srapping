import pymssql

def connectionPool ():
            conn = pymssql.connect(server='designdb.database.windows.net', 
                            user='designdb-admin@designdb', 
                            password='Design2022', 
                            database='RawData')
            return conn,conn.cursor()

connection, cursor = connectionPool() 