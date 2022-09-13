import sys
sys.path.insert(0, '../')
from db import connection, cursor

 
#cursor.execute("INSERT INTO RawData (ServerTimestamp, PositionDirection, PositionLatitude,PositionLongitude,PositionSpeed) VALUES ('2', '23', '50', '45', '23');")

cursor.execute("select * from RawData FOR JSON AUTO")
data = cursor.fetchall()
connection.commit()
print(data)