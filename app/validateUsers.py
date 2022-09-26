import sys
sys.path.insert(0, '../')
from db import connection, cursor
from passlib.hash import pbkdf2_sha256

def emailValidation(email):
    cursor.execute(f"""SELECT CASE WHEN EXISTS (
                        SELECT *
                        FROM dbo.[User]
                        WHERE Email='%s' 
                    )
                    THEN CAST('True' AS VARCHAR)
                    ELSE CAST('False' AS VARCHAR) END""" % email)
    user_exists = cursor.fetchone()
    connection.commit()
    return user_exists[0]

def addUser(user):
    cursor.execute(f""" INSERT INTO dbo.[User] VALUES (%s, %s, %s, %s)""", 
                        (user['email'], user['name'], user['surname'], user['password']))
    connection.commit()

def userVerification(email):
    cursor.execute(f"""SELECT CASE WHEN EXISTS (
                        SELECT *
                        FROM dbo.[User]
                        WHERE Email='%s'
                    )
                    THEN CAST('True' AS VARCHAR)
                    ELSE CAST('False' AS VARCHAR) END""" % email)
    user_exists = cursor.fetchone()
    connection.commit()
    return user_exists[0]

def passwordVerification(password, email):
    cursor.execute(f"select Password from dbo.[User] where Email='%s'"% email)
    password_hashed = cursor.fetchone()
    connection.commit()
    hashedPassword = pbkdf2_sha256.verify(password, password_hashed[0])
    return hashedPassword
