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
    user_exists = cursor.fetchall()
    connection.commit()
    return user_exists

def addUser(user):
    cursor.execute(f""" INSERT INTO dbo.[User] VALUES (%s, %s, %s, %s)""", 
                        (user['email'], user['name'], user['surname'], user['password']))
    connection.commit()

def userVerification(user):
    cursor.execute(f"select Password from dbo.[User] where Email='%s'"% user['email'])
    password = cursor.fetchall()
    connection.commit()
    print(password[0][0])
    hashedPassword = pbkdf2_sha256.verify(user['password'], password[0][0])
    print(hashedPassword)
    cursor.execute(f"""SELECT CASE WHEN EXISTS (
                        SELECT *
                        FROM dbo.[User]
                        WHERE Email='%s'
                    )
                    THEN CAST('True' AS VARCHAR)
                    ELSE CAST('False' AS VARCHAR) END""" % user['email'])
    user_exists = cursor.fetchall()
    connection.commit()
    if user_exists[0][0] == 'True' and hashedPassword == True:
      return True
    elif user_exists[0][0] == 'False':
      return 'The email address does not exist.'
    elif hashedPassword == False:
      return 'The password entered is incorrect'
