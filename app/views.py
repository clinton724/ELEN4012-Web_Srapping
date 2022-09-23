from app import app
from flask import render_template, request, redirect
import sys
sys.path.insert(0, '../')
from db import connection, cursor
import bcrypt

@app.route("/")
def home():
    return render_template('welcome.html')

@app.route("/dashboard")
def dashboard():
    return render_template('index.html')

@app.route("/signup", methods=["GET", "POST"])
def signup():
    errors = {}
    if request.method == "POST":
        name = request.form["name"]
        surname = request.form["surname"]
        email = request.form["email"]
        password = request.form["password"]
        salt = bcrypt.gensalt()
        hashedPassword = bcrypt.hashpw(password.encode(), salt)
        cursor.execute(f"""SELECT CASE WHEN EXISTS (
                        SELECT *
                        FROM dbo.[User]
                        WHERE Email='%s' 
                    )
                    THEN CAST('True' AS VARCHAR)
                    ELSE CAST('False' AS VARCHAR) END""" % email)
        user_exists = cursor.fetchall()
        connection.commit()
        print(user_exists[0][0])
        if user_exists[0][0] == 'True':
            errors['email'] = ["A user with the specified email already exists."]
        else:
            cursor.execute(f""" INSERT INTO dbo.[User] VALUES (%s, %s, %s, %s)""", 
                        (email, name, surname, hashedPassword))
            connection.commit()
            return redirect('/dashboard')
    
    return render_template('signup.html', errors=errors)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        return redirect('/dashboard')
    else:
        return render_template('login.html')
