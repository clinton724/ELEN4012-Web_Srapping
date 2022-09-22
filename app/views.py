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
    if request.method == "POST":
        name = request.form["name"]
        surname = request.form["surname"]
        email = request.form["email"]
        password = request.form["password"]
        salt = bcrypt.gensalt()
        hashedPassword = bcrypt.hashpw(password.encode(), salt)
        
        cursor.execute(f""" INSERT INTO dbo.[User] VALUES (%s, %s, %s, %s)""", 
                     (email, name, surname, hashedPassword))
        connection.commit()
        return redirect('/dashboard')
    else:
       return render_template('signup.html')

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        return redirect('/dashboard')
    else:
        return render_template('login.html')
