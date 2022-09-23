from app import app
from flask import render_template, request, redirect
from .validateUsers import emailValidation, addUser, userVerification
import bcrypt
from passlib.hash import pbkdf2_sha256

@app.route("/")
def home():
    return render_template('welcome.html')

@app.route("/dashboard")
def dashboard():
    return render_template('index.html')

@app.route("/signup", methods=["GET", "POST"])
def signup():
    errors = {}
    user = {}
    if request.method == "POST":
        user['name'] = request.form["name"]
        user['surname'] = request.form["surname"]
        user['email'] = request.form["email"]
        password = request.form["password"]
        hashedPassword = pbkdf2_sha256.using(rounds=8000, salt_size=10).hash(password)
        user['password'] = hashedPassword
        user_exists = emailValidation(user['email'])
        if user_exists[0][0] == 'True':
            errors['email'] = ["A user with the specified email already exists."]
        else:
            addUser(user)
            return redirect('/dashboard')
    
    return render_template('signup.html', errors=errors)

@app.route("/login", methods=["GET", "POST"])
def login():
    validate = ''
    user = {}
    if request.method == "POST":
        user['email'] = request.form["email"]
        user['password'] = request.form["password"]
        user_exists = userVerification(user)
        if user_exists == True:
            return redirect('/dashboard')
        else:
            return render_template('login.html', errors=user_exists)
    
    return render_template('login.html')
