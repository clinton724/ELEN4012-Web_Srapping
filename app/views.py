from app import app
from flask import render_template, request, redirect

@app.route("/")
def home():
    return render_template('welcome.html')

@app.route("/dashboard")
def dashboard():
    return render_template('index.html')

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        user = request.form["name"]
        surname = request.form["surname"]
        email = request.form["email"]
        password = request.form["password"]
        print(user, " ", surname)
        return redirect('/')
    else:
       return render_template('signup.html')

@app.route("/login")
def login():
    return render_template('login.html')
