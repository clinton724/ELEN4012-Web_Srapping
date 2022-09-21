from app import app
from flask import render_template

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/signup")
def signup():
    return render_template('signup.html')

@app.route("/about")
def about():
    return render_template('about.html')