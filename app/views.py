from app import app
from flask import render_template, request, redirect, flash
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, EmailField, ValidationError
from wtforms.validators import InputRequired, Email
from .validateUsers import emailValidation, addUser, userVerification, passwordVerification
import bcrypt
from passlib.hash import pbkdf2_sha256
from flask_sqlalchemy import SQLAlchemy

app.config['SECRET_KEY'] = 'mynameis'
server = 'designdb.database.windows.net'
database = 'rawData'
app.config['SQLALCHEMY_DATABASE_URI'] = f'mssql+pymssql://designdb-admin@designdb:Design2022@{server}/{database}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
#login_manager.login_view = "login"

#Create Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    Email = db.Column(db.String(50), nullable=False, unique=True)
    FirstName = db.Column(db.String(50), nullable=False)
    Surname = db.Column(db.String(50), nullable=False)
    Password = db.Column(db.String(100), nullable=False) 

@login_manager.user_loader
def load_user(user_id):
     return User.query.get(int(user_id))

#Create a form class
class SignupForm(FlaskForm):
    name = StringField("Name", validators=[InputRequired()], render_kw={"placeholder": "Enter name"})
    surname  = StringField('Surname', validators=[InputRequired()], render_kw={"placeholder": "Enter surname"})
    email = EmailField('Email', validators=[InputRequired(), Email()], render_kw={"placeholder": "Enter email"})
    password = PasswordField('Password', validators=[InputRequired()], render_kw={"placeholder": "Enter paswword"})
    submit = SubmitField('Sign up')

class LoginForm(FlaskForm):
    email = EmailField('Email', validators=[InputRequired(), Email()], render_kw={"placeholder": "Enter email"})
    password = PasswordField('Password', validators=[InputRequired()], render_kw={"placeholder": "Enter paswword"})
    submit = SubmitField('Login')

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('welcome.html')

@app.route("/dashboard", methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('index.html')

@app.route("/signup", methods=["GET", "POST"])
def signup():
    errors = {}
    user = {}
    name = None
    surname = None
    form = SignupForm()
    if form.validate_on_submit():
        if emailValidation(form.email.data) == 'False':
            user['email'] = form.email.data
            user['name'] = form.name.data
            user['surname'] = form.surname.data
            password = form.password.data
            hashedPassword = pbkdf2_sha256.using(rounds=8000, salt_size=10).hash(password)
            user['password'] = hashedPassword
            addUser(user)
            name = form.name.data
            surname = form.surname.data
            form.email.data = ''
            form.name.data = ''
            form.surname.data = ''
            form.password.data = ''
            return redirect('/dashboard') 
        else:
            flash("The email address entered already exits.")       
    return render_template('signup.html', form=form, name=name, surname=surname)

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(Email=form.email.data).first()
        if user:
            if passwordVerification(form.password.data, form.email.data) == True:
                login_user(user)
                return redirect('/dashboard')
            else:
                flash("You have entered an incorrect password.")  
        else:
            flash("The email address entered does not exist.") 
    return render_template('login.html', form=form)

#Custom error pages

#INVALID URL
@app.errorhandler(404)
def page_not_found(error):
        return render_template('404.html'), 404

#INTERNAL SERVER ERROR
@app.errorhandler(500)
def page_not_found(error):
        return render_template('500.html'), 500


