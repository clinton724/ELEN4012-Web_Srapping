from django.shortcuts import render, redirect
from .forms import CreateUserForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .db import connection, cursor

# Create your views here.
def home(request):
    return render(request, 'home.html', {})

def signup(request):
    cursor.execute("select * from dbo.[User] FOR JSON AUTO")
    data = cursor.fetchall()
    print(data)
    if request.user.is_authenticated:
        return redirect('dashboard')
    else:
        if request.method == 'POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                form.save()
                username = form.cleaned_data['username']
                password = form.cleaned_data['password1']
                user = authenticate(username=username, password=password)
                login(request, user)
                return redirect('dashboard')
        else:
            form = CreateUserForm()

        return render(request, 'signup.html', {"form":form,})

def loginPage(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    else:
        if request.method == 'POST':
            username= request.POST.get('username')
            password = request.POST.get('password')
            user = authenticate(request, username = username, password = password)
            print(user)
            if user is not None:
                login(request, user)
                return redirect('dashboard')
            else:
                messages.success(request, ("There was an error"))
                return redirect('login')
        
        else: return render(request, 'login.html', {})

@login_required(login_url='login')
def dashboard(request):
    return render(request, 'index.html', {})

@login_required(login_url='login')
def logout_user(request):
    logout(request)
    return redirect('home')


