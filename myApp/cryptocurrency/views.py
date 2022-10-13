from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm

# Create your views here.
def home(request):
    return render(request, 'home.html', {})

def signup(request):
    form = UserCreationForm()
    context = {'form':form}
    return render(request, 'signup.html', context)

def login(request):
    return render(request, 'login.html', {})


