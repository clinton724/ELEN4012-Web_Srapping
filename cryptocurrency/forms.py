from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django import forms

class CreateUserForm(UserCreationForm):
     first_name = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder':"Enter First Name"}), max_length=50)
     last_name = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder':"Enter Last Name"}), max_length=50)
     email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder':"Enter email"}), max_length=50)
     
     class Meta:
          model = User
          fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')
     
     def __init__(self, *args, **kwargs):
          super(CreateUserForm, self).__init__(*args, **kwargs)
          self.fields['username'].widget.attrs['class'] = 'form-control'
          self.fields['username'].widget.attrs['placeholder'] = 'Enter username'
          self.fields['password1'].widget.attrs['class'] = 'form-control'
          self.fields['password1'].widget.attrs['placeholder'] = 'Password'
          self.fields['password2'].widget.attrs['class'] = 'form-control'
          self.fields['password2'].widget.attrs['placeholder'] = 'Password'