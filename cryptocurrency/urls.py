from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('login', views.loginPage, name="login"),
    path('signup', views.signup, name="signup"),
    path('dashboard', views.dashboard, name='dashboard'),
    path('logout', views.logout_user, name='logout'),
    path('analytics/<int:coin>', views.analytics, name='analytics'),
    path('coins', views.coins, name='coins')
]
