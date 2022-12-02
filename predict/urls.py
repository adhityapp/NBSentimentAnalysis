from django.urls import path

from . import views

urlpatterns = [
    path(r'', views.index),
    path(r'home/', views.index),
    path(r'predict/', views.predict),
    path(r'result/', views.result),
    path(r'about/', views.about),
]
