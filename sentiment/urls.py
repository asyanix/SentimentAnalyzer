from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Главная страница
    path('upload_csv/', views.upload_csv, name='upload_csv'),
]
