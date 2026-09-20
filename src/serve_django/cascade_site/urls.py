from django.urls import path

from classifier import views

urlpatterns = [
    path("", views.root),
    path("health", views.health),
    path("classify", views.classify),
    path("classify/batch", views.classify_batch),
]
