from django.contrib import admin
from django.urls import path
from API import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('', views.home, name="home" ),
    path('upload', views.upload, name='upload'),
    path('build', views.build, name='model'),
    path('matching', views.match, name='mat'),
    path('result', views.reslt, name='reslt'),
]  + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
