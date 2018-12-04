from django.urls import path, include
from django.conf.urls import url
from . import views
from django.views.static import serve
from django.conf import settings
from django.contrib import admin
from django.conf.urls.static import static

urlpatterns = [
    path('', views.webadmin, name='webadmin'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)