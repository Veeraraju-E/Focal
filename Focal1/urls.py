# image_tagger/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('index', views.index, name='index'),
    path('uploads/<path:filename>', views.uploaded_file, name='uploaded_file'),
    path('tag_image', views.tag_image, name='tag_image'),
    path('next_image/<str:image>', views.next_image, name='next_image'),
    path('prev_image/<str:image>', views.prev_image, name='prev_image'),
    path('explore', views.explore, name='explore'),
]
