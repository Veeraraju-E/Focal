# image_tagger/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('uploads/<path:filename>', views.uploaded_file, name='uploaded_file'),
    path('tag_image', views.tag_image, name='tag_image'),
    path('next_image/<str:image>', views.next_image, name='next_image'),
    path('prev_image/<str:image>', views.prev_image, name='prev_image'),
    path('find_tags', views.find_tags, name='find_tags'),
    path('update_tags', views.update_tags, name='update_tags'),
]
