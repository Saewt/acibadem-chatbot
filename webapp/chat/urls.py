from django.urls import path
from . import views, api

urlpatterns = [
    path('', views.chat_view, name='chat'),
    path('api/chat/', api.chat_endpoint, name='chat_api'),
    path('api/chat/stream/', api.chat_stream_endpoint, name='chat_stream_api'),
]
