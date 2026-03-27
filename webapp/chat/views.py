from django.shortcuts import render
from .models import Conversation


def chat_view(request):
    conversations = Conversation.objects.order_by('-created_at')[:20]
    return render(request, 'chat/index.html', {'conversations': conversations})
