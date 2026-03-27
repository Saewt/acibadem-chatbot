from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json


@csrf_exempt
@require_POST
def chat_endpoint(request):
    # TODO: Phase 3 — implement RAG pipeline
    try:
        body = json.loads(request.body)
        question = body.get('question', '').strip()
    except (json.JSONDecodeError, KeyError):
        return JsonResponse({'error': 'Invalid request body'}, status=400)

    if not question:
        return JsonResponse({'error': 'Question cannot be empty'}, status=400)

    return JsonResponse({
        'answer': 'Chatbot henüz hazır değil.',
        'conversation_id': None,
        'sources': [],
        'cached': False,
    })


@csrf_exempt
@require_POST
def chat_stream_endpoint(request):
    # TODO: Phase 3 — implement streaming RAG pipeline
    try:
        body = json.loads(request.body)
        question = body.get('question', '').strip()
    except (json.JSONDecodeError, KeyError):
        return JsonResponse({'error': 'Invalid request body'}, status=400)

    if not question:
        return JsonResponse({'error': 'Question cannot be empty'}, status=400)

    def stream():
        yield 'data: {"token": "Chatbot henüz hazır değil."}\n\n'
        yield 'data: {"done": true}\n\n'

    return StreamingHttpResponse(stream(), content_type='text/event-stream')
