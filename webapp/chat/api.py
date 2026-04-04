import json

from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .services import ConversationNotFoundError, chat, chat_stream


@csrf_exempt
@require_POST
def chat_endpoint(request):
    try:
        body = json.loads(request.body)
        question = body.get('question', '').strip()
        conversation_id = body.get('conversation_id')
    except (json.JSONDecodeError, KeyError):
        return JsonResponse({'error': 'Invalid request body'}, status=400)

    if not question:
        return JsonResponse({'error': 'Question cannot be empty'}, status=400)

    if conversation_id in {'', None}:
        conversation_id = None
    elif not isinstance(conversation_id, int):
        return JsonResponse({'error': 'conversation_id must be an integer'}, status=400)

    try:
        payload = chat(question=question, conversation_id=conversation_id)
    except ConversationNotFoundError:
        return JsonResponse({'error': 'Conversation not found'}, status=404)
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=502)

    return JsonResponse(payload)


@csrf_exempt
@require_POST
def chat_stream_endpoint(request):
    try:
        body = json.loads(request.body)
        question = body.get('question', '').strip()
        conversation_id = body.get('conversation_id')
    except (json.JSONDecodeError, KeyError):
        return JsonResponse({'error': 'Invalid request body'}, status=400)

    if not question:
        return JsonResponse({'error': 'Question cannot be empty'}, status=400)

    if conversation_id in {'', None}:
        conversation_id = None
    elif not isinstance(conversation_id, int):
        return JsonResponse({'error': 'conversation_id must be an integer'}, status=400)

    try:
        stream = chat_stream(question=question, conversation_id=conversation_id)
    except ConversationNotFoundError:
        return JsonResponse({'error': 'Conversation not found'}, status=404)
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=502)

    response = StreamingHttpResponse(stream, content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response
