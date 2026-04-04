import logging
import sys
import threading

from django.apps import AppConfig
from django.conf import settings

logger = logging.getLogger(__name__)
_WARMUP_STARTED = False


def _is_serving_process() -> bool:
    return 'runserver' in sys.argv or any(
        arg.endswith('gunicorn') or arg == 'gunicorn' for arg in sys.argv
    )


class ChatConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chat'

    def ready(self):
        global _WARMUP_STARTED
        if _WARMUP_STARTED or not settings.CHAT_WARMUP_ENABLED:
            return
        if not _is_serving_process():
            return

        from .services import warm_embedding_model

        def _warmup() -> None:
            try:
                warm_embedding_model()
                logger.info('chat_embedding_warmup status=success')
            except Exception as exc:  # pragma: no cover - defensive startup guard
                logger.warning('chat_embedding_warmup status=failed error=%s', exc)

        _WARMUP_STARTED = True
        threading.Thread(target=_warmup, name='chat-embedding-warmup', daemon=True).start()
