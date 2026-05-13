import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-change-me')
DEBUG = os.environ.get('DJANGO_DEBUG', 'True') == 'True'
ALLOWED_HOSTS = ['*']


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'pgvector.django',
    'chat',
    'scraper',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('POSTGRES_DB', 'acu_chatbot'),
        'USER': os.environ.get('POSTGRES_USER', 'acu_user'),
        'PASSWORD': os.environ.get('POSTGRES_PASSWORD', 'acu_password'),
        'HOST': os.environ.get('POSTGRES_HOST', 'db'),
        'PORT': os.environ.get('POSTGRES_PORT', '5432'),
        'CONN_MAX_AGE': 600,
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'tr'
TIME_ZONE = 'Europe/Istanbul'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': f"redis://{os.environ.get('REDIS_HOST', 'redis')}:{os.environ.get('REDIS_PORT', '6379')}",
    }
}

MODEL_RUNNER_HOST = os.environ.get(
    'MODEL_RUNNER_HOST', 'http://model-runner.docker.internal'
).rstrip('/')
MODEL_RUNNER_BASE_URL = f'{MODEL_RUNNER_HOST}/engines/v1'
LLM_BACKEND = os.environ.get('LLM_BACKEND', 'ollama').strip().lower()
LLM_DEFAULT_BASE_URL = (
    'http://host.docker.internal:11434'
    if LLM_BACKEND == 'ollama'
    else MODEL_RUNNER_BASE_URL
)
LLM_BASE_URL = os.environ.get('LLM_BASE_URL', LLM_DEFAULT_BASE_URL).rstrip('/')
LLM_MODEL = os.environ.get('LLM_MODEL', 'qwen3:8b')
LLM_THINK = _env_bool('LLM_THINK', False)
LLM_WARMUP_ENABLED = _env_bool('LLM_WARMUP_ENABLED', False)
LLM_MAX_TOKENS = int(os.environ.get('LLM_MAX_TOKENS', '1536'))
LLM_TIMEOUT = float(os.environ.get('LLM_TIMEOUT', '60'))
LLM_MAX_CONCURRENT_REQUESTS = max(
    int(os.environ.get('LLM_MAX_CONCURRENT_REQUESTS', '1')),
    1,
)
LLM_QUEUE_TIMEOUT = max(float(os.environ.get('LLM_QUEUE_TIMEOUT', '0')), 0.0)
EMBEDDING_MODEL = os.environ.get(
    'EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2'
)
EMBEDDING_BACKEND = os.environ.get('EMBEDDING_BACKEND', 'local')  # local | api
EMBEDDING_API_URL = os.environ.get(
    'EMBEDDING_API_URL', 'http://host.docker.internal:8001'
)
EMBEDDING_API_TIMEOUT = int(os.environ.get('EMBEDDING_API_TIMEOUT', '30'))
EMBEDDING_BATCH_SIZE = int(os.environ.get('EMBEDDING_BATCH_SIZE', '2'))
CACHE_TTL = int(os.environ.get('CACHE_TTL', '3600'))
RAG_RETRIEVE_LIMIT = int(os.environ.get('RAG_RETRIEVE_LIMIT', '6'))
RAG_PER_PAGE_LIMIT = int(os.environ.get('RAG_PER_PAGE_LIMIT', '3'))
RAG_MAX_CHUNK_CHARS = int(os.environ.get('RAG_MAX_CHUNK_CHARS', '800'))
RAG_MAX_CONTEXT_CHARS = int(os.environ.get('RAG_MAX_CONTEXT_CHARS', '3600'))
RAG_VECTOR_DISTANCE_STRICT = float(os.environ.get('RAG_VECTOR_DISTANCE_STRICT', '0.72'))
RAG_VECTOR_DISTANCE_BROAD = float(os.environ.get('RAG_VECTOR_DISTANCE_BROAD', '0.85'))
RAG_RRF_K = int(os.environ.get('RAG_RRF_K', '60'))
RAG_QUERY_EXPANSION_ENABLED = _env_bool('RAG_QUERY_EXPANSION_ENABLED', True)
RERANK_ENABLED = _env_bool('RERANK_ENABLED', True)
RERANK_CANDIDATE_LIMIT = int(os.environ.get('RERANK_CANDIDATE_LIMIT', '18'))
RERANK_OUTPUT_LIMIT = int(os.environ.get('RERANK_OUTPUT_LIMIT', '12'))
RERANK_MIN_SCORE = float(os.environ.get('RERANK_MIN_SCORE', '-100.0'))
RERANK_API_TIMEOUT = int(os.environ.get('RERANK_API_TIMEOUT', '30'))
SEMANTIC_TOPIC_ENABLED = _env_bool('SEMANTIC_TOPIC_ENABLED', True)
SEMANTIC_TOPIC_THRESHOLD = float(os.environ.get('SEMANTIC_TOPIC_THRESHOLD', '0.55'))
ANSWER_MODE = os.environ.get('ANSWER_MODE', 'structured_first')
CHAT_WARMUP_ENABLED = os.environ.get('CHAT_WARMUP_ENABLED', 'True') == 'True'
ACIBADEM_DATASET_ROOT = os.environ.get(
    'ACIBADEM_DATASET_ROOT',
    str(BASE_DIR.parent.parent / 'scraping'),
)
KNOWLEDGE_BOOTSTRAP_ENABLED = os.environ.get('KNOWLEDGE_BOOTSTRAP_ENABLED', 'True') == 'True'
KNOWLEDGE_BOOTSTRAP_FAIL_ON_MISSING_DATA = (
    os.environ.get('KNOWLEDGE_BOOTSTRAP_FAIL_ON_MISSING_DATA', 'True') == 'True'
)
KNOWLEDGE_SYNC_KEY = os.environ.get('KNOWLEDGE_SYNC_KEY', 'acibadem_knowledge')
KNOWLEDGE_SYNC_ENABLED = os.environ.get('KNOWLEDGE_SYNC_ENABLED', 'False') == 'True'
KNOWLEDGE_SYNC_RUN_ON_START = _env_bool('KNOWLEDGE_SYNC_RUN_ON_START', False)
KNOWLEDGE_SYNC_INTERVAL_HOURS = int(os.environ.get('KNOWLEDGE_SYNC_INTERVAL_HOURS', '168'))
KNOWLEDGE_SCHEDULER_POLL_SECONDS = int(
    os.environ.get('KNOWLEDGE_SCHEDULER_POLL_SECONDS', '300')
)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        }
    },
    'loggers': {
        'chat': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'scraper': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}
