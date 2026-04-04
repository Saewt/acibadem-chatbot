import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-change-me')
DEBUG = os.environ.get('DJANGO_DEBUG', 'True') == 'True'
ALLOWED_HOSTS = ['*']

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
LLM_MODEL = os.environ.get('LLM_MODEL', 'ai/qwen3:4B-UD-Q4_K_XL')
EMBEDDING_MODEL = os.environ.get(
    'EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2'
)
CACHE_TTL = int(os.environ.get('CACHE_TTL', '3600'))
RAG_RETRIEVE_LIMIT = int(os.environ.get('RAG_RETRIEVE_LIMIT', '6'))
RAG_PER_PAGE_LIMIT = int(os.environ.get('RAG_PER_PAGE_LIMIT', '3'))
RAG_MAX_CHUNK_CHARS = int(os.environ.get('RAG_MAX_CHUNK_CHARS', '700'))
RAG_MAX_CONTEXT_CHARS = int(os.environ.get('RAG_MAX_CONTEXT_CHARS', '4000'))
CHAT_WARMUP_ENABLED = os.environ.get('CHAT_WARMUP_ENABLED', 'True') == 'True'

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
