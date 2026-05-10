# Repository Guidelines

## Project Structure & Module Organization
This repository is split into two Python services:

- `webapp/`: Django application. Main settings live in `config/`, chat features in `chat/`, scraping and ingestion code in `scraper/`, templates in `templates/`, and frontend assets in `static/`.
- `embedding_api/`: FastAPI service that serves sentence-transformer embeddings from `main.py`.
- `docker-compose.yml`: local orchestration for PostgreSQL with `pgvector`, Redis, the Django app, an optional containerized embedding API, and the scheduler.

Keep Django tests close to each app in `tests.py`. Store schema changes in the app-local `migrations/` directories.

## Build, Test, and Development Commands
- `docker compose up --build -d`: build and start the default stack on `http://localhost:8000`.
- `cd embedding_api && uvicorn main:app --host 0.0.0.0 --port 8001`: run the host-native embedding API expected by the default `.env`.
- `docker compose exec web python manage.py test chat scraper`: run Django test suites.
- `cd embedding_api && python -m pytest tests.py`: run FastAPI embedding API tests.
- `docker compose exec web python manage.py bootstrap_knowledge`: manually bootstrap the knowledge base from the mounted dataset.
- `docker compose logs -f web`: inspect runtime issues.

## Coding Style & Naming Conventions
Use Python 3.12 style with 4-space indentation and PEP 8-friendly line lengths. Follow the existing import order: standard library, third-party packages, then local modules. Use `snake_case` for functions, modules, and management commands; `PascalCase` for Django models and test classes. Keep view and service logic separated, following the current `views.py` and `services.py` pattern.

## Testing Guidelines
Use `django.test.TestCase` or `SimpleTestCase` for `webapp/` and `pytest` for `embedding_api/`. Name tests by behavior, for example `test_chat_endpoint_rejects_empty_question`. Add or update tests for every user-visible change, scraper rule change, or management command branch. There is no declared coverage gate, so keep coverage focused on changed behavior.

## Commit & Pull Request Guidelines
Recent history follows concise, Conventional Commit-style subjects such as `feat: ...` and `fix: ...`. Keep commits scoped and imperative. PRs should describe the behavior change, list test coverage, mention any dataset or `.env` implications, and include screenshots when touching `webapp/templates/` or `webapp/static/`.

## Configuration Notes
Start from `.env.example`. The default compose setup expects a read-only dataset mount at `/data/acibadem-dataset` and a host embedding API at `http://host.docker.internal:8001`.
