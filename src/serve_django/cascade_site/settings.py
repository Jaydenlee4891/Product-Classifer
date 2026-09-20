"""Minimal Django settings for a stateless inference service.

No database, no sessions, no templates, no staticfiles. Every one of those is absent on
purpose: this service holds 3GB of frozen tensors and answers JSON, and each installed app
is another thing that runs in `ready()` for every worker and every management command.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Dev default only. A real deployment supplies this and DEBUG stays off.
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-not-a-secret-do-not-ship")
DEBUG = os.environ.get("DJANGO_DEBUG", "") == "1"
ALLOWED_HOSTS = [h for h in os.environ.get(
    "DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",") if h]

INSTALLED_APPS = ["classifier.apps.ClassifierConfig"]

# CommonMiddleware only. No sessions, no CSRF: this is a token-authenticated JSON API in
# any real deployment, and CSRF protects cookie-authenticated form posts, which these are
# not. Saying that out loud matters more than the two lines it saves -- "I turned off CSRF"
# is a red flag unless you can say why it does not apply.
MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
    "classifier.middleware.TracingMiddleware",
]

ROOT_URLCONF = "cascade_site.urls"
WSGI_APPLICATION = "cascade_site.wsgi.application"
ASGI_APPLICATION = "cascade_site.asgi.application"

DATABASES = {}          # the ORM is never touched
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": os.environ.get("DJANGO_LOG_LEVEL", "INFO")},
}

# --- this service's own configuration, read by classifier/runtime.py -------------------
CASCADE_DATA = os.environ.get("CASCADE_DATA", str(BASE_DIR.parent.parent / "data"))
CASCADE_DEVICE = os.environ.get("CASCADE_DEVICE", "auto")
CASCADE_PROVIDER = os.environ.get("CASCADE_PROVIDER", "cached")
# Load the models in AppConfig.ready() instead of on first request. Correct for a single
# long-lived process; wrong for a multi-worker pool, and wrong for management commands.
CASCADE_PRELOAD = os.environ.get("CASCADE_PRELOAD", "") == "1"
