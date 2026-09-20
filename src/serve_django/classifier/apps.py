"""Where Django and FastAPI actually differ.

FastAPI's lifespan handler runs once per process and a uvicorn deployment is usually one
process. `AppConfig.ready()` also runs once per process -- but a Gunicorn/WSGI deployment
runs several, and each gets its own 3GB copy of the weights. ready() additionally runs for
EVERY management command, so an unguarded load makes `migrate` pay a 10.8-second retriever
cold start.

So eager loading here is opt-in (CASCADE_PRELOAD=1), and even then it is skipped for
management commands. The default is lazy: the first request into each worker pays.
"""
import logging
import sys

from django.apps import AppConfig
from django.conf import settings

log = logging.getLogger(__name__)

# Commands that legitimately want the models in memory. Everything else -- migrate,
# collectstatic, shell, test discovery -- must not pay for them.
SERVING_COMMANDS = {"runserver", "warm_cascade"}


class ClassifierConfig(AppConfig):
    name = "classifier"
    verbose_name = "Long-tail cascade classifier"

    def ready(self) -> None:
        if not getattr(settings, "CASCADE_PRELOAD", False):
            log.info("cascade: lazy load (set CASCADE_PRELOAD=1 to load at startup)")
            return
        argv1 = sys.argv[1] if len(sys.argv) > 1 else ""
        if argv1 and argv1 not in SERVING_COMMANDS:
            log.info("cascade: skipping preload for management command %r", argv1)
            return
        from classifier import runtime
        log.info("cascade: preloading models")
        runtime.get_runtime()
