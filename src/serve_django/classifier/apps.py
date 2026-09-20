"""Where Django and FastAPI actually differ.

FastAPI's lifespan handler runs once per process and a uvicorn deployment is usually one
process. `AppConfig.ready()` also runs once per process -- but a Gunicorn/WSGI deployment
runs several, and each gets its own 3GB copy of the weights. ready() additionally runs for
EVERY management command, so an unguarded load makes `migrate` pay a 10.8-second retriever
cold start.

So eager loading here is opt-in (CASCADE_PRELOAD=1), and even then it is skipped for
management commands and for runserver's file-watching parent process. The default is lazy: the first request into each worker pays.
"""
import logging
import os
import sys

from django.apps import AppConfig
from django.conf import settings

log = logging.getLogger(__name__)

# Commands that legitimately want the models in memory. Everything else -- migrate,
# collectstatic, shell, test discovery -- must not pay for them.
SERVING_COMMANDS = {"runserver", "warm_cascade"}


def skip_reason(argv: list[str], environ) -> str | None:
    """Why this process must not preload, or None if it should. Pure, so it is testable
    without starting a server."""
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd and cmd not in SERVING_COMMANDS:
        return f"management command {cmd!r}"
    # `runserver` with the autoreloader runs ready() in TWO processes: a parent that only
    # watches files and a child (RUN_MAIN=true) that serves. Preloading in both loads the
    # weights twice and one copy is never used. `--noreload` is a single process that never
    # sets RUN_MAIN, so it must be exempt or nothing would ever load.
    if cmd == "runserver" and "--noreload" not in argv and environ.get("RUN_MAIN") != "true":
        return "autoreloader parent (the RUN_MAIN child preloads)"
    return None


class ClassifierConfig(AppConfig):
    name = "classifier"
    verbose_name = "Long-tail cascade classifier"

    def ready(self) -> None:
        if not getattr(settings, "CASCADE_PRELOAD", False):
            log.info("cascade: lazy load (set CASCADE_PRELOAD=1 to load at startup)")
            return
        reason = skip_reason(sys.argv, os.environ)
        if reason:
            log.info("cascade: skipping preload for %s", reason)
            return
        from classifier import runtime
        log.info("cascade: preloading models")
        runtime.get_runtime()
