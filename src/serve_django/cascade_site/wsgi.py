"""WSGI entry point. This is the one that multiplies memory.

    gunicorn --chdir src/serve_django cascade_site.wsgi:application --workers 6

Six workers is six copies of ~3GB of weights. Do not reach for --preload to fix that:
loading before fork looks like sharing, but CUDA/MPS contexts do not survive fork() and
copy-on-write stops helping the moment refcounts touch the pages. If you need one copy,
you need one process -- see README.md.
"""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cascade_site.settings")
application = get_wsgi_application()
