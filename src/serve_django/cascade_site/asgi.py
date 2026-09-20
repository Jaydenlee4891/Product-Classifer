"""ASGI entry point, for async views.

Async only helps the part of this service that waits on I/O -- the synchronous LLM call in
Stage 3, which averages 1,797 ms. The PyTorch forward passes are compute-bound and will
block the event loop unless pushed to a thread, which is what views.py does via
sync_to_async(thread_sensitive=False). torch releases the GIL inside its ops, so those
threads genuinely overlap; the Python-level pre/post-processing does not.
"""
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cascade_site.settings")
application = get_asgi_application()
