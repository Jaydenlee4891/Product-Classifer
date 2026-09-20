"""One CascadeRuntime per process, built at most once.

This is the whole integration. `serve.graph.CascadeRuntime` is the same object the FastAPI
app uses and the same one `verify.py` measures, so the Django front end cannot drift from
the numbers in the README -- there is no second implementation to drift.
"""
import threading

from django.conf import settings

_runtime = None
_lock = threading.Lock()


def get_runtime():
    """Double-checked locking, because under a threaded server two requests can arrive
    before the first finishes loading, and loading 3GB twice is the failure mode."""
    global _runtime
    if _runtime is None:
        with _lock:
            if _runtime is None:
                from serve.graph import CascadeRuntime
                _runtime = CascadeRuntime.load(
                    settings.CASCADE_DATA,
                    device=settings.CASCADE_DEVICE,
                    provider=settings.CASCADE_PROVIDER,
                )
    return _runtime


def is_loaded() -> bool:
    """For /health: distinguishes 'up' from 'up and warm', which matters because the
    first deferred request pays 10.8s to build the retriever."""
    return _runtime is not None


def reset_for_tests(stub=None) -> None:
    global _runtime
    _runtime = stub
