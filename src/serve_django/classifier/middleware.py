"""Opens a LangSmith tracing context per request and tags it with the Django request id.

Without this a trace has no route back to the access log line that produced it. With it,
`grep` on a request id in the log finds the trace in LangSmith and vice versa.

THE PART THAT DOES NOT WORK BY ITSELF: this context is per-process. If Stage 3 is moved to
a Celery worker -- which is the right call, since it blocks for ~1.8s -- the context does
not cross the boundary. The parent run id has to be put in the task payload and re-attached
inside the worker, or the LLM call becomes an orphan trace.

Degrades to a no-op when langsmith is not installed or tracing is off.
"""
import uuid
from contextlib import nullcontext


class TracingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        try:
            from langsmith import tracing_context
            self._ctx = tracing_context
        except Exception:
            self._ctx = None

    def __call__(self, request):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.request_id = rid
        cm = nullcontext()
        if self._ctx is not None:
            cm = self._ctx(metadata={"django_request_id": rid, "path": request.path})
        with cm:
            response = self.get_response(request)
        response["x-request-id"] = rid
        return response
