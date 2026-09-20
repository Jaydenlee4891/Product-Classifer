# The same cascade, served from Django

`src/serve/app.py` serves this system with FastAPI. This is the Django equivalent, and it
exists because the difference between them is not style — it is **process model**, and the
process model is what decides whether a 3GB model server is affordable.

Both front ends call the identical object: `serve.graph.CascadeRuntime`. No model code is
duplicated here, so whichever one you run, the numbers in the README describe it.

## Run it

```
pip install "django>=5.0"
DJANGO_SETTINGS_MODULE=cascade_site.settings \
  python src/serve_django/manage.py runserver 8001
```

```
curl -s localhost:8001/health | python3 -m json.tool
curl -s localhost:8001/classify -H 'content-type: application/json' \
     -d '{"title":"AmazonBasics Modern Euro Toilet Paper Holder"}'
```

## The thing worth knowing

FastAPI's lifespan handler loads the models **once per process**, and a uvicorn deployment
is typically one process. Django's equivalent hook, `AppConfig.ready()`, also runs once per
process — but a conventional Gunicorn/WSGI deployment runs *several* worker processes, and
each one gets its own copy.

Sized with this project's own numbers: mean end-to-end service time is 409 ms, so 10 req/s
is ~4.1 workers busy on average, and you want headroom, so call it 6 sync workers. Resident
set is roughly 3GB (DistilBERT 265MB, cross-encoder 90MB, Qwen3-Embedding-0.6B 2.4GB fp32,
plus the label embedding matrix). **That is ~18GB of weights for one service, six copies of
the same frozen tensors.**

Two traps that follow:

- `ready()` also fires for `manage.py migrate`, `collectstatic` and every other management
  command. An unguarded model load there makes each of them pay the 10.8s retriever cold
  start. Hence `CASCADE_PRELOAD` below.
- Gunicorn's `--preload` loads the app before forking, which looks like the fix. It is not
  a safe one: CUDA and MPS contexts do not survive `fork()` reliably, and copy-on-write
  stops helping as soon as Python refcounts touch the pages.

And one the latency table makes obvious: p50 is 25 ms, p95 is 2,182 ms, because 82% of
requests stop at S1 and 18% make a synchronous LLM call averaging 1,797 ms. Under sync WSGI
workers each of those occupies a whole worker for two seconds, so a small pool is
head-of-line blocked by its own slow tail.

## What this implementation does about it

It loads **lazily by default** and eagerly only when `CASCADE_PRELOAD=1`. Lazy is the right
default for a worker pool: the first request into each worker pays the load, management
commands pay nothing, and you opt into eager loading for a single-process deployment where
you want the cost at boot instead of in a user's request.

What it deliberately does **not** do is pretend Django is the right home for the models. In
a real deployment the answer is to put the service boundary at the model: Django owns HTTP,
auth, persistence and the admin; the weights live in one process — the FastAPI service
called over localhost, or a Celery worker pool — because many cheap workers and one
expensive one are opposite requirements. Given 82% of requests finish in 25 ms, you would
keep S1 in the request path and offload only the escalated tier. **The τ gate built for cost
turns out to be the same line you would draw for deployment.**

## LangSmith across a process boundary

Tracing context is per-request: `classifier/middleware.py` opens one and threads the Django
request id in as metadata, so a trace maps back to an access log line. The gotcha is Celery
— trace context does *not* cross a process boundary on its own. The parent run id has to
travel in the task payload and be re-attached inside the worker, or the LLM call shows up
as an orphan trace with no route back to the request that caused it.

## DRF

Not used. These are three JSON endpoints with no models, no auth and no browsable API, so
serializers would be ceremony. A production version inside a Django shop would almost
certainly use DRF for content negotiation, throttling and schema generation — and the
validation here is written to be replaced by a serializer, not to argue against one.
