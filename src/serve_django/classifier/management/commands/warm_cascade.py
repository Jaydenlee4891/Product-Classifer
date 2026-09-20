"""Load the models and report what it cost.

    python src/serve_django/manage.py warm_cascade

Useful as a container readiness step: run it before the process starts taking traffic and
the first real request does not pay the load. Named in apps.SERVING_COMMANDS so preloading
is allowed for it.
"""
import time

from django.core.management.base import BaseCommand

from classifier import runtime


class Command(BaseCommand):
    help = "Load the cascade into this process and report timings."

    def add_arguments(self, parser):
        parser.add_argument("--defer-one", action="store_true",
                            help="also classify a synthetic item likely to defer, which "
                                 "forces the lazy retriever to build (~10.8s)")

    def handle(self, *args, **opts):
        t0 = time.perf_counter()
        rt = runtime.get_runtime()
        self.stdout.write(f"S1 + cross-encoder loaded in {time.perf_counter() - t0:.2f}s "
                          f"on {rt.cascade.device}")
        if opts["defer_one"]:
            t1 = time.perf_counter()
            out = rt.classify({"title": "obscure replacement cartridge, 2-pack"})
            self.stdout.write(f"first classify: {time.perf_counter() - t1:.2f}s "
                              f"(stage {out['stage']}, retriever now "
                              f"{'warm' if rt.cascade.retriever is not None else 'cold'})")
