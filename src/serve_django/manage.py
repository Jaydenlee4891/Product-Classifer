#!/usr/bin/env python
"""Django's CLI entry point. Note what is NOT here: no model loading.

`AppConfig.ready()` runs for every management command, so eager loading there would make
`migrate`, `collectstatic` and `shell` each pay a 10.8-second retriever cold start and
3GB of resident memory for nothing. See classifier/apps.py.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # cascade_site, classifier
sys.path.insert(0, str(HERE.parent))   # serve, pipeline, recall — the cascade itself


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cascade_site.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
