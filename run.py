#!/usr/bin/env python3
"""Entry point for the REPL: `python3 run.py [--model ...]`."""

import sys

from ember.core import main

if __name__ == "__main__":
    sys.exit(main())
