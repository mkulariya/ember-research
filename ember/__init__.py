"""ember: a deep research agent on a small open-weights model.

The agent core is a single module (`ember.core`) kept deliberately whole --
one readable loop, stdlib plus `openai`. Use `bootstrap()` for programmatic
runs and `main()` for the REPL.
"""

from ember.core import Agent, Config, bootstrap, main

__version__ = "0.1.0"
__all__ = ["Agent", "Config", "bootstrap", "main"]
