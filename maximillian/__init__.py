"""Maximillian — autonomous recursive software-engineering loop.

A 14-stage, graph-guided DAG with bounded cyclic recovery (test-till-pass),
dual-ledger telemetry for DSPy optimization, and air-gapped HTML reporting.
"""

__version__ = "5.2.0"

from .schemas import RoundResult  # noqa: F401
from .state_machine import Config, StateMachine  # noqa: F401
