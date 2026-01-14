"""
Debug utilities for fast debugging mainly in CLI situations.

import ez4d.debug as edb
"""

from tqdm import tqdm
from ipdb import set_trace
bp = set_trace  # shortcut for set_trace

from rich import inspect


from .stack_utils import show_call_stacks as trace_up

__all__ = ['set_trace', 'bp', 'inspect', 'trace_up']