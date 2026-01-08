"""
Debug utilities for fast debugging mainly in CLI situations.

import ez4d.debug as edb
"""

from .stack_utils import show_call_stacks as trace_up

__all__ = ['trace_up']