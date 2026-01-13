"""SQLite datetime compatibility for Python 3.12+.

This module registers datetime adapters and converters to fix the deprecation warning:
"The default datetime adapter is deprecated as of Python 3.12"

Import this module before using sqlite3 with datetime objects.
See: https://docs.python.org/3/library/sqlite3.html#default-adapters-and-converters-deprecated
"""

import sqlite3
from datetime import datetime


def _adapt_datetime(val: datetime) -> str:
    """Adapt datetime to ISO format string for SQLite storage."""
    return val.isoformat(" ")


def _convert_datetime(val: bytes) -> datetime:
    """Convert ISO format string from SQLite to datetime."""
    return datetime.fromisoformat(val.decode())


# Register the adapters and converters globally (runs once at import time)
sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_converter("timestamp", _convert_datetime)
