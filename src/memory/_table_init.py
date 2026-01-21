"""Table initialization tracking to avoid redundant CREATE TABLE calls."""

# Module-level set tracking which tables have been ensured
# This persists for the lifetime of the process
_initialized_tables: set[str] = set()


def is_table_initialized(table_name: str) -> bool:
    """Check if a table has already been initialized in this process."""
    return table_name in _initialized_tables


def mark_table_initialized(table_name: str) -> None:
    """Mark a table as initialized."""
    _initialized_tables.add(table_name)


def reset_initialized_tables() -> None:
    """Reset initialized tables tracking (for testing only)."""
    _initialized_tables.clear()
