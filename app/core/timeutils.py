"""Timezone-aware timestamp helpers.

SQLModel stores ``datetime`` columns as UTC and rejects naive values, so every
timestamp written to the database must be produced by :func:`utcnow`.
"""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime) -> datetime:
    """Return ``value`` in UTC, assuming UTC for naive values read from legacy rows."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
