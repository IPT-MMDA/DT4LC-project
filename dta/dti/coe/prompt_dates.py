"""Dates a prompt names ("NDVI for 1 June", "between 1 and 9 June", "2023-06-01"), matched to attachments.

A single-image analysis on several files reads the one whose ``metadata.date`` the prompt names; a
change request that names two files' dates compares them, earlier as before. Month names are a fixed
English table, not the locale's. Numeric dates are ISO (``2023-06-01``) or month-day (``06-01``).
"""

from __future__ import annotations

import datetime
import re

from dta.dti.schemas import Attachment

# (year, month, day); a ``None`` part matches any value ("on the 1st" has neither year nor month).
DateMention = tuple[int | None, int | None, int]

_MONTHS = {
    name: i + 1
    for i, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sept", "sep"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ]
    )
    for name in names
}
_MON = r"(?P<mon>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b\.?"
_DAY = r"(?P<d1>\d{1,2})(?:st|nd|rd|th)?"
_DAY2 = r"(?:\s*(?:and|to|or|until|through|&|-|–|—)\s*(?P<d2>\d{1,2})(?:st|nd|rd|th)?)?"
_YEAR = r"(?:,?\s+(?P<year>\d{4})\b)?"

_PATTERNS = [
    re.compile(r"\b(?P<year>\d{4})-(?P<m>\d{1,2})-(?P<d1>\d{1,2})\b"),  # 2023-06-01
    re.compile(rf"\b{_DAY}{_DAY2}\s+(?:of\s+)?{_MON}{_YEAR}", re.IGNORECASE),  # 1 June, 1st of June, 1 and 9 June
    re.compile(rf"\b{_MON}\s+{_DAY}\b{_DAY2}{_YEAR}", re.IGNORECASE),  # June 1, June 1st, June 1 and 9
    re.compile(r"(?<![\d.-])(?P<m>\d{2})-(?P<d1>\d{2})(?![\d-])"),  # 06-01 (month-day, two digits each)
    re.compile(r"\bon\s+the\s+(?P<d1>\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE),  # on the 1st
]


def _valid(year: int | None, month: int | None, day: int) -> bool:
    try:
        datetime.date(year or 2000, month or 1, day)  # 2000: a leap year, so 29 February is valid
    except ValueError:
        return False
    return True


def prompt_dates(prompt: str) -> list[DateMention]:
    """Every date ``prompt`` names, in pattern order; a range "1 and 9 June" gives both days.

    Each pattern blanks what it matched, so "1-9 June" is two days of June, not month 1 day 9.
    """
    found: list[DateMention] = []
    text = prompt
    for pattern in _PATTERNS:
        for m in pattern.finditer(text):
            parts = m.groupdict()
            year = int(parts["year"]) if parts.get("year") else None
            month = _MONTHS[parts["mon"].lower()] if parts.get("mon") else int(parts["m"]) if parts.get("m") else None
            for day in (parts["d1"], parts.get("d2")):
                if day and _valid(year, month, int(day)):
                    found.append((year, month, int(day)))
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    return found


def _named(mention: DateMention, iso: str) -> bool:
    try:
        day = datetime.date.fromisoformat(iso)
    except ValueError:
        return False
    year, month, d = mention
    return d == day.day and month in (None, day.month) and year in (None, day.year)


def attachments_named_by_date(prompt: str, attachments: list[Attachment]) -> list[Attachment]:
    """The attachments whose ``metadata.date`` the prompt names, each once, earliest first."""
    mentions = prompt_dates(prompt)
    named = [
        a for a in attachments if a.metadata and a.metadata.date and any(_named(m, a.metadata.date) for m in mentions)
    ]
    return sorted(named, key=lambda a: a.metadata.date if a.metadata and a.metadata.date else "")
