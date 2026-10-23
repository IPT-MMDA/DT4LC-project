"""The dates a prompt names, and the attachments they pick (v0.6.0 run 2: "NDVI for 1 June")."""

import locale

import pytest

from dta.dti.coe.prompt_dates import attachments_named_by_date, prompt_dates
from dta.dti.schemas import Attachment, RasterMetadata


@pytest.mark.parametrize(
    ("prompt", "dates"),
    [
        ("Calculate NDVI for 1 June", [(None, 6, 1)]),
        ("Calculate NDVI for June 1", [(None, 6, 1)]),
        ("NDVI for the 1st of June", [(None, 6, 1)]),
        ("NDVI for June 1st, 2023", [(2023, 6, 1)]),
        ("NDVI for 1 Jun", [(None, 6, 1)]),
        ("NDVI for 1 JUNE 2023", [(2023, 6, 1)]),
        ("NDVI for 2023-06-01", [(2023, 6, 1)]),
        ("NDVI for 06-01", [(None, 6, 1)]),
        ("NDVI on the 1st", [(None, None, 1)]),
        ("Compare water between 1 and 9 June", [(None, 6, 1), (None, 6, 9)]),
        ("Compare water 1-9 June", [(None, 6, 1), (None, 6, 9)]),
        ("Compare 1 June and 9 June", [(None, 6, 1), (None, 6, 9)]),
        ("Compare June 1 to June 9", [(None, 6, 1), (None, 6, 9)]),
        ("Snow on 3 Sept and 14 Feb", [(None, 9, 3), (None, 2, 14)]),
    ],
)
def test_prompt_dates(prompt: str, dates: list[tuple[int | None, int | None, int]]) -> None:
    assert prompt_dates(prompt) == dates


@pytest.mark.parametrize(
    "prompt",
    [
        "Calculate NDVI",
        "Track water weekly through June",
        "Redo the water map with a 0.1 threshold",
        "Use bands B5-B6",
        "Map NDVI for 31 June",  # not a date
        "You may 2x the threshold",
        "Fields within 0.25-0.50 of the mean",
    ],
)
def test_prompt_without_a_date(prompt: str) -> None:
    assert prompt_dates(prompt) == []


def test_month_names_do_not_depend_on_the_locale() -> None:
    """The month table is fixed English: a non-English LC_TIME changes nothing."""
    before = locale.setlocale(locale.LC_TIME)
    for name in ("de_DE.UTF-8", "fr_FR.UTF-8", "uk_UA.UTF-8", "C"):
        try:
            locale.setlocale(locale.LC_TIME, name)
        except locale.Error:
            continue
        try:
            assert prompt_dates("NDVI for 1 June") == [(None, 6, 1)]
            assert prompt_dates("NDVI für 1 Juni") == []  # only English names
        finally:
            locale.setlocale(locale.LC_TIME, before)


def _dated(*dates: str | None) -> list[Attachment]:
    return [
        Attachment(
            id=f"a{i}",
            filename=f"f{i}.tif",
            mime_type="image/tiff",
            path=f"/x/f{i}.tif",
            metadata=RasterMetadata(date=d) if d else None,
        )
        for i, d in enumerate(dates)
    ]


def test_attachments_named_by_date_are_earliest_first() -> None:
    files = _dated("2023-06-09", "2023-06-01", None)

    assert [a.id for a in attachments_named_by_date("between 9 and 1 June", files)] == ["a1", "a0"]
    assert [a.id for a in attachments_named_by_date("NDVI for 1 June 2023", files)] == ["a1"]


@pytest.mark.parametrize("prompt", ["NDVI for 5 June", "NDVI for 1 June 2022", "NDVI for 1 July", "Calculate NDVI"])
def test_no_attachment_matches(prompt: str) -> None:
    assert attachments_named_by_date(prompt, _dated("2023-06-09", "2023-06-01", None)) == []
