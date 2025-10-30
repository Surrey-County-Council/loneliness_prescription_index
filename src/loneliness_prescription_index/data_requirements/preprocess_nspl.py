from pathlib import Path
from pprint import pprint
from typing import Literal, get_args
from zipfile import ZipFile
from loguru import logger
import pendulum
import polars as pl
from loneliness_prescription_index.config import DATA_DIR
from loneliness_prescription_index.data_requirements.schemas import (
    LocationColumn,
    PostcodeAreaTypes,
    PostcodeColumn,
)


# USER INPUTS!
NSPL_DATE = "AUG_2025"


# USER GUIDANCE!
NSPL_URL: str = "https://geoportal.statistics.gov.uk/search?q=PRD_NSPL&sort=Date%20Created%7Ccreated%7Cdesc"
"""Refresh instructions

download the latest zipfile from the 'NSPL_URL' to the 'NSPL_DOWNLOADS_DIRECTORY' assuming the zipfile contains the date in the filename as 
described by the 'NSPL_DATE' variable

update the 'NSPL_DATE' variable to reflect the file date

update the NSPL_SCHEMA_MAPPER to reflect the required columns
"""

# USER GENERATED CONSTANSTS!
NSPL_DOWNLOADS_DIRECTORY: Path = DATA_DIR / "postcode_lookup"

NSPL_ZIPFILE_PATH: Path = NSPL_DOWNLOADS_DIRECTORY / f"NSPL_{NSPL_DATE}.zip"

CSV_PATH_IN_ARCHIVE: str = f"Data/NSPL_{NSPL_DATE}_UK.csv"

NSPL_SCHEMA_OVERRIDES: dict[LocationColumn, type[pl.Float64]] = {
    "lat": pl.Float64,
    "long": pl.Float64,
}
"""unless specified here, all other columns are read as a string"""


class RefreshRequired(Exception):
    pass


def get_date_from_user_input() -> pendulum.Date:
    normalised_date = NSPL_DATE.title()
    return pendulum.from_format(normalised_date, "MMM_YYYY").date()


def archive_should_be_refreshed() -> bool:
    refresh_duration = pendulum.duration(months=6)
    duration_passed = (
        get_date_from_user_input() - pendulum.today().date()
    ).as_duration()
    return duration_passed > refresh_duration


def read_nspl(skip_errors=False, **kwargs) -> pl.DataFrame:
    """reads the NSPL csv specified in this script with custom defaults for schema values.

    kwargs can be passed to the pl.read_csv function for

    infer_schema=False
    schema_overrides=NSPL_SCHEMA_OVERRIDES
    """
    if "infer_schema" not in kwargs:
        kwargs["infer_schema"] = False
    if "schema_overrides" not in kwargs:
        kwargs["schema_overrides"] = NSPL_SCHEMA_OVERRIDES
    if archive_should_be_refreshed():
        if not skip_errors:
            raise RefreshRequired(
                f"data appears to be stale for NSPL. Check if download required at {NSPL_URL}, else set 'skip_errors' to True"
            )
        logger.warning(f"please download latest data from {NSPL_URL}")
    with ZipFile(NSPL_ZIPFILE_PATH) as zf:
        return pl.read_csv(zf.open(CSV_PATH_IN_ARCHIVE), **kwargs)


def print_archive_columns():
    """useful to test newly downloaded files and add the column mappings to the 'NSPL_SCHEMA_MAPPER' variable"""
    single_row = read_nspl(n_rows=1)
    pprint(single_row.columns)


ColumnNames = Literal[PostcodeAreaTypes, PostcodeColumn, LocationColumn]

COLUMN_MAPPER: dict[ColumnNames, str] = {
    "postcode": "pcds",
    "lat": "lat",
    "long": "long",
    "lsoa21cd": "lsoa21cd",
    "msoa21cd": "msoa21cd",
    "county": "cty25cd",
    "lad": "lad25cd",
    "ward": "wd25cd",
}
"""helps ensure consistent schema
PracticeAreaTypes | PracticeType reflects the acceptable user inputs
"""

# in addition to typing checks this manual check will all mappings are present
mapping_inconsistencies = COLUMN_MAPPER.keys() ^ set(get_args(ColumnNames))
if mapping_inconsistencies:
    raise TypeError(
        f"COLUMN_MAPPER contains incorrect keys. diff: {mapping_inconsistencies}"
    )


def preprocess_postcodes() -> pl.DataFrame:
    try:
        return read_nspl().select(**COLUMN_MAPPER)
    except pl.exceptions.ColumnNotFoundError as e:
        raise RefreshRequired(
            "columns in the archive no longer match the user defined mapping", e
        )


if __name__ == "__main__":
    # integration tests when script is run
    # will confirm the columns exist in the mapping and the data is up to date
    print_archive_columns()
    print(preprocess_postcodes())
