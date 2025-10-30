from typing import Literal, get_args
from loguru import logger
import pendulum
import polars as pl

from loneliness_prescription_index.config import DATA_DIR
from loneliness_prescription_index.data_requirements.schemas import (
    PracticeAreaTypes,
    PracticeType,
)


# USER INPUTS!

ODS_DOWNLOAD_DATE = "2025-10-17"
"""Change this variable when you download new data. Furthed guidance below"""

FILE_PATH = DATA_DIR / "gp_lookup" / "ODS Advanced Search (Include headers).csv"

# USER GUIDANCE!
BASE_URL = (
    "https://www.odsdatasearchandexport.nhs.uk/?search=advorg&primaryroles=177"
    f"&lastChangeDateEnd={pendulum.today().to_date_string()}"
    "&columns=111010100000000010000001000001000000000100001000000000000000000000101010000"
)
"""Refresh instructions

update the 'ODS_DOWNLOAD_DATE' variable to reflect the date of download

go to the webpage generating the search for said date (this should contain all prescribing cost centers and include mapping columns required)

the open_webpage function can help!

go to 'Export' and 'download data as csv with headers

double check the schema is as expected
"""


def open_webpage():
    import webbrowser

    webbrowser.open(BASE_URL)


class RefreshRequired(Exception):
    pass


def get_date_from_user_input() -> pendulum.Date:
    return pendulum.Date.fromisoformat(ODS_DOWNLOAD_DATE)


def data_should_be_refreshed() -> bool:
    refresh_duration = pendulum.duration(months=6)
    duration_passed = (
        get_date_from_user_input() - pendulum.today().date()
    ).as_duration()
    return duration_passed > refresh_duration


def read_gp_codes() -> pl.DataFrame:
    if data_should_be_refreshed() or not FILE_PATH.exists():
        open_webpage()
        raise RefreshRequired(f"Download the latest data to {FILE_PATH}")
    return pl.read_csv(FILE_PATH)


ColumnNames = Literal[PracticeAreaTypes, PracticeType]

COLUMN_MAPPER: dict[ColumnNames, str] = {
    "practice_code": "Code",
    "pcn_name": "Is Partner To - Name",
    "sub_icb_name": "Geographic Primary Care Organisation Name",
    "icb_name": "High Level Health Geography Name",
    "nhser_name": "National Grouping Name",
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


def preprocess_gp_codes() -> pl.DataFrame:
    logger.info(f"using gp codes downloaded by user on {ODS_DOWNLOAD_DATE}")
    return read_gp_codes().select(**COLUMN_MAPPER)


if __name__ == "__main__":
    # integration tests when script is run
    # will confirm the columns exist in the mapping and the data is up to date
    print(preprocess_gp_codes())
