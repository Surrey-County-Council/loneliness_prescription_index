"""
https://opendata.nhsbsa.net/dataset/english-prescribing-data-epd

the api provides practice code and postcode as geographical markers
"""

from typing import Iterable
from loneliness_prescription_index.data_requirements.static_illness_lookup import (
    create_illness_lookup,
)


YEAR = 2024


def dataset_name(year: int, month: int):
    return f"EPD_{year}{month}"


def single_file_sql_query(cols_out: str, year: int, month: int):
    return f"SELECT {cols_out} FROM {dataset_name(year, month)}"


def single_illness_logic(illness: str):
    return f"SUM(CASE WHEN BNF_DESCRIPTION LIKE '%{illness}%' THEN 1 ELSE 0 END) AS PRESCRIPTIONS_FOR_{illness}"


def single_year_sql_query(: Iterable[str], year: int):
    illness_cols = ", ".join(single_illness_logic(ill) for ill in illnesses)
    cols_out = f"YEAR_MONTH, PRACTICE_CODE, POSTCODE, SUM(ITEMS) AS TOTAL_PRESCRIPTIONS, {illness_cols}"
    sub_queries = [
        single_file_sql_query(cols_out, year, month) for month in range(1, 13)
    ]
    union_query = " UNION ".join(sub_queries)
    group_query = f"{union_query} GROUP BY YEAR_MONTH, PRACTICE_CODE, POSTCODE"
    return group_query


print(single_year_sql_query(create_illness_lookup(), 2023))
