from collections.abc import Iterable
from typing import Literal, Never, overload
import polars as pl
from pydantic import validate_call

from loneliness_prescription_index.data_requirements.download_epds import (
    YearOfEpd,
    grouped_dfs_for_run,
)
from loneliness_prescription_index.data_requirements.static_illness_lookup import (
    IllnessConfig,
    Medication,
)


def get_area_mapper(area_type: str) -> pl.LazyFrame: ...


def area_type_maps_to(area_type: str) -> Literal["postcode", "practice_code"]: ...


@overload
def map_practice_files(
    year: int, area_type: str, *, tag: str, illness: str
) -> Never: ...


@overload
def map_practice_files(
    year: int, area_type: str, *, tag: str, illness: None = None
): ...


@overload
def map_practice_files(
    year: int, area_type: str, *, tag: None = None, illness: str
): ...


@overload
def map_practice_files(
    year: int, area_type: str, *, tag: None = None, illness: None = None
): ...


def map_practice_files(
    year: int, area_type: str, *, tag: str | None = None, illness: str | None = None
):
    medicine_names = [
        x.name for x in IllnessConfig.load().get_medications(tag=tag, illness=illness)
    ]
    medicine_df = grouped_dfs_for_run(year, medicine_names)
    print(medicine_df)
    # join_df = get_area_mapper(area_type)

    # group_df = medicine_df.join(
    #     join_df, on=area_type_maps_to(area_type), how="left"
    # ).group_by(
    #     "PRACTICE_CODE", "YEAR", "POSTCODE", area_type
    # ).agg(
    #     pl.sum("total_items"), pl.col("grand_total")
    # )

    # group_df.collect().write_csv()


if __name__ == "__main__":
    map_practice_files(2024, "lsoa", tag="Loneliness")
