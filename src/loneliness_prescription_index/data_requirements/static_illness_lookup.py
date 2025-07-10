"""
the DrugslisthealthdataJG1.CSV hosted file referenced on https://github.com/datasciencecampus/loneliness
is a list of illensses and medicines, each associated with loneliness
unfortunately the url is invalid when visited
"""

from typing import Iterable
import pandas as pd
from loneliness_prescription_index.config import DATA_DIR


PATH = DATA_DIR / "DrugslisthealthdataJG1.csv"
DTYPE = {"illness": pd.StringDtype, "medication": pd.StringDtype}
NAMES = list(DTYPE.keys())


def read_csv_medication_list() -> pd.DataFrame:
    return pd.read_csv(PATH, names=NAMES, dtype=DTYPE)

def read_yaml_


def create_illness_lookup(include_loneliness: bool = False) -> dict[str, str]:
    """Generate dict of illness: search regex"""

    lookup_df = read_csv_medication_list()

    def medications_for_illness(illness: str) -> Iterable[str]:
        return (
            lookup_df[lookup_df["illness"] == illness]["medication"]
            .astype(str)
            .__iter__()
        )

    illnesses: Iterable[str] = (
        lookup_df["illness"].astype(str).drop_duplicates().__iter__()
    )

    lookup = {
        illness_name: "|".join(medications_for_illness(illness_name))
        for illness_name in illnesses
    }
    if include_loneliness:
        loneliness: str = "|".join(
            lookup_df["medication"].astype(str).drop_duplicates().__iter__()
        )
        lookup["loneliness"] = loneliness
    return lookup

def create_loneliness_lookup():
    lookup_df = read_csv_medication_list()
    return "|".join(
            lookup_df["medication"].astype(str).drop_duplicates().__iter__()
        )


if __name__ == "__main__":
    print(create_illness_lookup())