from typing import Literal

PracticeType = Literal["practice_code"]

PracticeAreaTypes = Literal["pcn_name", "sub_icb_name", "icb_name", "nhser_name"]

PostcodeColumn = Literal["postcode"]

PostcodeAreaTypes = Literal[
    "lsoa21cd",
    "msoa21cd",
    "county",
    "lad",
    "ward",
]

LocationColumn = Literal["lat", "long"]
