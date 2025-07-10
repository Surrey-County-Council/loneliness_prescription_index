"""
the national statistics postcode lookup dataset can be found on the open geography portal https://geoportal.statistics.gov.uk
this dataset is updated quarterly by ons so is likely to require manually updating

when imported this module wll warn the user if the lookup is potentially out of date

the file provides postcodes and maps to the respective area_types desired
it includes a latitude and longitude for geographical analysis

the following url may be useful to search for the latest dataset. It is likely it will be downloaded as a zip archive
https://geoportal.statistics.gov.uk/search?categories=%252Fcategories%252Fpostcode%2520products%252Fnational%2520statistics%2520postcode%2520lookup&sort=Date%20Created%7Ccreated%7Cdesc
"""

from datetime import datetime
from loguru import logger
import pandas as pd
from loneliness_prescription_index.config import DATA_DIR

LATEST_FILE = "NSPL_MAY_2025_UK.csv"
NUM_MONTHS_BEFORE_REFRESH = 3


_, MONTH, YEAR, _ = LATEST_FILE.split("_")
FILE_DATE = datetime.strptime(MONTH + YEAR, "%b%Y")
if datetime.now().month - FILE_DATE.month > NUM_MONTHS_BEFORE_REFRESH:
    logger.warning(f"file: {LATEST_FILE} may be out of date")


PATH = DATA_DIR / LATEST_FILE
DTYPE = {
    "pcd": pd.StringDtype,  # postcode
    "lat": pd.Float64Dtype,
    "long": pd.Float64Dtype,
    "msoa21": pd.StringDtype,
    "laua": pd.StringDtype,  # Local Authority / Unitary Authority Code
}
USECOLS = list(DTYPE.keys())


def read_file() -> pd.DataFrame:
    return pd.read_csv(PATH, usecols=USECOLS, dtype=DTYPE)


def get_postcode_lookup() -> pd.DataFrame:
    """returns a cleaned postcode lookup limited to the area_types hardcoded in the module"""
    df = read_file()
    df.rename(columns={"pcd": "POSTCODE"}, inplace=True)
    df["POSTCODE"] = df["POSTCODE"].str.replace(" ", "").str.upper()
    return df
