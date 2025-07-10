"""
Add data on associated disease to prescription counts, aggregate to prescriptions by illness by postcode, calculate
z-scores
"""

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib

from loneliness_prescription_index.config import DATA_DIR, OUTPUT_DIR, WORKING_DIR
from loneliness_prescription_index.data_requirements.static_illness_lookup import (
    create_illness_lookup,
)


matplotlib.use("agg")
import matplotlib.pyplot as plt

EPD_PATH = DATA_DIR / "prescriptions"


def plot_column_hist(column: pd.Series, savefile: Path, n_bins=50):
    """Convenience plotting of diagnostic histograms"""
    mu = np.mean(column)
    sigma = np.std(column)
    fig, ax = plt.subplots()

    n, bins, patches = ax.hist(column.dropna(), n_bins, density=True)

    # add a 'best fit' line
    y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -0.5 * (1 / sigma * (bins - mu)) ** 2
    )
    ax.plot(bins, y, "--")
    ax.set_xlabel("Number")
    ax.set_ylabel("Probability density")
    ax.set_title(column.name)  # type: ignore

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.savefig(savefile, bbox_inches="tight")
    return None


def agg_file_by_illness(file_in: Path, illness_lookup: dict[str, str]):
    """
    reads file from 'data / prescriptions / {filename}.csv'
    create counts for each illness
    postcode for each practice code is assumed not to change
    illness is extracted from BNF_DESCRIPTION
    counts for each month will need grouping again later
    """
    df = pd.read_csv(
        file_in,
        usecols=["YEAR_MONTH", "PRACTICE_CODE", "POSTCODE", "BNF_DESCRIPTION", "ITEMS"],
    )

    # Create a column of number of prescriptions for each illness
    for illness in illness_lookup.keys():
        df[illness] = (
            df["BNF_DESCRIPTION"]
            .str.contains(illness_lookup[illness], case=False)
            .astype("int16")
            * df["ITEMS"]
        )
        print(f"Total detected of {illness}: {sum(df[illness])}")

    df.drop(["BNF_DESCRIPTION"], axis=1, inplace=True)

    # Aggregate counts of prescriptions by illness by practice code
    df["POSTCODE"] = df["POSTCODE"].str.replace(" ", "")
    df["YEAR"] = df["YEAR_MONTH"].apply(lambda x: str(x)[:4])
    df["MONTH"] = df["YEAR_MONTH"].apply(lambda x: str(x)[4:])

    df = (
        df.groupby(["PRACTICE_CODE", "YEAR", "MONTH", "POSTCODE"])[
            [x for x in illness_lookup.keys()] + ["ITEMS"]
        ]
        .sum()
        .reset_index()
    )

    # filter out hospitals, care homes and university campus GP's based on their objectively low mental condition
    # prescription counts
    df = df[df[[x for x in illness_lookup.keys()]].sum(axis=1) >= 500]

    # filter out addiction clinics based on their dominant counts of addiction-related prescriptions
    df = df[
        df[[x for x in illness_lookup.keys() if x != "addiction"]].sum(axis=1)
        >= df["addiction"]
    ]

    df.to_csv(WORKING_DIR / file_in.name, index=False)

    return None


def get_nspl_df():
    # Load the National Statistics Postcode Lookup to get geographic data
    nspl_file = DATA_DIR / "NSPL_NOV_2020_UK.csv"
    nspl_df = pd.read_csv(
        nspl_file, usecols=["pcd", "lat", "long", "msoa11", "laua"]
    ).rename(columns={"pcd": "POSTCODE"})

    # Drop non-England postcodes from lookup to reduce memory use in merge
    nspl_df = nspl_df[nspl_df["laua"].str.startswith("E")]

    nspl_df["POSTCODE"] = nspl_df["POSTCODE"].str.replace(" ", "").str.upper()

    return nspl_df


def create_practice_files(year: int, illness_lookup: dict[str, str]):
    for file in EPD_PATH.glob(f"*_{year}*.csv"):
        print(f"Counting prescriptions for {file}")
        agg_file_by_illness(file, illness_lookup)


def combine_practice_files(year: int, illness_lookup: dict[str, str]):
    """
    following the groupby performed for each month, a groupby is performed for the whole year
    """
    df = (
        pd.concat([pd.read_csv(file) for file in WORKING_DIR.glob(f"*_{year}*.csv")])
        .groupby(["PRACTICE_CODE", "YEAR", "POSTCODE"])[
            [x for x in illness_lookup.keys()] + ["ITEMS"]
        ]
        .sum()
        .reset_index()
    )
    return df


def calculate_zscores(
    prescription_counts_for_illness: pd.DataFrame,
    illness_lookup: dict[str, str],
    nspl_df: pd.DataFrame,
    lower_clip=0.005,
    upper_clip=0.995,
):
    # Calculate the zscores
    for illness in illness_lookup.keys():
        # Sort out the fun column names
        perc_col = illness + "_perc"
        trim_col = perc_col + "_trim"
        zscore_col = trim_col + "_zscore"

        # Calculate percentage of prescriptions for each illness
        prescription_counts_for_illness[perc_col] = (
            100.0
            * prescription_counts_for_illness[illness]
            / prescription_counts_for_illness["ITEMS"]
        )

        # Floor and Roof the outliers, based on percentiles, for each condition
        percentiles = (
            prescription_counts_for_illness[perc_col]
            .quantile([lower_clip, upper_clip])
            .values
        )
        prescription_counts_for_illness[trim_col] = np.clip(
            prescription_counts_for_illness[perc_col], percentiles[0], percentiles[1]
        )

        # Calculate zscore for each condition
        prescription_counts_for_illness[zscore_col] = (
            prescription_counts_for_illness[trim_col]
            - prescription_counts_for_illness[trim_col].mean()
        ) / prescription_counts_for_illness[trim_col].std(ddof=0)

    # Calculate a by-GP loneliness score
    prescription_counts_for_illness["loneliness_prac"] = (
        prescription_counts_for_illness[
            [col for col in prescription_counts_for_illness.columns if "zscore" in col]
        ].sum(axis=1)
    )

    prescription_counts_for_illness.merge(nspl_df, on="POSTCODE", how="left").to_csv(
        OUTPUT_DIR / f"presc_{prescription_counts_for_illness['YEAR'][0]}.csv",
        index=False,
    )

    # Plot some diagnostics/QA histograms of the results
    columns = [
        x
        for x in prescription_counts_for_illness.columns
        if sum([x.startswith(illness) for illness in illness_lookup.keys()]) > 0
    ]

    for col in columns + ["loneliness_prac"]:
        print(f"Plotting {col}")
        plot_column_hist(
            prescription_counts_for_illness[col],
            OUTPUT_DIR
            / f"presc_{prescription_counts_for_illness['YEAR'][0]}_{col}.csv",
        )

    return None


def main():
    # Lots of folder and lookup file settings
    year = 2017
    illness_lookup = create_illness_lookup()
    nspl_df = get_nspl_df()
    create_practice_files(year, illness_lookup)

    # Filter illnesses to loneliness-related
    loneliness_lookup = {
        key: illness_lookup[key]
        for key in illness_lookup.keys()
        if key
        in ["depression", "alzheimers", "blood pressure", "insomnia", "social anxiety"]
    }

    combined_practice_data = combine_practice_files(year, loneliness_lookup)

    # Calculate z-scores by practice over the year in question, plot distributions for QA
    print("Calculating z-scores and appending geographic data")

    calculate_zscores(
        combined_practice_data,
        loneliness_lookup,
        nspl_df=nspl_df,
        lower_clip=0.005,
        upper_clip=0.995,
    )
