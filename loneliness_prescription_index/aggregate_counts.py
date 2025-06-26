"""
Add data on associated disease to prescription counts, aggregate to prescriptions by illness by postcode, calculate
z-scores
"""

import numpy as np
import pandas as pd

from os import path, listdir

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_column_hist(column, savefile, n_bins=50):
    """ Convenience plotting of diagnostic histograms """
    mu = np.mean(column)
    sigma = np.std(column)
    fig, ax = plt.subplots()

    n, bins, patches = ax.hist(column.dropna(), n_bins, density=1)

    # add a 'best fit' line
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
    ax.plot(bins, y, '--')
    ax.set_xlabel('Number')
    ax.set_ylabel('Probability density')
    ax.set_title(column.name)

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.savefig(savefile, bbox_inches='tight')
    return None


def create_illness_lookup(lookup_table_file):
    """ Generate dict of illness: search regex """
    lookup_df = pd.read_csv(lookup_table_file, names=["illness", "medication"])

    lookup_df['illness'] = lookup_df['illness'].str.lower().str.strip()
    lookup_df['medication'] = lookup_df['medication'].str.lower().str.strip()

    lookup = {ill: "|".join(lookup_df[lookup_df['illness'] == ill]['medication'])
              for ill in lookup_df['illness'].unique()}

    # And if you're looking for loneliness, you want any of these prescriptions
    # lookup['loneliness'] = "|".join(lookup_df['medication'].unique())
    return lookup


def agg_file_by_illness(file_in, folder_out, illness_lookup):
    """
    Wrapper for process of aggregating prescription counts by associated illness.
    :param file_in: path to prescriptions csv file
    :param folder_out: where to dump the aggregated file
    :param illness_lookup: dict of <illness>:<regex of associated medications>
    :return: None
    """
    df = pd.read_csv(file_in,
                     usecols=['YEAR_MONTH', 'PRACTICE_CODE', 'POSTCODE', 'BNF_DESCRIPTION', 'ITEMS'])

    # Create a column of number of prescriptions for each illness
    for illness in illness_lookup.keys():
        df[illness] = df['BNF_DESCRIPTION'].str.contains(illness_lookup[illness], case=False).astype('int16') * df['ITEMS']
        print(f"Total detected of %s: %d" % (illness, sum(df[illness])))

    df.drop(['BNF_DESCRIPTION'], axis=1, inplace=True)

    # Aggregate counts of prescriptions by illness by practice code
    df['POSTCODE'] = df['POSTCODE'].str.replace(" ", "")
    df['YEAR'] = df['YEAR_MONTH'].apply(lambda x: str(x)[:4])
    df['MONTH'] = df['YEAR_MONTH'].apply(lambda x: str(x)[4:])

    df = df.groupby(['PRACTICE_CODE', 'YEAR', 'MONTH', 'POSTCODE'])[[x for x in illness_lookup.keys()] + ["ITEMS"]]\
           .sum()\
           .reset_index()

    # filter out hospitals, care homes and university campus GP's based on their objectively low mental condition
    # prescription counts
    df = df[df[[x for x in illness_lookup.keys()]].sum(axis=1) >= 500]

    # filter out addiction clinics based on their dominant counts of addiction-related prescriptions
    df = df[df[[x for x in illness_lookup.keys() if x != "addiction"]].sum(axis=1) >= df['addiction']]

    df.to_csv(path.join(folder_out, path.split(file_in)[-1]), index=False)

    return None


def calculate_zscores(files_in, folder_out, illness_lookup, nspl_file, lower_clip=0.005, upper_clip=0.995):
    """
    Wrapper for aggregating the by-practice counts for multiple files and then calculating z-scores including outlier,
    and then appending geographic information using the National Statistics Postcode Lookup (NSPL) removal.
    :param files_in:
    :param file_out:
    :param illness_lookup:
    :param lower_clip:
    :param upper_clip:
    :return: None (but writes out a nice aggregated and scored file)
    """
    # Read in files, aggregate counts from all files by practice, year and postcode
    df = pd.concat([pd.read_csv(file) for file in files_in])\
           .groupby(['PRACTICE_CODE', 'YEAR', 'POSTCODE'])[[x for x in illness_lookup.keys()] + ["ITEMS"]]\
           .sum()\
           .reset_index()

    # Calculate the zscores
    for illness in illness_lookup.keys():

        # Sort out the fun column names
        perc_col = illness + "_perc"
        trim_col = perc_col + "_trim"
        zscore_col = trim_col + "_zscore"

        # Calculate percentage of prescriptions for each illness
        df[perc_col] = 100.0 * df[illness] / df['ITEMS']

        # Floor and Roof the outliers, based on percentiles, for each condition
        percentiles = df[perc_col].quantile([lower_clip, upper_clip]).values
        df[trim_col] = np.clip(df[perc_col], percentiles[0], percentiles[1])

        # Calculate zscore for each condition
        df[zscore_col] = (df[trim_col] - df[trim_col].mean()) / df[trim_col].std(ddof=0)

    # Calculate a by-GP loneliness score
    df['loneliness_prac'] = df[[col for col in df.columns if "zscore" in col]].sum(axis=1)

    # Load the National Statistics Postcode Lookup to get geographic data
    nspl_df = pd.read_csv(nspl_file,
                          usecols=["pcd", "lat", "long", "msoa11", "laua"]) \
                .rename(columns={"pcd": "POSTCODE"})

    # Drop non-England postcodes from lookup to reduce memory use in merge
    nspl_df = nspl_df[nspl_df['laua'].str.startswith("E") == True]

    nspl_df['POSTCODE'] = nspl_df['POSTCODE'].str.replace(" ", "").str.upper()

    df.merge(nspl_df, on="POSTCODE", how="left") \
      .to_csv(path.join(folder_out, "presc_" + str(df['YEAR'][0]) + ".csv"), index=False)

    # Plot some diagnostics/QA histograms of the results
    columns = [x for x in df.columns if sum([x.startswith(illness) for illness in illness_lookup.keys()]) > 0]

    for col in columns + ['loneliness_prac']:
        print(f"Plotting %s" % col)
        plot_column_hist(df[col], path.join(folder_out, str(df['YEAR'][0]) + "_" + col + '.png'))

    return None


# Lots of folder and lookup file settings
year = 2017
# TODO configure lookup depending on whether loneliness or illness data is desired
illness_lookup = create_illness_lookup(path.join("data", "DrugslisthealthdataJG1.CSV"))
# TODO fetch data not provied from 
# https://geoportal.statistics.gov.uk/datasets/national-statistics-postcode-lookup-november-2020
nspl_file = path.join("data", "NSPL_NOV_2020_UK.csv")
working_folder = "working"
output_folder = "output"

# Get list of prescription files for the specified year
# TODO fetch data not provied from 
# https://opendata.nhsbsa.net/dataset/english-prescribing-data-epd/resource/8ae6b792-2a0c-4f4b-826c-dc6483dc32a7
epd_files = [path.join("data", "prescriptions", x)
             for x in listdir(path.join("data", "prescriptions")) if "_"+str(year) in x]

# Aggregate counts of prescriptions by practice by month

for file in epd_files:
    print(f"\nCounting prescriptions for %s" % file)
    agg_file_by_illness(file, working_folder, illness_lookup)

# Filter illnesses to loneliness-related
illness_lookup = {key: illness_lookup[key] for key in illness_lookup.keys()
                  if key in ["depression", "alzheimers", "blood pressure", "insomnia", "social anxiety"]}

# Calculate z-scores by practice over the year in question, plot distributions for QA
print("Calculating z-scores and appending geographic data")
practice_files = [path.join(working_folder, x)
             for x in listdir(path.join(working_folder)) if "_"+str(year) in x]

calculate_zscores(practice_files,
                  output_folder,
                  illness_lookup,
                  nspl_file=nspl_file,
                  lower_clip=0.005,
                  upper_clip=0.995)
