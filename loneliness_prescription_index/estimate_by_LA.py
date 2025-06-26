"""
Uses inverse distance weighting to estimate loneliness prescription stats at different geographic levels.
Note that for calculations here the CRS 'epsg:27700' is used, because distances are in m which supports the custom
IDW functions.
"""

# Modin has CPU-parallelised versions of many pandas functions it'll use in preference
# If parallel alternative not available, defaults to native pandas
# The caveat is that geopandas uses vanilla pandas df's under the hood
# so if you want to merge a modin df to a geopandas df you first have to
# convert the modin df with _to_pandas() private method.
# Geopandas hidden dependencies:  mapclassify, descartes

from os import path
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt

# This suppresses an erroneous geopandas warning triggered by a plotting function
# See https://github.com/geopandas/geopandas/issues/1565 for associated bug report
import warnings
warnings.filterwarnings("ignore", "The GeoSeries you are attempting", UserWarning)


def distance_matrix(x0, y0, x1, y1):
    """ Helper for calculating distance matrix fast using numpy. """
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    # Make a distance matrix between pairwise observations
    # Note: from <http://stackoverflow.com/questions/1871536>
    # (Yay for ufuncs!)
    d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
    d1 = np.subtract.outer(obs[:, 1], interp[:, 1])

    return np.hypot(d0, d1)


def simple_lisa(map, property_column, threshold=1.0):
    """
    Local Indicator of Spatial Autocorrelation (LISA)
    Indicates whether a geographic area is similar to its neighbours.
    :param map: geoDataFrame with geometries
    :param property_column: associated value of interest
    :param threshold: threshold at which to binarise value of interest
    :return: list of LISA scores, list of support for each score
    """
    # Recode property of interest as binary based on threshold
    map['high_val'] = map[property_column] >= threshold

    scores = []
    n_neighbours = []
    for index, row in map.iterrows():

        # Find neighbouring property_column values
        neighbour_values = maps[~maps.geometry.disjoint(row.geometry)]['high_val']

        # Score is fraction with identical value to area of interest
        scores.append(sum(neighbour_values == row['high_val']) / len(neighbour_values))
        n_neighbours.append(len(neighbour_values))

    return scores, n_neighbours


def knn_idw(known_points, known_values, unknown_points, k=None, distance_limit=None):
    """
    Taken from https://stackoverflow.com/questions/3104781/inverse-distance-weighted-idw-interpolation-with-python
    Make sure your coordinate system is symmetric (ie; it's using distance not degrees) or funky things will happen.
    Implements simple Inverse-Distance Weighting (IDW) for a series of Shapely geometry points, in whatever Coordinate
    Reference System (CRS) the data is provided in.
    :param known_points: geoSeries of Shapely point geometries at which the value to be interpolated is known.
    :param known_values: Series of some numeric (float) value to be interpolated, corresponding to known_points.
    :param unknown_points: geoSeries of Shapely point geometries at which to interpolate.
    :param k: k Nearest Neighbors to apply IDW with.
    :param distance_limit: Maximum allowable distance for a known point to contribute to IDW of an unknown point.
    :return: Interpolated values (array of float).
    """
    # Convert geoSeries of Shapely Points into arrays of x and y coordinates
    xi, yi = zip(*[(point.x, point.y) for point in unknown_points]); xi = np.array(xi); yi = np.array(yi)
    x, y = zip(*[(point.x, point.y) for point in known_points]); x = np.array(x); y = np.array(y)
    z = np.array(known_values)

    dist = distance_matrix(x, y, xi, yi)

    # Remove influence of any point further than the distance of k'th nearest
    if k:
        dist_mask = np.array([np.sort(dist, axis=0)[k-1], ] * dist.shape[0])

        dist = np.where(dist <= dist_mask, dist, np.inf)

    # Remove influence of any point further than distance_limit
    if distance_limit:
        dist = np.where(dist <= distance_limit, dist, np.inf)

    # In IDW, weights are 1 / distance
    weights = 1.0 / dist

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    return np.dot(weights.T, z)


def plot_attribute(data, column, cmap="OrRd"):
    """ Helper for plotting base maps """
    f, ax1 = plt.subplots(1, 1, figsize=(15, 10))

    data.plot(column=column,
              ax=ax1,
              cmap=cmap,
              edgecolor="grey",
              scheme='quantiles',
              legend=True,
              missing_kwds={"color": "lightgrey",
                            "edgecolor": None,
                            "hatch": None,
                            "label": "Missing values"})

    return f, ax1


# Lots of folder and lookup file settings
geography_column = 'lad20cd'
# TODO source the correct shp file
boundary_file = path.join(
    "data", "boundaries", "LAD_2020", "Local_Authority_Districts_(May_2020)_Boundaries_UK_BFE.shp"
)
# TODO use the correct output file
prescriptions_file = path.join("output", "presc_2020.csv")
year_string = prescriptions_file.split("_")[1].strip(".csv")
output_folder = "output"

""" -------------------------------
    Interpolate at LA using IDW
------------------------------- """
print("Loading maps and data")
maps = gpd.read_file(boundary_file)\
          .to_crs({'init': 'epsg:27700'})

# drop the non-England LA's
maps = maps[maps[geography_column].str[:1] == 'E'].copy()

# Load GP data
gp_df = pd.read_csv(prescriptions_file).rename(columns={"laua": geography_column})
gp_gdf = gpd.GeoDataFrame(gp_df,
                          crs="epsg:4326",
                          geometry=gpd.points_from_xy(gp_df.long.astype(float),
                                                      gp_df.lat.astype(float))).to_crs(epsg=27700)

# Filter out those with bad geometry data (practices with no postcode/address)
gp_gdf = gp_gdf[gp_gdf.is_valid]

print("Interpolating loneliness, calculating LISA")
k = 10
maps['loneliness'] = knn_idw(gp_gdf['geometry'], gp_gdf['loneliness_prac'], maps.centroid, k=k)
maps['LISA_score'], maps['LISA_support'] = simple_lisa(maps, "loneliness", threshold=0.0)

print("Plotting data")
f, _ = plot_attribute(maps, "loneliness", cmap="Blues")
f.savefig(path.join(output_folder, "LA_{}_loneliness_interpolation_k{}.png".format(year_string, k)))

f, _ = plot_attribute(maps, "LISA_score", cmap="Reds")
f.savefig(path.join(output_folder, "LA_{}_loneliness_LISA_k{}.png".format(year_string, k)))

# Save final data as Local Authority code loneliness lookup file
maps[[geography_column, 'loneliness', 'LISA_score', 'LISA_support']]\
    .sort_values("loneliness", ascending=False)\
    .to_csv(path.join(output_folder, "LA_{}_loneliness_stats_k{}.csv".format(year_string, k)), index=False)


""" -------------------------------
    Alternative: aggregate to LA
------------------------------- """
print("Loading maps and data")
maps = gpd.read_file(boundary_file)\
          .to_crs({'init': 'epsg:27700'})

# drop the non-England LA's
maps = maps[maps[geography_column].str[:1] == 'E'].copy()

# Load GP data
gp_df = pd.read_csv(prescriptions_file).rename(columns={"laua": geography_column})

# Calculate LA-level loneliness stat as mean of all GP loneliness scores within LA
print("\nAggregating loneliness stat as mean of all GP's within area")
score_cols = [col for col in gp_df.columns if "score" in col]
la_df = gp_df.groupby(geography_column)[score_cols].mean().reset_index()
la_df['loneliness'] = la_df[score_cols].sum(axis=1)
maps = maps.merge(la_df, on=geography_column, how="left")

# Calculate support (number of prescriptions) within each LA
la_support_df = gp_df.groupby(geography_column).sum("ITEMS").reset_index()[[geography_column, 'ITEMS']]
maps = maps.merge(la_support_df, on=geography_column, how="left")

# Save results
print("Plotting and saving results")
maps[[geography_column, 'loneliness', 'ITEMS']]\
    .sort_values("loneliness", ascending=False)\
    .to_csv(path.join(output_folder, "LA_{}_loneliness_agg.csv".format(year_string)), index=False)

# Plot results
f, ax1 = plot_attribute(maps, "loneliness", cmap="Blues")
f.savefig(path.join(output_folder, "LA_{}_loneliness_agg.png".format(year_string)))
