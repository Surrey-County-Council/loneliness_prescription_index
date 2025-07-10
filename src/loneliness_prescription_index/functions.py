from dataclasses import dataclass
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd

from loneliness_prescription_index.config import OUTPUT_DIR


def plot_attribute(data: pd.DataFrame, column, cmap="OrRd"):
    """Helper for plotting base maps"""
    f, ax1 = plt.subplots(1, 1, figsize=(15, 10))

    data.plot(
        column=column,
        ax=ax1,
        cmap=cmap,
        edgecolor="grey",
        scheme="quantiles",
        legend=True,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": None,
            "hatch": None,
            "label": "Missing values",
        },
    )

    return f, ax1


def distance_matrix(x0, y0, x1, y1):
    """Helper for calculating distance matrix fast using numpy."""
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    # Make a distance matrix between pairwise observations
    # Note: from <http://stackoverflow.com/questions/1871536>
    # (Yay for ufuncs!)
    d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
    d1 = np.subtract.outer(obs[:, 1], interp[:, 1])

    return np.hypot(d0, d1)


def simple_lisa(maps, property_column, threshold=1.0):
    """
    Local Indicator of Spatial Autocorrelation (LISA)
    Indicates whether a geographic area is similar to its neighbours.
    :param map: geoDataFrame with geometries
    :param property_column: associated value of interest
    :param threshold: threshold at which to binarise value of interest
    :return: list of LISA scores, list of support for each score
    """
    # Recode property of interest as binary based on threshold
    maps["high_val"] = maps[property_column] >= threshold

    scores = []
    n_neighbours = []
    for index, row in maps.iterrows():
        # Find neighbouring property_column values
        neighbour_values = maps[~maps.geometry.disjoint(row.geometry)]["high_val"]

        # Score is fraction with identical value to area of interest
        scores.append(sum(neighbour_values == row["high_val"]) / len(neighbour_values))
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
    xi, yi = zip(*[(point.x, point.y) for point in unknown_points])
    xi = np.array(xi)
    yi = np.array(yi)
    x, y = zip(*[(point.x, point.y) for point in known_points])
    x = np.array(x)
    y = np.array(y)
    z = np.array(known_values)

    dist = distance_matrix(x, y, xi, yi)

    # Remove influence of any point further than the distance of k'th nearest
    if k:
        dist_mask = np.array(
            [
                np.sort(dist, axis=0)[k - 1],
            ]
            * dist.shape[0]
        )

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


@dataclass
class ModelCreationConfig:
    boundary_file: Path
    boundary_file_geography_column: str
    gp_file_geography_column: str
    year: int
    k: int


def get_maps(cfg: ModelCreationConfig):
    print("Loading maps and data")
    maps = gpd.read_file(cfg.boundary_file).to_crs({"init": "epsg:27700"})
    # drop the non-England LA's
    maps = maps[maps[cfg.boundary_file_geography_column].str[:1] == "E"].copy()
    return maps


def get_gp_df(cfg: ModelCreationConfig):
    prescriptions_file = OUTPUT_DIR / f"presc_{cfg.year}.csv"
    gp_df = pd.read_csv(prescriptions_file).rename(
        columns={cfg.gp_file_geography_column: cfg.boundary_file_geography_column}
    )
    return gp_df


def get_gp_gdf(cfg: ModelCreationConfig):
    gp_df = get_gp_df(cfg)
    gp_gdf = gpd.GeoDataFrame(
        gp_df,
        crs="epsg:4326",
        geometry=gpd.points_from_xy(gp_df.long.astype(float), gp_df.lat.astype(float)),
    ).to_crs(epsg=27700)

    # Filter out those with bad geometry data (practices with no postcode/address)
    gp_gdf = gp_gdf[gp_gdf.is_valid]
    return gp_gdf


def interpolate(cfg: ModelCreationConfig):
    """-------------------------------
        Interpolate at `GEOGRAPHY TYPE` using IDW
    -------------------------------"""
    maps = get_maps(cfg)
    gp_gdf = get_gp_gdf(cfg)

    print("Interpolating loneliness, calculating LISA")
    maps["loneliness"] = knn_idw(
        gp_gdf["geometry"], gp_gdf["loneliness_prac"], maps.centroid, k=cfg.k
    )
    maps["LISA_score"], maps["LISA_support"] = simple_lisa(
        maps, "loneliness", threshold=0.0
    )

    print("Plotting data")
    f, _ = plot_attribute(maps, "loneliness", cmap="Blues")
    f.savefig(
        OUTPUT_DIR
        / f"{cfg.boundary_file_geography_column}_{cfg.year}_loneliness_interpolation_k{cfg.k}.png"
    )

    f, _ = plot_attribute(maps, "LISA_score", cmap="Reds")
    f.savefig(
        OUTPUT_DIR
        / f"{cfg.boundary_file_geography_column}_{cfg.year}_loneliness_LISA_k{cfg.k}.png"
    )

    # Save final data as Local Authority code loneliness lookup file
    maps[
        [cfg.boundary_file_geography_column, "loneliness", "LISA_score", "LISA_support"]
    ].sort_values("loneliness", ascending=False).to_csv(
        OUTPUT_DIR
        / f"{cfg.boundary_file_geography_column}_{cfg.year}_loneliness_stats_k{cfg.k}.csv",
        index=False,
    )


def aggregate(cfg: ModelCreationConfig):
    """-------------------------------
        Alternative: aggregate to `GEOGRAPHY TYPE`
    -------------------------------"""
    print("Loading maps and data")

    maps = get_maps(cfg)
    gp_df = get_gp_df(cfg)

    # Calculate LA-level loneliness stat as mean of all GP loneliness scores within LA
    print("\nAggregating loneliness stat as mean of all GP's within area")
    score_cols = [col for col in gp_df.columns if "score" in col]
    area_type_df = (
        gp_df.groupby(cfg.boundary_file_geography_column)[score_cols]
        .mean()
        .reset_index()
    )
    area_type_df["loneliness"] = area_type_df[score_cols].sum(axis=1)
    maps = maps.merge(area_type_df, on=cfg.boundary_file_geography_column, how="left")

    # Calculate support (number of prescriptions) within the area type
    support_df = (
        gp_df.groupby(cfg.boundary_file_geography_column)["ITEMS"].sum().reset_index()
    )
    maps = maps.merge(support_df, on=cfg.boundary_file_geography_column, how="left")

    # Save results
    print("Plotting and saving results")
    maps[[cfg.boundary_file_geography_column, "loneliness", "ITEMS"]].sort_values(
        "loneliness", ascending=False
    ).to_csv(
        OUTPUT_DIR
        / f"{cfg.boundary_file_geography_column}_{cfg.year}_loneliness_agg.csv",
        index=False,
    )

    # Plot results
    f, _ = plot_attribute(maps, "loneliness", cmap="Blues")
    f.savefig(
        OUTPUT_DIR
        / f"{cfg.boundary_file_geography_column}_{cfg.year}_loneliness_agg.png"
    )
