from loneliness_prescription_index.config import DATA_DIR
from loneliness_prescription_index.functions import (
    ModelCreationConfig,
    aggregate,
    interpolate,
)

cfg = ModelCreationConfig(
    boundary_file=DATA_DIR
    / "boundaries"
    / "MSOA_2011"
    / "Middle_Layer_Super_Output_Areas__December_2011__Boundaries_Full_Extent__BFE__EW_V3.shp",
    boundary_file_geography_column="MSOA21CD",
    gp_file_geography_column="msoa21",
    year=2020,
    k=5,
)


if __name__ == "__main__":
    interpolate(cfg)
    aggregate(cfg)
