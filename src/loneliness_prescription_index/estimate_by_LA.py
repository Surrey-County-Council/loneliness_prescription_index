from loneliness_prescription_index.config import DATA_DIR
from loneliness_prescription_index.functions import (
    ModelCreationConfig,
    aggregate,
    interpolate,
)


cfg = ModelCreationConfig(
    boundary_file=DATA_DIR
    / "boundaries"
    / "LAD_2020"
    / "Local_Authority_Districts_(May_2020)_Boundaries_UK_BFE.shp",
    boundary_file_geography_column="lad20cd",
    gp_file_geography_column="laua",
    year=2020,
    k=10,
)

if __name__ == "__main__":
    interpolate(cfg)
    aggregate(cfg)
