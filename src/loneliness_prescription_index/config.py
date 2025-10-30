from pathlib import Path
from loguru import logger


PROJ_ROOT: Path = Path(__file__).resolve().parents[2]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
WORKING_DIR = PROJ_ROOT / "working"
OUTPUT_DIR = PROJ_ROOT / "output"
TEST_DIR = PROJ_ROOT / "test"
