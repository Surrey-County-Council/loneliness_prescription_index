from datetime import date, datetime
from pathlib import Path
from typing import (
    Annotated,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
)
from zipfile import ZipFile
from annotated_types import Ge, Le, Len
from loguru import logger
import pendulum
from pydantic import AfterValidator, BaseModel, HttpUrl, RootModel, validate_call
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
import polars as pl

from loneliness_prescription_index.config import DATA_DIR


# validation types and functional validators
TwoCharacterString = Annotated[str, Len(max_length=2, min_length=2)]
FourCharacterString = Annotated[str, Len(max_length=4, min_length=4)]
SixCharacterString = Annotated[str, Len(max_length=6, min_length=6)]
TenCharacterString = Annotated[str, Len(max_length=10, min_length=10)]
YearAfter2014 = Annotated[int, Ge(2014)]
Month = Annotated[int, Ge(1), Le(12)]


@validate_call(validate_return=True)
def get_date_suffix(
    value: SixCharacterString | TenCharacterString,
) -> SixCharacterString:
    return value[-6:]


@validate_call(validate_return=True)
def get_year(value: SixCharacterString | TenCharacterString) -> FourCharacterString:
    return get_date_suffix(value)[:-2]


@validate_call(validate_return=True)
def get_year_int(value: SixCharacterString | TenCharacterString) -> YearAfter2014:
    return int(get_year(value))


@validate_call(validate_return=True)
def get_month(value: SixCharacterString | TenCharacterString) -> TwoCharacterString:
    return get_date_suffix(value)[-2:]


@validate_call(validate_return=True)
def get_month_int(value: SixCharacterString | TenCharacterString) -> Month:
    return int(get_month(value))


@validate_call(validate_return=True)
def get_dataset_name_prefix(dataset_name: TenCharacterString) -> Literal["EPD_"]:
    match dataset_name[:4]:
        case "EPD_":
            return "EPD_"
        case prefix:
            raise ValueError(
                f"dataset_name {dataset_name} must start with 'EPD_', got {prefix}"
            )


def validate_year_month(value: SixCharacterString) -> SixCharacterString:
    get_year_int(value)
    get_month_int(value)
    return value


def validate_dataset_name(dataset_name: TenCharacterString) -> TenCharacterString:
    get_dataset_name_prefix(dataset_name)
    validate_year_month(get_date_suffix(dataset_name))
    return dataset_name


EpdYearMonth = Annotated[SixCharacterString, AfterValidator(validate_year_month)]
EpdDatasetName = Annotated[TenCharacterString, AfterValidator(validate_dataset_name)]


@validate_call
def get_date(val: EpdYearMonth | EpdDatasetName) -> date:
    return pendulum.from_format(get_date_suffix(val), "YYYYMM").date()


@validate_call
def create_dataset_name(year: YearAfter2014, month: Month):
    return f"EPD_{year:0>4}{month:0>2}"


def initialised_dir(p: Path):
    if p.is_file():
        raise ValueError(f"path {p} is a file and does not require initialisation")
    if not p.exists():
        logger.trace(f"creating directory at {p}")
        p.mkdir(parents=True)
    return p


class DownloadLogRecord(BaseModel):
    root_dir: ClassVar[Path] = DATA_DIR / "raw_epds"
    record_type: Literal["raw_epds"] = "raw_epds"
    name: EpdDatasetName
    url: HttpUrl  # the url to read the csv from
    zip_url: HttpUrl  # the url to read the zip from
    latest_download_date: datetime | None = None
    update_required: bool = True

    @property
    def archive_name(self):
        return f"{self.name}.zip"

    @property
    def archive(self) -> Path:
        return self.__class__.root_dir / self.archive_name

    def is_valid_to_use(self) -> bool:
        return (
            self.latest_download_date is not None
            and not self.update_required
            and self.archive.exists()
        )


class ItemsByPrescription(BaseModel):
    name: Literal["items_by_prescription"] = "items_by_prescription"
    medication_to_use: Literal["CHEMICAL_SUBSTANCE_BNF_DESCR"] = (
        "CHEMICAL_SUBSTANCE_BNF_DESCR"
    )
    aggregation_type: Literal["ITEMS"] = "ITEMS"


class EpdGroupLogRecord(BaseModel):
    root_dir: ClassVar[Path] = initialised_dir(DATA_DIR / "grouped_epds")
    record_type: Literal["grouped_epds"] = "grouped_epds"
    name: EpdDatasetName
    group_method: ItemsByPrescription = ItemsByPrescription()
    latest_process_date: datetime | None = None
    update_required: bool = True

    @property
    def group_dir(self) -> Path:
        return initialised_dir(self.__class__.root_dir / self.group_method.name)

    @property
    def partition_path(self):
        return initialised_dir(
            self.group_dir
            / f"year={get_year(self.name)}"
            / f"month={get_month(self.name)}"
        )

    def is_valid_to_use(self) -> bool:
        return (
            self.latest_process_date is not None
            and not self.update_required
            and any(self.partition_path.glob("*.parquet"))
        )

    def corresponding_download(
        self, download_log: "EpdDownloadLog| None"
    ) -> DownloadLogRecord:
        if download_log is None:
            download_log = EpdDownloadLog.load_yaml()
        try:
            log = download_log[self.name]
            assert log.is_valid_to_use()
            return log
        except (KeyError, AssertionError) as e:
            raise FileNotFoundError(
                f"No corresponding download log record found for dataset {self.name}", e
            )

    @property
    def file_name(self):
        return f"{self.name}.parquet"


T = TypeVar("T", EpdGroupLogRecord, DownloadLogRecord)


class KeyValueStore(RootModel[dict[EpdDatasetName, T]], Generic[T]):
    __abstract__ = True
    log_file_name: ClassVar[str]
    root_dir: ClassVar[Path]

    @classmethod
    def log_file_path(cls):
        return initialised_dir(cls.root_dir) / cls.log_file_name

    def save_yaml(self) -> None:
        to_yaml_file(self.__class__.log_file_path(), self)

    def __getitem__(self, key: EpdDatasetName) -> T:
        return self.root.__getitem__(key)

    def __setitem__(self, key: EpdDatasetName, value: T) -> None:
        self.root.__setitem__(key, value)

    def get(self, key: EpdDatasetName) -> T | None:
        return self.root.get(key)


class EpdDownloadLog(KeyValueStore[DownloadLogRecord]):
    root_dir = DownloadLogRecord.root_dir
    log_file_name = "download_log.yml"

    @classmethod
    def load_yaml(cls) -> "EpdDownloadLog":
        if not cls.log_file_path().exists():
            return cls.model_validate({})
        return parse_yaml_file_as(EpdDownloadLog, cls.root_dir / cls.log_file_name)


class EpdGroupLog(KeyValueStore[EpdGroupLogRecord]):
    root_dir = EpdGroupLogRecord.root_dir
    log_file_name = "group_log.yml"

    @classmethod
    def load_yaml(cls) -> "EpdGroupLog":
        if not cls.log_file_path().exists():
            return cls.model_validate({})
        return parse_yaml_file_as(EpdGroupLog, cls.root_dir / cls.log_file_name)


if __name__ == "__main__":
    print(EpdDownloadLog.load_yaml()["EPD_202401"])
