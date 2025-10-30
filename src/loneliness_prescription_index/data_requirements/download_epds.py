"""
https://opendata.nhsbsa.net/dataset/english-prescribing-data-epd

the api provides practice code and postcode as geographical markers

other areas in the dataset appear to change over time
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipExtFile, ZipFile
from rich import progress
from typing import (
    BinaryIO,
    Generic,
    Literal,
    TypeVar,
)
import anyio
from loguru import logger
from pydantic import (
    HttpUrl,
    BaseModel,
)
import httpx
import pendulum
import polars as pl

from loneliness_prescription_index.data_requirements.epd_logging import (
    EpdDownloadLog,
    DownloadLogRecord,
    EpdDatasetName,
    EpdGroupLog,
    EpdGroupLogRecord,
    create_dataset_name,
)


class NhsbsaClient(httpx.AsyncClient):
    """by default we want an async client to be created with the base url.

    If further constraints need to be added to the client (eg headers), they should be added here"""

    def __init__(self) -> None:
        super().__init__(
            base_url="https://opendata.nhsbsa.net/api/3/action", timeout=None
        )


T = TypeVar("T")


class Response(BaseModel, Generic[T]):
    """Generic response expected from the base endpoint"""

    success: Literal[True]
    help: HttpUrl
    result: T

    @classmethod
    def parse_response(cls, response: httpx.Response) -> T:
        if response.is_error:
            logger.debug(response.headers)
            raise ConnectionError(response)
        data = response.json()
        if not data["success"]:
            raise ConnectionError(response, data)
        return cls.model_validate(data).result


class DatasetMetadata(BaseModel):
    bq_table_name: str  # name of the table we will try to query
    # name: str  # also name of the table we will try to query
    url: HttpUrl  # the url to read the csv from


class EpdDatasetMetadata(DatasetMetadata):
    """Example

    >>> data = EpdDatasetMetadata(bq_table_name="...", url=HttpUrl("https://www.example.com"), name="EPD_202201")
    >>> data.yearmonth
    '202201'
    >>> data.year
    2022
    >>> data.month
    1
    """

    name: EpdDatasetName  # also name of the table we will try to query
    url: HttpUrl  # the url to read the csv from
    zip_url: HttpUrl  # the url to read the zip from


class EpdDatasetMetadataCollection(BaseModel):
    author: str
    author_email: str
    num_resources: int
    resources: list[EpdDatasetMetadata]

    def find(self, name: EpdDatasetName):
        for resource in self.resources:
            if resource.name == name:
                return resource
        raise KeyError(f"dataset {name} does not exist")


async def get_package_list(
    client: NhsbsaClient, endpoint: Literal["/package_list"] = "/package_list"
) -> list[str]:
    return Response[list[str]].parse_response(await client.get(endpoint))


async def get_epd_metadata(
    endpoint: Literal["/package_show"] = "/package_show",
    package_name="english-prescribing-data-epd",
) -> EpdDatasetMetadataCollection:
    async with NhsbsaClient() as client:
        package_list = await get_package_list(client)
        if package_name not in package_list:
            raise ValueError(f"package: {package_name} was not found online")
        return Response[EpdDatasetMetadataCollection].parse_response(
            await client.get(endpoint, params={"id": package_name})
        )


def configure_download_items(
    log: EpdDownloadLog,
    metadata: EpdDatasetMetadataCollection,
    dataset_name: EpdDatasetName,
    force_update: bool = False,
) -> DownloadLogRecord:
    existing_item = log.root.get(dataset_name)
    remote_item = metadata.find(dataset_name)

    if existing_item is None:
        existing_item = DownloadLogRecord.model_validate(
            remote_item, from_attributes=True
        )
        log.root[dataset_name] = existing_item
    elif existing_item.url != remote_item.url:
        existing_item.url = remote_item.url
        existing_item.update_required = True
        logger.trace(f"the stored url for {dataset_name} has been updated")

    if not existing_item.is_valid_to_use() or force_update:
        existing_item.update_required = True
    if existing_item.update_required and existing_item.archive.exists():
        existing_item.archive.unlink()
        logger.warning(f"cache for {dataset_name} has been removed")

    if existing_item.update_required:
        logger.trace(f"{dataset_name} requires download")
    return existing_item


async def stream_download(
    client: NhsbsaClient, log_record: DownloadLogRecord, progress: progress.Progress
):
    async with await anyio.open_file(log_record.archive, mode="wb") as f:
        async with client.stream(
            "GET", log_record.zip_url.encoded_string(), follow_redirects=True
        ) as response:
            response.raise_for_status()
            content_length = int(response.headers["content-length"])
            progress_id = progress.add_task(
                f"downloading EPD dataset from {response.url}",
                total=content_length,
                filename=log_record.archive_name,
            )
            async for byte_chunk in response.aiter_bytes():
                await f.write(byte_chunk)
                progress.advance(progress_id, len(byte_chunk))
    log_record.latest_download_date = pendulum.now()
    log_record.update_required = False


async def async_stream_many(records: list[DownloadLogRecord]):
    with progress.Progress(
        progress.TextColumn("[bold blue]{task.fields[filename]}"),
        progress.BarColumn(),
        progress.DownloadColumn(),
        progress.TransferSpeedColumn(),
        "remaining",
        progress.TimeRemainingColumn(),
    ) as progress_bar:
        async with NhsbsaClient() as client:
            async with anyio.create_task_group() as tg:
                for record in records:
                    tg.start_soon(stream_download, client, record, progress_bar)


def download_epds(
    datasets: list[EpdDatasetName],
    log: EpdDownloadLog | None = None,
    overwrite: bool = False,
):
    if log is None:
        log = EpdDownloadLog.load_yaml()
    metadata = anyio.run(get_epd_metadata)
    logger.success("fetched remote metadata for the epd datasets")
    relevant_logs = [
        configure_download_items(log, metadata, name, overwrite) for name in datasets
    ]
    required_actions = [action for action in relevant_logs if action.update_required]
    if not required_actions:
        logger.info("no download required")
        return
    logger.info(f"{len(required_actions)} datasets require downloading")
    anyio.run(async_stream_many, required_actions)
    log.save_yaml()
    logger.success("download complete")


def configure_preprocess_items(
    group_log: EpdGroupLog,
    dataset_name: EpdDatasetName,
    force_update: bool = False,
):
    existing_item = group_log.root.get(dataset_name)
    if existing_item is None:
        existing_item = EpdGroupLogRecord(name=dataset_name)
        group_log.root[dataset_name] = existing_item
    if not existing_item.is_valid_to_use() or force_update:
        existing_item.update_required = True

    if existing_item.update_required and existing_item.partition_path.exists():
        map(Path.unlink, existing_item.partition_path.iterdir())
        logger.warning(f"processed cache for grouped {dataset_name} has been removed")
    if existing_item.update_required:
        logger.trace(f"{dataset_name} requires preprocessing")
    return existing_item


def process_epd(
    record: EpdGroupLogRecord,
    download_log: EpdDownloadLog,
    progress_bar: progress.Progress,
    task_id: progress.TaskID,
):
    """in testing we observed a reduction in dataset records of up to 80% when grouping by POSTCODE, PRACTICE_CODE and medication
    this reduction in dataset size is very useful for downstream aggregations,

    furthermore using a parquet format to partition by year and month provides flexibility and speed for downstream filtering opperations and
    reduces the storage requirements due to the inbuilt compression

    the available medication_column_to_use ("CHEMICAL_SUBSTANCE_BNF_DESCR" or "BNF_DESCRIPTION") grant different levels of detail:
    - CHEMICAL_SUBSTANCE_BNF_DESCR is the active ingredient representing different doses
    - BNF_DESCRIPTION is the full description including doseage and delivery method

    the available aggregation_type ("ITEMS", "TOTAL_QUANTITY, "ACTUAL_COST") represent different ways to aggregate the prescriptions:
    - ITEMS is the number of prescription items dispensed of a medication described by BNF_DESCRIPTION
    - TOTAL_QUANTITY is the total quantity of medication dispensed (e.g. number of tablets) described by BNF_DESCRIPTION
    - ACTUAL_COST is the total cost of the prescription

    by default, we assume that each item represents a prescription for a single patient for a fixed period of time
    an group defined by CHEMICAL_SUBSTANCE_BNF_DESCR and the sum of ITEMS represents an estimate for how many prescriptions for a medication were written in a month

    the caveats for this estimate include
    - some addictive medications will be prescribed to the same patient for shorter time periods and more frequently
    - some medications may be prescribed in multiple pack sizes for the same patient

    if a different grouping method is required it might be worth considering using different GroupLogRecord's and storing the parquets in different folders

    """
    archive = record.corresponding_download(download_log).archive
    progress_bar.start_task(task_id)
    with ZipFile(archive) as zf:
        progress_bar.update(
            task_id,
            description=f"reading dataset {record.name} from archive",
            start=True,
            spinner=progress.SpinnerColumn(speed=1),
        )
        df = (
            pl.scan_csv(
                zf.open(f"{record.name}.csv", force_zip64=True).read(),
                low_memory=True,
                infer_schema=False,
                schema_overrides={
                    "ITEMS": pl.Int64,
                    "TOTAL_QUANTITY": pl.Float64,
                    "ACTUAL_COST": pl.Float64,
                },
            )
            .group_by(
                practice_code="PRACTICE_CODE",
                postcode=pl.col("POSTCODE").str.replace_all(r"\s+", ""),
                medication=record.group_method.medication_to_use,
                year=pl.col("YEAR_MONTH").str.slice(0, 4).cast(pl.Int16),
                month=pl.col("YEAR_MONTH").str.slice(4, 2).cast(pl.Int8),
                # maintain_order=True,  # in case this is the reason for the schema length error
            )
            .agg(
                total_items=pl.col(record.group_method.aggregation_type).sum(),
                # total_records=pl.len(),
            )
        )
        df.sink_parquet(
            pl.PartitionByKey(
                record.group_dir,
                by=["year", "month", "medication"],
            ),
            mkdir=True,
        )
        progress_bar.update(
            task_id,
            description=f"completed writing dataset {record.name} as partitioned parquet",
            completed=1,
            spinner=progress.SpinnerColumn(speed=0),
        )
        record.update_required = False
        record.latest_process_date = pendulum.now()


def preprocess_downloaded_epds(
    download_log: EpdDownloadLog,
    group_log: EpdGroupLog,
    datasets: list[EpdDatasetName],
    overwrite: bool = False,
):
    with progress.Progress(
        progress.TextColumn("[bold blue]{task.description}"),
        progress.SpinnerColumn(speed=0),
        progress.TimeElapsedColumn(),
    ) as progress_bar:
        records = [
            configure_preprocess_items(group_log, dataset_name, overwrite)
            for dataset_name in datasets
        ]
        actions = [record for record in records if record.update_required]
        tasks = [
            progress_bar.add_task(
                f"waiting to process EPD dataset {record.name}", start=False
            )
            for record in actions
        ]
        for task_record, task_id in zip(actions, tasks):
            process_epd(task_record, download_log, progress_bar, task_id)
    group_log.save_yaml()
    logger.success("preprocessing complete")


def preprocess_epds(
    datasets: list[EpdDatasetName],
    download_log: EpdDownloadLog | None = None,
    overwrite: bool = False,
):
    if download_log is None:
        download_log = EpdDownloadLog.load_yaml()
    download_epds(datasets, download_log, overwrite)
    group_log = EpdGroupLog.load_yaml()
    preprocess_downloaded_epds(download_log, group_log, datasets, overwrite)


def preprocess_epds_for_year(
    year: int, log: EpdDownloadLog | None = None, overwrite: bool = False
):
    if log is None:
        log = EpdDownloadLog.load_yaml()
    datasets = [create_dataset_name(year, month) for month in range(1, 13)]
    preprocess_epds(datasets, log, overwrite)
    logger.success(f"data for year {year} is available")


if __name__ == "__main__":
    preprocess_epds_for_year(2023)
