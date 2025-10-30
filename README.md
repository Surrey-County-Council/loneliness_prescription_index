# Prescribing indexes
The aim for this repo is to enable analysts to quickly set up, and complete analysis on the English Prescribing Datasets

The dependencies for the repo can be recreated using [uv](https://docs.astral.sh/uv/getting-started/)

```shell
pip install uv
uv run ...
```

The above command will be sufficient for most use cases with a python 3.12 instalation to begin analysis on an epd dataset, 
this automates installation of project dependencies, the creation of a vitual enviornment, and fetching of the epd data required for the year specified.

Optimisations using async requests and polars LazyFrames mean data for a year (up to 1TB when in csv form) can be sourced and transformed in under half an hour on most computers, (my current run time is just over 15 minutes). 

Obvious caveats for this claim include download speed and pc specs so mileage may vary.

The full data is partitioned into parquet files, further enhancing the speed of analysis in typical use cases (grouping or filtering by year, month and medication). An understanding of the following is required for analysis:
- polars LazyFrame and the scan_parquet operation
- hive style parquet partitioning in polars (ie. the ability to read multiple parquet files by providing a directory containing only parquet files, rather than a single file, you do not need to use Hive or Hadoop!)

Partitioning is used so that the polars query optimiser can apply filters based on file path, this massively improves performance for data of this scale in read and write opperations. However this feature is unstable at the time of writing, so analysts wanting to use alternative versions of polars may encounter issues.

a testing.ipynb file is provided for analysts to explore how to read and work with the above format. jupyter and ipython dependencies are included in the requirements file (pyproject.toml)

Of note for pandas users - polars provides the .to_pandas() method on dataframes, so following optimising the read filter and group steps, you can return to pandas if you wish. (in fact that is reccommended for GeoDataFrame analysis, where polars has far less support at the time of writing)

The EPD grouping applied to get parquet files is defined using pydantic models. the models describe the users desired approach to downloading and grouping the data. Currently users requiring different grouping strategies will need to fork the repo and add their own code, but the design should make some customisation eaier for users who understant the EPS csv schema, and the fundamentals of Pydantic models.

Grouping and downloading of the data is logged using a simple key value yaml store. this avoid a requirement for database creation and exposes the logs and configurations in a readable format within the data directories.

forking the repo to meet your own needs is encouraged! as are pull requests and questions. (see contact info below).

For users wishing to aggregate the data to different area types, the epd datasets can be linked via ODS codes to nhs organisations and by postcode to the ONS NSPL datasets.
Automation for this process is currently not feasible and it is recommended that the user understands their need to ensure a valid area type mapping from an EPD to the relevant mapping for other area types. This may be different based on the year of the EPD.

However, the scripts provided simplify the retrieval of data where it cannot be automated. see:

src/loneliness_prescription_index/data_requirements/preprocess_nspl.py

src/loneliness_prescription_index/data_requirements/preprocess_ods_codes.py



# Thanks to our ONS colleagues for providing the files for this pipeline
the intention for development has been to streamline the speed with which analysis can get started.

While the initial pipeline was explicitly related to exploring loneliness (and our intention is to reproduce this), 
there is significant room to use the EPD data for other pieces of analysis.

