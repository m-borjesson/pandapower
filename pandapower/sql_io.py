# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd
import numpy as np

from pandapower import io_utils, pandapowerNet

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.errors
    import psycopg2.sql as psql

    PSYCOPG2_INSTALLED = True
except ImportError:
    psycopg2 = None  # type: ignore[assignment]
    PSYCOPG2_INSTALLED = False

try:
    import sqlite3

    SQLITE_INSTALLED = True
except ImportError:
    sqlite3 = None  # type: ignore[assignment]
    SQLITE_INSTALLED = False

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        LongType,
        DoubleType,
        BooleanType,
        TimestampType,
    )

    PYSPARK_INSTALLED = True
except ImportError:
    SparkSession = None  # type: ignore[assignment]
    StructType = None  # type: ignore[assignment]
    StructField = None  # type: ignore[assignment]
    StringType = None  # type: ignore[assignment]
    LongType = None  # type: ignore[assignment]
    DoubleType = None  # type: ignore[assignment]
    BooleanType = None  # type: ignore[assignment]
    TimestampType = None  # type: ignore[assignment]
    PYSPARK_INSTALLED = False

import logging

logger = logging.getLogger(__name__)


def match_sql_type(dtype):
    if dtype in ("float", "float32", "float64"):
        return "double precision"
    elif dtype in ("int", "int32", "int64", "uint32", "uint64", "Int64"):
        return "bigint"
    elif dtype in ("object", "str"):
        return "varchar"
    elif dtype == "bool":
        return "boolean"
    elif "datetime" in dtype:
        return "timestamp"
    else:
        raise UserWarning(f"unsupported type {dtype}")


def check_if_sql_table_exists(cursor, table_name):
    query = f"SELECT EXISTS (SELECT FROM information_schema.tables " \
            f"WHERE table_schema = '{table_name.split('.')[0]}' " \
            f"AND table_name = '{table_name.split('.')[-1]}');"
    cursor.execute(query)
    (exists,) = cursor.fetchone()
    return exists


def get_sql_table_columns(cursor, table_name):
    query = f"SELECT * FROM information_schema.columns " \
            f"WHERE table_schema = '{table_name.split('.')[0]}' " \
            f"AND table_name   = '{table_name.split('.')[-1]}';"
    cursor.execute(query)
    colnames = [desc[0] for desc in cursor.description]
    list_idx = colnames.index("column_name")
    columns_data = cursor.fetchall()
    columns = [c[list_idx] for c in columns_data]
    return columns


def download_sql_table(cursor, table_name, **id_columns):
    # first we check if table exists:
    exists = check_if_sql_table_exists(cursor, table_name)
    if not exists:
        raise UserWarning(f"table {table_name} does not exist or the user has no access to it")

    if len(id_columns.keys()) == 0:
        query = f"SELECT * FROM {table_name}"
    else:
        columns_string = ' and '.join([f"{str(k)} = '{str(v)}'" for k, v in id_columns.items()])
        query = f"SELECT * FROM {table_name} WHERE {columns_string}"

    cursor.execute(query)
    colnames = [desc[0] for desc in cursor.description]
    table = cursor.fetchall()
    df = pd.DataFrame(table, columns=colnames)
    with pd.option_context('future.no_silent_downcasting', True):
        df = df.fillna(np.nan).infer_objects()
    index_name = f"{table_name.split('.')[-1]}_id"
    if index_name in df.columns:
        df = df.set_index(index_name)
    if len(id_columns) > 0:
        df.drop(id_columns.keys(), axis=1, inplace=True)
    return df


def upload_sql_table(conn, cursor, table_name, table, index_name=None, timestamp=False, **id_columns):
    # index_name allows using a custom column for the table index and disregard the DataFrame index,
    # otherwise a <table_name>_id is used as index_name and DataFrame index is also uploaded to the database
    table = table.where(pd.notnull(table), None)
    if index_name is None:
        index_name = f"{table_name.split('.')[-1]}_id"
        index_type = match_sql_type(str(table.index.dtype))
        table_columns = [c for c in table.columns if c not in id_columns]
        tuples_index = True
    else:
        index_type = match_sql_type(str(table[index_name].dtype))
        table_columns = [c for c in table.columns if c != index_name and c not in id_columns]
        tuples_index = False

    # Create a list of tuples from the dataframe values
    if len(id_columns.keys()) > 0:
        tuples = [(*tuple(x), *id_columns.values()) for x in table[table_columns].itertuples(index=tuples_index)]
    else:
        tuples = [tuple(x) for x in table[table_columns].itertuples(index=tuples_index)]
    # Replace pd.NA values with None for conversion to postgres NULL
    tuples = [tuple(None if value is pd.NA else value for value in row) for row in tuples]

    # Comma-separated dataframe columns
    sql_columns = [index_name, *table_columns, *id_columns.keys()]
    sql_column_types = [index_type,
                        *[match_sql_type(t) for t in table[table_columns].dtypes.astype(str).values],
                        *[match_sql_type(np.result_type(type(v)).name) for v in id_columns.values()]]

    # check if all columns already exist and if not, add more columns
    existing_columns = get_sql_table_columns(cursor, table_name)
    new_columns = [('"%s"' % c, t) for c, t in zip(sql_columns, sql_column_types) if c not in existing_columns]
    if len(new_columns) > 0:
        logger.info(f"adding columns {new_columns} to table {table_name}")
        column_statement = ", ".join(f"ADD COLUMN {c} {t}" for c, t in new_columns)
        query = f"ALTER TABLE {table_name} {column_statement};"
        cursor.execute(query)
        conn.commit()

    if timestamp:
        add_timestamp_column(conn, cursor, table_name)

    # SQL query to execute
    columns = [psql.Identifier(c.replace('%', '%%')) for c in sql_columns]
    query = psql.SQL("INSERT INTO {tbl}({fields}) VALUES({placeholders})").format(
        tbl=psql.Identifier(*table_name.split('.')),
        fields=psql.SQL(',').join(columns),
        placeholders=psql.SQL(',').join(psql.Placeholder() * len(sql_columns))
    )
    
    # batch_size = 1000
    # for chunk in tqdm(chunked(tuples, batch_size)):
    #     cursor.executemany(query, chunk)
    #     conn.commit()
    psycopg2.extras.execute_batch(cursor, query, tuples, page_size=100)
    conn.commit()


def check_postgresql_catalogue_table(cursor, table_name, grid_id, grid_id_column, download=False):
    table_exists = check_if_sql_table_exists(cursor, table_name)

    if not table_exists:
        if download:
            raise UserWarning(f"grid catalogue {table_name} does not exist")
        else:
            query = f"CREATE TABLE {table_name} ({grid_id_column} BIGSERIAL PRIMARY KEY, " \
                    f"timestamp TIMESTAMPTZ DEFAULT now());"
            cursor.execute(query)
    else:
        existing_columns = get_sql_table_columns(cursor, table_name)
        if grid_id_column not in existing_columns:
            raise UserWarning(f"grid_id_column {grid_id_column} is missing in grid catalogue {table_name}")
        if grid_id is None:
            if download:
                raise UserWarning(f"grid_id ({grid_id_column}) is None: {grid_id}")
            return  # we don't need to check for duplicates if grid_id is None (means we are uploading a new net)
        query = f"SELECT COUNT(*) FROM {table_name} where {grid_id_column}={grid_id}"
        cursor.execute(query)
        (found,) = cursor.fetchone()
        if download and found == 0:
            raise UserWarning(f"found no entries in {table_name} where {grid_id_column}={grid_id}")
        if not download and found > 0:
            raise UserWarning(f"found {found} duplicate entries in grid_catalogue where {grid_id_column}={grid_id}")


def create_postgresql_catalogue_entry(conn, cursor, grid_id, grid_id_column, catalogue_table_name):
    # check if a grid with the provided ids was already added
    check_postgresql_catalogue_table(cursor, catalogue_table_name, grid_id, grid_id_column)
    # create a "catalogue" table to keep track of all grids available in the DB
    query = f"INSERT INTO {catalogue_table_name}({grid_id_column}) VALUES({'DEFAULT' if grid_id is None else grid_id}) " \
            f"RETURNING {grid_id_column}"
    cursor.execute(query)
    conn.commit()
    (written_grid_id,) = cursor.fetchone()
    return written_grid_id


def add_timestamp_column(conn, cursor, table_name):
    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS timestamp TIMESTAMPTZ;")
    conn.commit()
    cursor.execute(f"ALTER TABLE {table_name} ALTER COLUMN timestamp SET DEFAULT now();")
    conn.commit()


def create_sql_table_if_not_exists(conn, cursor, table_name, grid_id_column, catalogue_table_name):
    query = f"CREATE TABLE IF NOT EXISTS {table_name}({grid_id_column} BIGINT, " \
            f"FOREIGN KEY({grid_id_column}) REFERENCES {catalogue_table_name}({grid_id_column})" \
            f"ON DELETE CASCADE);"
    cursor.execute(query)
    conn.commit()


def delete_postgresql_net(
        grid_id: int,
        host: str,
        user: str,
        password: str,
        database: str,
        schema: str,
        grid_id_column: str = "grid_id",
        grid_catalogue_name: str = "grid_catalogue",
        port: Optional[int] = None
) -> None:
    """
    Removes a grid model from the PostgreSQL database.

    :param grid_id: unique grid_id that will be used to identify the data for the grid model
    :param host: hostname for the DB, e.g. "localhost"
    :param user:
    :param password:
    :param database: name of the database
    :param schema: name of the database schema (e.g. 'postgres')
    :param grid_id_column: name of the column for "grid_id" in the PosgreSQL tables, default="grid_id".
    :param grid_catalogue_name: name of the catalogue table that includes all grid_id values and the timestamp when the
        grid data were added
    :param port: port at which the database is listening
    """
    if not PSYCOPG2_INSTALLED:
        raise UserWarning("install the package psycopg2 to use PostgreSQL I/O in pandapower")

    conn = psycopg2.connect(host=host, user=user, password=password, database=database, port=port)
    cursor = conn.cursor()
    catalogue_table_name = grid_catalogue_name if schema is None else f"{schema}.{grid_catalogue_name}"
    check_postgresql_catalogue_table(cursor, catalogue_table_name, grid_id, grid_id_column, download=True)
    query = f"DELETE FROM {catalogue_table_name} WHERE {grid_id_column}={grid_id};"
    cursor.execute(query)
    # query = f'DROP SCHEMA IF EXISTS "{schema}" CASCADE; CREATE SCHEMA IF NOT EXISTS "{schema}";'
    # cursor.execute(query)
    conn.commit()


def from_sql(conn, schema, grid_id, grid_id_column="grid_id", grid_catalogue_name="grid_catalogue",
             empty_dict_like_object=None, grid_tables=None):
    """
    Downloads an existing pandapowerNet from a PostgreSQL database.

    Parameters
    ----------
    conn : connection to SQL database (e.g. SQLite, PostgreSQL)
    schema : str
        name of the database schema (e.g. 'postgres')
    grid_id : int
        unique grid_id that will be used to identify the data for the grid model
    grid_id_column : str
        name of the column for "grid_id" in the PosgreSQL tables, default="grid_id".
    grid_catalogue_name : str
        name of the catalogue table that includes all grid_id values and the timestamp when the grid data were added
    empty_dict_like_object : dict-like
        If None, the output of pandapower.create_empty_network() is used as an empty element to be filled by
        the grid data. Give another dict-like object to start filling that alternative object with the data.

    Returns
    -------
    net : pandapowerNet
    """
    cursor = conn.cursor()
    id_columns = {grid_id_column: grid_id}
    if grid_tables is None:
        catalogue_table_name = grid_catalogue_name if schema is None else f"{schema}.{grid_catalogue_name}"
        check_postgresql_catalogue_table(cursor, catalogue_table_name, grid_id, grid_id_column, download=True)
        grid_tables = download_sql_table(cursor, "grid_tables" if schema is None else f"{schema}.grid_tables", **id_columns)

    d = {}
    for element in grid_tables.table.values:
        table_name = element if schema is None else f"{schema}.{element}"
        try:
            tab = download_sql_table(cursor, table_name, **id_columns)
        except UserWarning as err:
            logger.debug(err)
            continue
        except psycopg2.errors.UndefinedTable as err:
            logger.info(f"skipped {element} due to error: {err}")
            continue

        if 'geo' in tab.columns:
            tab.geo = tab.geo.replace({'NaN': None})

        d[element] = tab

    net = io_utils.from_dict_of_dfs(d, net=empty_dict_like_object)

    return net


def to_sql(net, conn, schema, include_results=False, grid_id=None, grid_id_column="grid_id",
           grid_catalogue_name="grid_catalogue", index_name=None):
    """
    Uploads a pandapowerNet to a PostgreSQL database. The database must exist, the element tables
    are created if they do not exist.
    TODO: JSON serialization (e.g. for controller objects) is not implemented yet.

    Parameters
    ----------
    net : pandapowerNet
        the grid model to be uploaded to the database
    conn : connection to SQL database (e.g. SQLite, PostgreSQL)
    schema : str
        name of the database schema (e.g. 'postgres')
    include_results : bool
        specify whether the power flow results are included when the grid is uploaded, default=False
    grid_id : int
        unique grid_id that will be used to identify the data for the grid model, default None.
        If None, it will be set automatically by PostgreSQL
    grid_id_column : str
        name of the column for "grid_id" in the PosgreSQL tables, default="grid_id".
    grid_catalogue_name : str
        name of the catalogue table that includes all grid_id values and the timestamp when the grid data were added
    index_name : str
        name of the custom column to be used inplace of index in the element tables if it is not the standard DataFrame index

    Returns
    -------
    grid_id: int
        returns either the user-specified grid_id or the automatically generated grid_id of the grid model
    """
    cursor = conn.cursor()
    catalogue_table_name = grid_catalogue_name if schema is None else f"{schema}.{grid_catalogue_name}"
    d = io_utils.to_dict_of_dfs(net, include_results=include_results, include_empty_tables=False)
    written_grid_id = create_postgresql_catalogue_entry(conn, cursor, grid_id, grid_id_column, catalogue_table_name)
    id_columns = {grid_id_column: written_grid_id}
    d["grid_tables"] = pd.DataFrame(d.keys(), columns=["table"])
    for element, element_table in d.items():
        table_name = element if schema is None else f"{schema}.{element}"
        # None causes postgresql error, np.nan is better
        create_sql_table_if_not_exists(conn, cursor, table_name, grid_id_column, catalogue_table_name)
        upload_sql_table(conn=conn, cursor=cursor, table_name=table_name, table=element_table,
                         index_name=index_name, **id_columns)
        logger.debug(f"uploaded table {element}")
    return written_grid_id


def to_sqlite(net, filename, include_results=False):
    """
    Saves pandapowerNet an SQLite format

    Parameters
    ----------
    net : grid model
        pandapowerNet
    filename : path to a text file where the data will be stored
        str
    include_results : whether result tables should be included
        bool
    """
    if not SQLITE_INSTALLED:
        raise UserWarning("sqlite3 is not installed, install sqlite3 to use from_sqlite()")
    with sqlite3.connect(filename) as conn:
        dodfs = io_utils.to_dict_of_dfs(net, include_results=include_results)
        for name, data in dodfs.items():
            data.to_sql(name, conn)


def from_sqlite(filename):
    """
    Loads a grid model from SQLite format

    Parameters
    ----------
    filename : path to the text file where the data are stored

    Returns
    -------
    net : the grid model
        pandapowerNet
    """
    if not SQLITE_INSTALLED:
        raise UserWarning("sqlite3 is not installed, install sqlite3 to use from_sqlite()")
    with sqlite3.connect(filename) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        dodfs = {}
        for t, in cursor.fetchall():
            table = pd.read_sql_query("SELECT * FROM '%s'" % t, conn, index_col="index")
            table.index.name = None
            dodfs[t] = table
        net = io_utils.from_dict_of_dfs(dodfs)
    return net


def to_postgresql(
        net: pandapowerNet,
        host: str,
        user: str,
        password: str,
        database: str,
        schema: str,
        include_results: bool = False,
        grid_id: Optional[int] = None,
        grid_id_column: str = "grid_id",
        grid_catalogue_name: str = "grid_catalogue",
        index_name=None,
        port: Optional[int] = None
    ) -> int:
    """
    Uploads a pandapowerNet to a PostgreSQL database. The database must exist, the element tables
    are created if they do not exist.
    JSON serialization (e.g. for controller objects) is not implemented yet.

    :param pandapowerNet net: the grid model to be uploaded to the database
    :param str host: hostname for connecting to the database
    :param str user: username for logging in
    :param str password:
    :param str database: name of the database
    :param str schema: name of the database schema (e.g. 'postgres')
    :param bool include_results: specify whether the power flow results are included when the grid is uploaded
    :param int grid_id: unique grid_id that will be used to identify the data for the grid model, default None.
        If None, it will be set automatically by PostgreSQL
    :param str grid_id_column: name of the column for "grid_id" in the PosgreSQL tables, default="grid_id".
    :param str grid_catalogue_name: name of the catalogue table that includes all grid_id values and the timestamp when
        the grid data were added
    :param str index_name: name of the custom column to be used inplace of index in the element tables if it is not the
        standard DataFrame index
    :param port: the port to use for the PostgreSQL connection
    :return: returns either the user-specified grid_id or the automatically generated grid_id of the grid model
    """
    if not PSYCOPG2_INSTALLED:
        raise UserWarning("install the package psycopg2 to use PostgreSQL I/O in pandapower")
    logger.debug(f"Uploading the grid data to the DB schema {schema}")
    with psycopg2.connect(host=host, user=user, password=password, database=database, port=port) as conn:
        grid_id = to_sql(net, conn, schema, include_results, grid_id, grid_id_column, grid_catalogue_name, index_name)
    return grid_id


def from_postgresql(
        grid_id: int,
        host: str,
        user: str,
        password: str,
        database: str,
        schema: str,
        grid_id_column: str = "grid_id",
        grid_catalogue_name: str = "grid_catalogue",
        empty_dict_like_object: Optional[dict] = None,
        grid_tables = None,
        port: Optional[int] = None
):
    """
    Downloads an existing pandapowerNet from a PostgreSQL database.

    :param int grid_id: unique grid_id that will be used to identify the data for the grid model
    :param str host: hostname for connecting to the database
    :param str user: username for logging in
    :param str password:
    :param str database: name of the database
    :param str schema: name of the database schema (e.g. 'postgres')
    :param str grid_id_column: name of the column for "grid_id" in the PosgreSQL tables, default="grid_id".
    :param str grid_catalogue_name: name of the catalogue table that includes all grid_id values and the timestamp when
        the grid data were added
    :param empty_dict_like_object: If None, the output of pandapower.create_empty_network() is used as an empty element
        to be filled by the grid data.
        Give another dict-like object to start filling that alternative object with the data.
    :param grid_tables:
    :param port: port for connecting to the database
    :return: the loaded pandapower network
    """
    if not PSYCOPG2_INSTALLED:
        raise UserWarning("install the package psycopg2 to use PostgreSQL I/O in pandapower")

    with psycopg2.connect(host=host, user=user, password=password, database=database, port=port) as conn:
        net = from_sql(conn, schema, grid_id, grid_id_column, grid_catalogue_name, empty_dict_like_object, grid_tables)

    return net


def _coerce_pdf_to_spark_schema(pdf: pd.DataFrame, schema) -> pd.DataFrame:
    """
    Coerce pandas DataFrame columns to match the provided Spark schema.
    This avoids Arrow conversion errors when pandas 'object' columns contain
    floats, ints, bools, timestamps, etc.
    """
    coerced = pdf.copy()

    for field in schema.fields:
        col = field.name
        spark_type = field.dataType

        if col not in coerced.columns:
            continue

        series = coerced[col]

        if isinstance(spark_type, StringType):
            coerced[col] = series.where(series.isna(), series.astype(str))

        elif isinstance(spark_type, LongType):
            coerced[col] = pd.to_numeric(series, errors="coerce").astype("Int64")

        elif isinstance(spark_type, DoubleType):
            coerced[col] = pd.to_numeric(series, errors="coerce").astype(float)

        elif isinstance(spark_type, BooleanType):

            def _to_bool(value):
                if pd.isna(value):
                    return None
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return bool(value)
                if isinstance(value, str):
                    value_lower = value.strip().lower()
                    if value_lower in {"true", "1", "yes", "y"}:
                        return True
                    if value_lower in {"false", "0", "no", "n"}:
                        return False
                return bool(value)

            coerced[col] = series.map(_to_bool).astype(object)

        elif isinstance(spark_type, TimestampType):
            coerced[col] = pd.to_datetime(series, errors="coerce")

    coerced = coerced.astype(object).where(pd.notnull(coerced), None)
    return coerced


def _qualify_table_name(catalog_name: str | None, schema_name: str, table_name: str) -> str:
    """
    Build a fully qualified Spark table name.

    Examples
    --------
    catalog_name="dev_sandbox", schema_name="kraftsystemanalys", table_name="bus"
    -> "dev_sandbox.kraftsystemanalys.bus"

    catalog_name=None, schema_name="kraftsystemanalys", table_name="bus"
    -> "kraftsystemanalys.bus"
    """
    if catalog_name:
        return f"{catalog_name}.{schema_name}.{table_name}"
    return f"{schema_name}.{table_name}"


def _pandas_dtype_to_spark_type(dtype_str: str):
    """Map a pandas dtype string to a PySpark DataType instance."""
    dtype_lower = dtype_str.lower()

    if "int" in dtype_lower:
        return LongType()
    if "float" in dtype_lower:
        return DoubleType()
    if "bool" in dtype_lower:
        return BooleanType()
    if "datetime" in dtype_lower:
        return TimestampType()
    return StringType()


def _build_spark_schema(df: pd.DataFrame):
    """
    Build a PySpark StructType schema from a pandas DataFrame, including its index
    as the first field after reset_index().
    """
    index_df = df.reset_index()
    fields = [
        StructField(col, _pandas_dtype_to_spark_type(str(dtype)), nullable=True)
        for col, dtype in zip(index_df.columns, index_df.dtypes)
    ]
    return StructType(fields)


def _spark_table_exists(spark, full_table_name: str) -> bool:
    """Check whether a Spark SQL table exists."""
    try:
        spark.table(full_table_name)
        return True
    except Exception:
        return False


def _check_spark_catalogue(
    spark, full_catalogue_name: str, grid_id: Optional[int],
    grid_id_column: str, download: bool = False
) -> None:
    """
    Validate the Spark grid catalogue table, mirroring check_postgresql_catalogue_table.

    - download=False (upload): ensures catalogue exists (creates if absent) and grid_id is not a duplicate.
    - download=True (download/delete): ensures catalogue exists and grid_id is present.
    """
    exists = _spark_table_exists(spark, full_catalogue_name)
    if not exists:
        if download:
            raise UserWarning(f"grid catalogue {full_catalogue_name} does not exist")
        # Create an empty catalogue table so subsequent appends work
        catalogue_schema = StructType([
            StructField(grid_id_column, LongType(), nullable=False),
            StructField("timestamp", TimestampType(), nullable=True),
        ])
        spark.createDataFrame([], catalogue_schema).write.mode("error").saveAsTable(full_catalogue_name)
        return

    if grid_id is None:
        if download:
            raise UserWarning(f"grid_id ({grid_id_column}) is None: {grid_id}")
        return  # uploading a new net – auto-assign grid_id later; no duplicate check needed

    count_row = spark.sql(
        f"SELECT COUNT(*) AS cnt FROM {full_catalogue_name} WHERE {grid_id_column} = {int(grid_id)}"
    ).collect()
    found = count_row[0]["cnt"]

    if download and found == 0:
        raise UserWarning(f"found no entries in {full_catalogue_name} where {grid_id_column}={grid_id}")
    if not download and found > 0:
        raise UserWarning(f"found {found} duplicate entries in grid_catalogue where {grid_id_column}={grid_id}")


def _create_spark_catalogue_entry(
    spark, full_catalogue_name: str, grid_id: Optional[int], grid_id_column: str
) -> int:
    """
    Create a new entry in the Spark grid catalogue and return the written grid_id.
    Mirrors create_postgresql_catalogue_entry.
    """
    from datetime import datetime, timezone

    _check_spark_catalogue(spark, full_catalogue_name, grid_id, grid_id_column, download=False)

    if grid_id is None:
        row = spark.sql(f"SELECT MAX({grid_id_column}) AS max_id FROM {full_catalogue_name}").collect()
        max_id = row[0]["max_id"]
        grid_id = 1 if max_id is None else int(max_id) + 1
    else:
        grid_id = int(grid_id)

    catalogue_row = pd.DataFrame({
        grid_id_column: pd.array([grid_id], dtype="Int64"),
        "timestamp": [datetime.now(tz=timezone.utc)],
    })
    catalogue_schema = StructType([
        StructField(grid_id_column, LongType(), nullable=False),
        StructField("timestamp", TimestampType(), nullable=True),
    ])
    (
        spark.createDataFrame(catalogue_row, catalogue_schema)
        .write.mode("append")
        .saveAsTable(full_catalogue_name)
    )
    return grid_id


def _spark_delete_rows(spark, full_table_name: str, grid_id_column: str, grid_id: int) -> None:
    """Remove all rows with the given grid_id from a Spark table."""
    safe_grid_id = int(grid_id)
    try:
        spark.sql(f"DELETE FROM {full_table_name} WHERE {grid_id_column} = {safe_grid_id}")
    except Exception:
        # Fallback for non-Delta tables: read → filter → overwrite
        existing_df = spark.table(full_table_name)
        filtered_df = existing_df.filter(existing_df[grid_id_column] != safe_grid_id)
        filtered_df.write.mode("overwrite").saveAsTable(full_table_name)


def to_spark(
    net,
    spark,
    schema_name: str,
    include_results: bool = False,
    grid_id: Optional[int] = None,
    grid_id_column: str = "grid_id",
    grid_catalogue_name: str = "grid_catalogue",
    catalog_name: str | None = None,
) -> int:
    """
    Saves a pandapowerNet to Spark SQL tables, with multi-grid support via a grid catalogue.

    Parameters
    ----------
    net : pandapowerNet
        the grid model to be stored in Spark SQL
    spark : pyspark.sql.SparkSession
        the active Spark session
    schema_name : str
        Spark schema/database name
    include_results : bool
        whether to include result tables, default=False
    grid_id : int or None
        unique grid_id that will be used to identify the data for the grid model.
        If None, it will be assigned automatically.
    grid_id_column : str
        name of the column for "grid_id" in the Spark tables, default="grid_id".
    grid_catalogue_name : str
        name of the catalogue table that tracks all grids, default="grid_catalogue".
    catalog_name : str | None
        Spark catalog name, e.g. in Databricks Unity Catalog

    Returns
    -------
    grid_id : int
        the grid_id assigned to the uploaded grid model
    """
    if not PYSPARK_INSTALLED:
        raise UserWarning("install pyspark to use Spark SQL I/O in pandapower")

    if catalog_name:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
    else:
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {schema_name}")

    catalogue_full_name = _qualify_table_name(catalog_name, schema_name, grid_catalogue_name)
    written_grid_id = _create_spark_catalogue_entry(spark, catalogue_full_name, grid_id, grid_id_column)

    dodfs = io_utils.to_dict_of_dfs(
        net,
        include_results=include_results,
        include_empty_tables=False,
    )
    dodfs["grid_tables"] = pd.DataFrame(list(dodfs.keys()), columns=["table"])

    for name, data in dodfs.items():
        data_with_id = data.copy()
        data_with_id[grid_id_column] = written_grid_id

        schema = _build_spark_schema(data_with_id)
        index_df = data_with_id.reset_index()
        clean_df = _coerce_pdf_to_spark_schema(index_df, schema)

        spark_df = spark.createDataFrame(clean_df, schema=schema)
        full_table_name = _qualify_table_name(catalog_name, schema_name, name)
        spark_df.write.mode("append").saveAsTable(full_table_name)

    return written_grid_id


def from_spark(
    spark,
    schema_name: str,
    grid_id: int,
    grid_id_column: str = "grid_id",
    grid_catalogue_name: str = "grid_catalogue",
    catalog_name: str | None = None,
) -> pandapowerNet:
    """
    Loads a pandapowerNet from Spark SQL tables.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        the active Spark session
    schema_name : str
        Spark schema/database name
    grid_id : int
        unique grid_id that identifies the grid model to load
    grid_id_column : str
        name of the column for "grid_id" in the Spark tables, default="grid_id".
    grid_catalogue_name : str
        name of the catalogue table that tracks all grids, default="grid_catalogue".
    catalog_name : str | None
        Spark catalog name, e.g. in Databricks Unity Catalog

    Returns
    -------
    net : pandapowerNet
    """
    if not PYSPARK_INSTALLED:
        raise UserWarning("install pyspark to use Spark SQL I/O in pandapower")

    safe_grid_id = int(grid_id)
    catalogue_full_name = _qualify_table_name(catalog_name, schema_name, grid_catalogue_name)
    _check_spark_catalogue(spark, catalogue_full_name, safe_grid_id, grid_id_column, download=True)

    grid_tables_full_name = _qualify_table_name(catalog_name, schema_name, "grid_tables")
    grid_tables_pdf = (
        spark.table(grid_tables_full_name)
        .filter(f"{grid_id_column} = {safe_grid_id}")
        .drop(grid_id_column)
        .toPandas()
    )
    if "index" in grid_tables_pdf.columns:
        grid_tables_pdf = grid_tables_pdf.set_index("index")
        grid_tables_pdf.index.name = None

    dodfs = {}
    for element in grid_tables_pdf["table"].values:
        full_table_name = _qualify_table_name(catalog_name, schema_name, element)
        try:
            pdf = (
                spark.table(full_table_name)
                .filter(f"{grid_id_column} = {safe_grid_id}")
                .drop(grid_id_column)
                .toPandas()
            )
        except Exception as e:
            logger.debug(f"skipped {element} due to error: {e}")
            continue

        if "index" in pdf.columns:
            pdf = pdf.set_index("index")
            pdf.index.name = None

        dodfs[element] = pdf

    net = io_utils.from_dict_of_dfs(dodfs)
    return net


def delete_spark_net(
    spark,
    schema_name: str,
    grid_id: int,
    grid_id_column: str = "grid_id",
    grid_catalogue_name: str = "grid_catalogue",
    catalog_name: str | None = None,
) -> None:
    """
    Removes a grid model from Spark SQL tables.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        the active Spark session
    schema_name : str
        Spark schema/database name
    grid_id : int
        unique grid_id that identifies the grid model to delete
    grid_id_column : str
        name of the column for "grid_id" in the Spark tables, default="grid_id".
    grid_catalogue_name : str
        name of the catalogue table that tracks all grids, default="grid_catalogue".
    catalog_name : str | None
        Spark catalog name, e.g. in Databricks Unity Catalog
    """
    if not PYSPARK_INSTALLED:
        raise UserWarning("install pyspark to use Spark SQL I/O in pandapower")

    safe_grid_id = int(grid_id)
    catalogue_full_name = _qualify_table_name(catalog_name, schema_name, grid_catalogue_name)
    _check_spark_catalogue(spark, catalogue_full_name, safe_grid_id, grid_id_column, download=True)

    # Load grid_tables to find which element tables to clean up
    grid_tables_full_name = _qualify_table_name(catalog_name, schema_name, "grid_tables")
    grid_tables_pdf = (
        spark.table(grid_tables_full_name)
        .filter(f"{grid_id_column} = {safe_grid_id}")
        .toPandas()
    )
    if "index" in grid_tables_pdf.columns:
        grid_tables_pdf = grid_tables_pdf.set_index("index")
        grid_tables_pdf.index.name = None

    # Remove rows from every element table, then grid_tables itself
    for element in list(grid_tables_pdf["table"].values) + ["grid_tables"]:
        full_table_name = _qualify_table_name(catalog_name, schema_name, element)
        if _spark_table_exists(spark, full_table_name):
            _spark_delete_rows(spark, full_table_name, grid_id_column, safe_grid_id)

    # Remove entry from the catalogue
    _spark_delete_rows(spark, catalogue_full_name, grid_id_column, safe_grid_id)
