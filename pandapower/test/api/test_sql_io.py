# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import json
import os
import copy

import pandas as pd
import pandas.testing as pdt
import numpy as np
import pytest
import time

from pandapower import reset_results, runpp, pp_dir
from pandapower.networks import case9, case14, case39, simple_mv_open_ring_net, create_cigre_network_hv, mv_oberrhein
from pandapower.plotting.geo import convert_geodata_to_geojson
from pandapower.auxiliary import _preserve_dtypes
from pandapower.sql_io import download_sql_table, to_postgresql, from_postgresql, delete_postgresql_net
from pandapower.test import assert_res_equal

try:
    import psycopg2
    import psycopg2.errors

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


@pytest.fixture(params=[case9, case14, case39, simple_mv_open_ring_net,
                        create_cigre_network_hv, mv_oberrhein])
def net_in(request):
    net = request.param()
    # net.line.loc[0, "geo"] = '{"coordinates": [[1.1, 2.2], [3.3, 4.4]], "type": "LineString"}'
    # net.line.loc[11, "geo"] = '{"coordinates": [[5.5, 5.5], [6.6, 6.6], [7.7, 7.7]], "type": "LineString"}'
    # if len(net.trafo) > 0:
    #     net.trafo.tap_side = "lv"
    #     pp.control.DiscreteTapControl(net, net.trafo.index.values[0], 0.98, 1.02)
    return net


def get_postgresql_connection_data():
    filename = os.path.join(pp_dir, "test", "test_files", "postgresql_connect_data.json")
    if not os.path.isfile(filename):
        return {}, None
    with open(filename) as fp:
        connect_data = json.load(fp)
        schema = connect_data.pop("schema")

    return connect_data, schema


def postgresql_listening(**connect_data):
    if len(connect_data) == 0:
        return False
    try:
        conn = psycopg2.connect(**connect_data)
        conn.close()
        return True
    except psycopg2.OperationalError as ex:
        return False


def assert_postgresql_roundtrip(net_in, **kwargs):
    net = copy.deepcopy(net_in)
    if hasattr(net, "bus_geodata") or hasattr(net, "line_geodata"):
        convert_geodata_to_geojson(net)
    include_results = kwargs.pop("include_results", False)
    if not include_results:
        reset_results(net)
    else:
        runpp(net)
    connection_data, schema = get_postgresql_connection_data()
    grid_id = to_postgresql(net, schema=schema, include_results=include_results, **connection_data, **kwargs)

    net_out = from_postgresql(grid_id=grid_id, schema=schema, **connection_data, **kwargs)

    if not include_results:
        runpp(net)
        runpp(net_out)

    assert_res_equal(net, net_out)

    for element, table in net.items():
        # dictionaries (e.g. std_type) not included
        # json serialization/deserialization of objects not implemented
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        # code below: very difficult to compare columns with NaN values due to None vs np.nan and dtypes,
        # "1" vs 1 and dtype object
        # also sometimes order of rows is not same
        columns = table.columns
        table_in = table.fillna(np.nan)
        table_out = net_out[element][columns].loc[table_in.index].fillna(np.nan)
        _preserve_dtypes(table_out, table_in.dtypes)
        pdt.assert_frame_equal(table_in, table_out, check_dtype=False)

    # clean-up
    delete_postgresql_net(grid_id=grid_id, schema=schema, **connection_data)


POSTGRESQL_AVAILABLE = PSYCOPG2_INSTALLED and postgresql_listening(**get_postgresql_connection_data()[0])


@pytest.mark.skipif(not POSTGRESQL_AVAILABLE,
                    reason="testing happens on GitHub Actions where we create a temporary instance of PostgreSQL")
def test_postgresql(net_in):
    assert_postgresql_roundtrip(net_in, include_results=False)
    assert_postgresql_roundtrip(net_in, include_results=True)


@pytest.mark.skipif(not POSTGRESQL_AVAILABLE,
                    reason="testing happens on GitHub Actions where we create a temporary instance of PostgreSQL")
def test_unique():
    net = case9()
    connection_data, schema = get_postgresql_connection_data()
    grid_id = to_postgresql(net, **connection_data, schema=schema)
    with pytest.raises(UserWarning):
        to_postgresql(net, **connection_data, schema=schema, grid_id=grid_id)
    # clean-up:
    delete_postgresql_net(grid_id=grid_id, schema=schema, **connection_data)


@pytest.mark.skipif(not POSTGRESQL_AVAILABLE,
                    reason="testing happens on GitHub Actions where we create a temporary instance of PostgreSQL")
def test_delete():
    connection_data, schema = get_postgresql_connection_data()
    # cannot delete if the net does not exist
    with pytest.raises(UserWarning):
        delete_postgresql_net(grid_id=int(time.time()), schema=schema, **connection_data)

    # check that net is deleted
    net = case9()
    grid_id = to_postgresql(net, **connection_data, schema=schema)
    delete_postgresql_net(grid_id=grid_id, schema=schema, **connection_data)
    with pytest.raises(UserWarning):
        _ = from_postgresql(grid_id=grid_id, schema=schema, **connection_data)

    # check that it is not only deleted from the grid catalogue
    conn = psycopg2.connect(**connection_data)
    cursor = conn.cursor()
    for element in ("bus", "line", "load", "ext_grid", "gen"):
        tab = download_sql_table(cursor, f"{schema}.{element}", grid_id=grid_id)
        assert tab.empty


# ============================================================================
# Spark SQL I/O tests
# ============================================================================

try:
    from pyspark.sql import SparkSession as _SparkSession
    from pyspark.sql.types import (
        LongType as _LongType,
        DoubleType as _DoubleType,
        BooleanType as _BooleanType,
        StringType as _StringType,
        TimestampType as _TimestampType,
        StructType as _StructType,
        StructField as _StructField,
    )
    PYSPARK_INSTALLED = True
except ImportError:
    _SparkSession = None  # type: ignore[assignment]
    PYSPARK_INSTALLED = False

from pandapower.sql_io import (
    to_spark, from_spark, delete_spark_net,
    _qualify_table_name, _pandas_dtype_to_spark_type,
    _build_spark_schema, _coerce_pdf_to_spark_schema,
    _spark_table_exists, _check_spark_catalogue,
)


def test_qualify_table_name():
    """_qualify_table_name constructs fully-qualified Spark table names."""
    assert _qualify_table_name("cat", "sch", "tbl") == "cat.sch.tbl"
    assert _qualify_table_name(None, "sch", "tbl") == "sch.tbl"
    assert _qualify_table_name("", "sch", "tbl") == "sch.tbl"  # empty string is falsy


def test_to_spark_raises_without_pyspark(monkeypatch):
    import pandapower.sql_io as _sql_io
    monkeypatch.setattr(_sql_io, "PYSPARK_INSTALLED", False)
    with pytest.raises(UserWarning, match="install pyspark"):
        _sql_io.to_spark(None, None, "test_schema")


def test_from_spark_raises_without_pyspark(monkeypatch):
    import pandapower.sql_io as _sql_io
    monkeypatch.setattr(_sql_io, "PYSPARK_INSTALLED", False)
    with pytest.raises(UserWarning, match="install pyspark"):
        _sql_io.from_spark(None, "test_schema", grid_id=1)


def test_delete_spark_raises_without_pyspark(monkeypatch):
    import pandapower.sql_io as _sql_io
    monkeypatch.setattr(_sql_io, "PYSPARK_INSTALLED", False)
    with pytest.raises(UserWarning, match="install pyspark"):
        _sql_io.delete_spark_net(None, "test_schema", grid_id=1)


@pytest.mark.skipif(not PYSPARK_INSTALLED, reason="pyspark not installed")
def test_pandas_dtype_to_spark_type():
    assert isinstance(_pandas_dtype_to_spark_type("int64"), _LongType)
    assert isinstance(_pandas_dtype_to_spark_type("int32"), _LongType)
    assert isinstance(_pandas_dtype_to_spark_type("float64"), _DoubleType)
    assert isinstance(_pandas_dtype_to_spark_type("float32"), _DoubleType)
    assert isinstance(_pandas_dtype_to_spark_type("bool"), _BooleanType)
    assert isinstance(_pandas_dtype_to_spark_type("datetime64[ns]"), _TimestampType)
    assert isinstance(_pandas_dtype_to_spark_type("object"), _StringType)


@pytest.mark.skipif(not PYSPARK_INSTALLED, reason="pyspark not installed")
def test_build_spark_schema():
    df = pd.DataFrame({
        "name": pd.Series(["a", "b"], dtype="object"),
        "value": pd.Series([1.0, 2.0], dtype="float64"),
        "count": pd.Series([1, 2], dtype="int64"),
        "flag": pd.Series([True, False], dtype="bool"),
    })
    schema = _build_spark_schema(df)
    field_map = {f.name: type(f.dataType) for f in schema.fields}
    # reset_index() adds "index" as the first field
    assert field_map["index"] == _LongType
    assert field_map["name"] == _StringType
    assert field_map["value"] == _DoubleType
    assert field_map["count"] == _LongType
    assert field_map["flag"] == _BooleanType


@pytest.mark.skipif(not PYSPARK_INSTALLED, reason="pyspark not installed")
def test_coerce_pdf_to_spark_schema():
    schema = _StructType([
        _StructField("index", _LongType(), nullable=True),
        _StructField("name", _StringType(), nullable=True),
        _StructField("value", _DoubleType(), nullable=True),
        _StructField("flag", _BooleanType(), nullable=True),
    ])
    df = pd.DataFrame({
        "index": [0, 1],
        "name": pd.Series(["foo", None], dtype="object"),
        "value": pd.Series([1.5, None], dtype="object"),
        "flag": pd.Series([True, None], dtype="object"),
    })
    coerced = _coerce_pdf_to_spark_schema(df, schema)
    assert coerced["name"].iloc[0] == "foo"
    assert coerced["name"].iloc[1] is None
    assert coerced["value"].iloc[0] == 1.5
    assert coerced["value"].iloc[1] is None
    assert coerced["flag"].iloc[0] is True
    assert coerced["flag"].iloc[1] is None


@pytest.fixture(scope="module")
def spark_local(tmp_path_factory):
    if not PYSPARK_INSTALLED:
        pytest.skip("pyspark not installed")
    warehouse_dir = str(tmp_path_factory.mktemp("spark_warehouse"))
    spark = (
        _SparkSession.builder
        .master("local")
        .appName("pandapower_test")
        .config("spark.sql.warehouse.dir", warehouse_dir)
        .config("spark.driver.memory", "512m")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.mark.skipif(not PYSPARK_INSTALLED, reason="pyspark not installed")
def test_spark_roundtrip(spark_local):
    """Full to_spark / from_spark roundtrip: grid_id returned, net loaded correctly."""
    net = case9()
    grid_id = to_spark(net, spark_local, "pp_roundtrip")
    assert isinstance(grid_id, int)
    net_out = from_spark(spark_local, "pp_roundtrip", grid_id=grid_id)
    assert len(net_out.bus) == len(net.bus)
    assert len(net_out.line) == len(net.line)
    assert len(net_out.load) == len(net.load)
    assert len(net_out.gen) == len(net.gen)
    assert len(net_out.ext_grid) == len(net.ext_grid)
    # clean-up
    delete_spark_net(spark_local, "pp_roundtrip", grid_id=grid_id)


@pytest.mark.skipif(not PYSPARK_INSTALLED, reason="pyspark not installed")
def test_spark_auto_increment_grid_id(spark_local):
    """grid_id should auto-increment when not specified."""
    net = case9()
    gid1 = to_spark(net, spark_local, "pp_autoincrement")
    gid2 = to_spark(net, spark_local, "pp_autoincrement")
    assert gid2 == gid1 + 1
    # clean-up
    delete_spark_net(spark_local, "pp_autoincrement", grid_id=gid1)
    delete_spark_net(spark_local, "pp_autoincrement", grid_id=gid2)


@pytest.mark.skipif(not PYSPARK_INSTALLED, reason="pyspark not installed")
def test_spark_unique(spark_local):
    """Uploading two nets with the same explicit grid_id must raise UserWarning."""
    net = case9()
    grid_id = to_spark(net, spark_local, "pp_unique")
    with pytest.raises(UserWarning):
        to_spark(net, spark_local, "pp_unique", grid_id=grid_id)
    # clean-up
    delete_spark_net(spark_local, "pp_unique", grid_id=grid_id)


@pytest.mark.skipif(not PYSPARK_INSTALLED, reason="pyspark not installed")
def test_spark_delete(spark_local):
    """delete_spark_net removes data; subsequent from_spark raises UserWarning."""
    net = case9()
    grid_id = to_spark(net, spark_local, "pp_delete")
    delete_spark_net(spark_local, "pp_delete", grid_id=grid_id)
    with pytest.raises(UserWarning):
        from_spark(spark_local, "pp_delete", grid_id=grid_id)


@pytest.mark.skipif(not PYSPARK_INSTALLED, reason="pyspark not installed")
def test_spark_delete_nonexistent(spark_local):
    """Deleting a non-existent grid_id raises UserWarning."""
    with pytest.raises(UserWarning):
        delete_spark_net(spark_local, "pp_del_nonexistent", grid_id=99999)


@pytest.mark.skipif(not PYSPARK_INSTALLED, reason="pyspark not installed")
def test_spark_multiple_grids(spark_local):
    """Two different nets can be stored in the same schema and loaded independently."""
    net1 = case9()
    net2 = case14()
    gid1 = to_spark(net1, spark_local, "pp_multi")
    gid2 = to_spark(net2, spark_local, "pp_multi")
    assert gid1 != gid2

    out1 = from_spark(spark_local, "pp_multi", grid_id=gid1)
    out2 = from_spark(spark_local, "pp_multi", grid_id=gid2)

    assert len(out1.bus) == len(net1.bus)
    assert len(out2.bus) == len(net2.bus)
    assert len(out1.bus) != len(out2.bus)  # case9 vs case14 have different bus counts
    # clean-up
    delete_spark_net(spark_local, "pp_multi", grid_id=gid1)
    delete_spark_net(spark_local, "pp_multi", grid_id=gid2)


@pytest.mark.skipif(not PYSPARK_INSTALLED, reason="pyspark not installed")
def test_spark_custom_catalogue(spark_local):
    """Custom grid_id_column and grid_catalogue_name parameters work end-to-end."""
    net = case9()
    grid_id = to_spark(
        net, spark_local, "pp_custom_cat",
        grid_id_column="net_id",
        grid_catalogue_name="net_registry",
    )
    assert isinstance(grid_id, int)

    # catalogue table must exist under the custom name
    full_cat = _qualify_table_name(None, "pp_custom_cat", "net_registry")
    assert _spark_table_exists(spark_local, full_cat)

    net_out = from_spark(
        spark_local, "pp_custom_cat", grid_id=grid_id,
        grid_id_column="net_id",
        grid_catalogue_name="net_registry",
    )
    assert len(net_out.bus) == len(net.bus)

    # verify _check_spark_catalogue raises for missing grid_id
    with pytest.raises(UserWarning):
        _check_spark_catalogue(spark_local, full_cat, grid_id=99999,
                               grid_id_column="net_id", download=True)

    # clean-up
    delete_spark_net(
        spark_local, "pp_custom_cat", grid_id=grid_id,
        grid_id_column="net_id",
        grid_catalogue_name="net_registry",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
