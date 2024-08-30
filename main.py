import polars as pl
import pyarrow as pa
import pandas as pd
from src.timer import timing_data, time_this, time_many
from src.classes import IndexFilter
from src.sumo_utils import get_sumo_client, get_sumo_tables

from src.pyarrow_stat import calc_grouped_statistics_arrow
from src.polars_stat import calc_grouped_statistics_polars
from src.pandas_stat import calc_grouped_statistics_pandas
from src.pyarrow_and_polars_stat import calc_grouped_statistics_arrow_and_polars


@time_many(50)
def grouped_statistics_with_pyarrow(
    arrow_table: pa.Table, index_filters: list, groupby_cols: list, response_cols: list
):
    calculated_arrow_table = calc_grouped_statistics_arrow(
        arrow_table,
        index_filters=index_filters,
        groupby_cols=groupby_cols,
        response_cols=response_cols,
    )
    return calculated_arrow_table


@time_many(50)
def grouped_statistics_with_polars(
    polars_table: pl.DataFrame,
    index_filters: list,
    groupby_cols: list,
    response_cols: list,
):

    calculated_polars_table = calc_grouped_statistics_polars(
        polars_table,
        index_filters=index_filters,
        groupby_cols=groupby_cols,
        response_cols=response_cols,
    )
    return calculated_polars_table


@time_many(50)
def grouped_statistics_with_pandas(
    pandas_table: pd.DataFrame,
    index_filters: list,
    groupby_cols: list,
    response_cols: list,
):
    calculated_pandas_table = calc_grouped_statistics_pandas(
        pandas_table,
        index_filters=index_filters,
        groupby_cols=groupby_cols,
        response_cols=response_cols,
    )
    return calculated_pandas_table

@time_many(50)
def grouped_statistics_with_arrow_and_polars(
    arrow_table: pa.Table,
    index_filters: list,
    groupby_cols: list,
    response_cols: list,
):
    calculated_polars_table = calc_grouped_statistics_arrow_and_polars(
        arrow_table,
        index_filters=index_filters,
        groupby_cols=groupby_cols,
        response_cols=response_cols,
    )
    return calculated_polars_table

@time_this
def polars_to_arrow(polars_table: pl.DataFrame):
    return polars_table.to_arrow()


@time_this
def arrow_to_polars(arrow_table: pa.Table):
    return pl.from_arrow(arrow_table)


@time_this
def pandas_to_polars(pandas_table: pd.DataFrame):
    return pl.from_pandas(pandas_table)


####################################################################################################
####################################################################################################

## CONFIG

if __name__ == "__main__":
    case_uuid = "64fdd320-59f2-4f67-8b36-8314ae7e9b87"
    table_name = "ff_a"
    iteration_name = "iter-0"
    index_filters = [
        IndexFilter("REGION", ["AvaldsnesE"]),
        IndexFilter("ZONE", ["Eiriksson_Fm_2.1", "Draupne_Fm_1"]),
    ]
    # index_filters = []
    groupby_cols = ["REGION", "ZONE"]
    response_cols = [ "STOIIP_OIL"]

    ## GET SUMO DATA
    client = get_sumo_client()
    arrow_table = get_sumo_tables(
        client, case_uuid, table_name, iteration_name, as_pandas=False
    )
    pandas_table = get_sumo_tables(
        client, case_uuid, table_name, iteration_name, as_pandas=True
    )
    polars_table = pl.from_arrow(arrow_table)

    ## CALCULATIONS
    aggr_pandas_df = grouped_statistics_with_pandas(
        pandas_table, index_filters, groupby_cols, response_cols
    )
    aggr_polars_df = grouped_statistics_with_polars(
        polars_table, index_filters, groupby_cols, response_cols
    )
    aggr_pyarrow_df = grouped_statistics_with_pyarrow(
        arrow_table, index_filters, groupby_cols, response_cols
    )
    aggr_pyarrow_and_polars_df = grouped_statistics_with_arrow_and_polars(
        arrow_table,  index_filters, groupby_cols, response_cols
    )
    # Print with polars
    pl.Config.set_float_precision(7)
    pl.Config.set_tbl_cols(20)
    print("POLARS")
    print(aggr_polars_df.sort(groupby_cols))
    print("PANDAS")
    print(pandas_to_polars(aggr_pandas_df).sort(groupby_cols))
    print("PYARROW")
    print(arrow_to_polars(aggr_pyarrow_df).sort(groupby_cols))

    # Timings
    # Test polars to arrow
    polars_to_arrow(aggr_polars_df)
    for func, time in timing_data.items():
        print(f"{time} - {func}")
