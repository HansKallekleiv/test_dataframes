import asyncio
import pandas as pd
from sumo.wrapper import SumoClient
from fmu.sumo.explorer.objects import CaseCollection, Table
import pyarrow as pa
import pyarrow.compute as pc

from .timer import time_this, timing_data


## Sumo helpers
@time_this
def get_sumo_client(env: str = "prod"):
    return SumoClient(env=env)


@time_this
def get_sumo_tables(
    client: SumoClient,
    case_uuid: str,
    table_name: str,
    iteration_name: str,
    as_pandas: bool,
):
    if as_pandas:
        return asyncio.run(
            get_sumo_tables_pandas_async(client, case_uuid, table_name, iteration_name)
        )
    else:
        return asyncio.run(
            get_sumo_tables_arrow_async(client, case_uuid, table_name, iteration_name)
        )


async def get_sumo_tables_pandas_async(
    client: SumoClient, case_uuid: str, table_name: str, iteration_name: str
):
    """Fetch all table columns from Sumo"""
    case = CaseCollection(sumo=client).filter(uuid=case_uuid)[0]
    vol_table_collection = case.tables.filter(
        aggregation="collection",
        tagname=["vol", "volumes", "inplace"],
        iteration=iteration_name,
        name=table_name,
    )

    async def fetch_table_pandas(table: Table) -> pd.DataFrame:
        df = await table.to_pandas_async()
        return df

    dfs_duped = await asyncio.gather(
        *[fetch_table_pandas(table) for table in vol_table_collection]
    )
    dfs = []
    for df in dfs_duped:
        index_cols = ["REAL", "ZONE", "REGION", "FACIES"]
        df.set_index(index_cols, inplace=True)
        result_col = [col for col in df.columns if col not in index_cols][0]
        df = df[[result_col]]
        dfs.append(df)

    df = pd.concat(dfs, axis=1)
    df.reset_index(inplace=True)
    return df


async def get_sumo_tables_arrow_async(
    client: SumoClient, case_uuid: str, table_name: str, iteration_name: str
):
    """Fetch all table columns from Sumo"""
    case = CaseCollection(sumo=client).filter(uuid=case_uuid)[0]
    vol_table_collection = case.tables.filter(
        aggregation="collection",
        tagname=["vol", "volumes", "inplace"],
        iteration=iteration_name,
        name=table_name,
    )

    async def fetch_table_arrow(table: Table) -> pa.Table:
        # Fetch the table as an Arrow Table
        arrow_table = await table.to_arrow_async()
        return arrow_table

    # Fetch all Arrow tables concurrently
    arrow_tables = await asyncio.gather(
        *[fetch_table_arrow(table) for table in vol_table_collection]
    )

    index_cols = ["REAL", "ZONE", "REGION", "FACIES"]
    combined_table = arrow_tables[0]
    for i in range(1, len(arrow_tables)):
        volume_table: pa.Table = arrow_tables[i]
        # Keep only the index columns and the first result column
        volume_name = [
            col for col in volume_table.column_names if col not in index_cols
        ][0]
        volume_column = volume_table[volume_name]
        combined_table = combined_table.append_column(volume_name, volume_column)

    # Return the final combined Arrow table
    return combined_table
