import numpy as np
import pandas as pd
import polars as pl

from .timer import time_this

# Function to extend the dataframe
@time_this
def extend_dataframe_pandas(df: pd.DataFrame, start: int, end: int):
    # Calculate how many times we need to replicate the original data
    n_replications = (end - start) // len(df) + 1

    # Replicate the dataframe
    extended_df = pd.concat([df] * n_replications, ignore_index=True)

    # Update the REAL column
    extended_df["REAL"] = np.tile(np.arange(len(df)), n_replications)
    extended_df["REAL"] += start

    # Trim the dataframe to the desired range
    extended_df = extended_df[extended_df["REAL"] <= end]

    return extended_df


def extend_dataframe_polars(df: pl.DataFrame, start: int, end: int) -> pl.DataFrame:
    # Calculate how many times we need to replicate the original data
    n_replications = (end - start) // len(df) + 1

    # Replicate the dataframe
    extended_df = pl.concat([df] * n_replications)

    # Update the REAL column
    real_values = np.tile(np.arange(len(df)), n_replications) + start
    extended_df = extended_df.with_columns(
        pl.Series("REAL", real_values[: len(extended_df)])
    )

    # Trim the dataframe to the desired range
    extended_df = extended_df.filter(pl.col("REAL") <= end)

    return extended_df
