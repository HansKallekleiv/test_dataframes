import numpy as np
import pandas as pd

from .classes import IndexFilter


def calc_grouped_statistics_pandas(
    pandas_df: pd.DataFrame,
    index_filters: list[IndexFilter],
    groupby_cols: list[str],
    response_cols: list[str],
) -> pd.DataFrame:
    """Pandas implementation of the grouped mean calculation"""

    # Cleanup bad data
    pandas_df = pandas_df.drop(columns=["GRID"])[
        (pandas_df["ZONE"] != "Totals")
        & (pandas_df["REGION"] != "Totals")
        & (pandas_df["FACIES"] != "Totals")
    ]

    # Filter
    mask = pd.Series(True, index=pandas_df.index)
    for filter in index_filters:
        mask &= pandas_df[filter.name].isin(filter.values)
    pandas_df = pandas_df[mask]

    # Perform a groupby and sum
    per_group_with_real = ["REAL"] + groupby_cols
    per_group_summed = pandas_df.groupby(per_group_with_real).sum()

    # Calculate some properties
    if (
        "HCPV_OIL" in pandas_df.columns
        and "PORV_OIL" in pandas_df.columns
        and "SW_OIL" in response_cols
    ):
        per_group_summed["SW_OIL"] = 1 - (
            per_group_summed["HCPV_OIL"] / per_group_summed["PORV_OIL"]
        )
    if (
        "PORV_OIL" in pandas_df.columns
        and "BULK_OIL" in pandas_df.columns
        and "PORO_OIL" in response_cols
    ):
        per_group_summed["PORO_OIL"] = (
            per_group_summed["PORV_OIL"] / per_group_summed["BULK_OIL"]
        )

    numerical_columns = per_group_summed.select_dtypes(include=[np.number]).columns
    numerical_columns = [col for col in numerical_columns if col in response_cols]

    # Calculate statistics

    per_group_summed_mean = per_group_summed.groupby(groupby_cols)[
        numerical_columns
    ].agg([np.mean, np.std, p10, p90])

    # Combine multi-index columns
    per_group_summed_mean.reset_index(inplace=True)
    cols = []
    for col in per_group_summed_mean.columns:

        if col[1] == "":
            cols.append(col[0])
        else:
            cols.append("_".join(col))

    per_group_summed_mean.columns = cols
    return per_group_summed_mean


def p10(x):
    return np.quantile(x, 0.1)


def p90(x):
    return np.quantile(x, 0.9)
