import polars as pl
from .classes import IndexFilter


def calc_grouped_statistics_polars(
    polars_df: pl.DataFrame,
    index_filters: list[IndexFilter],
    groupby_cols: list[str],
    response_cols: list[str],
    drop_nans: bool = True,
) -> pl.DataFrame:
    """Polars implementation of grouped statistics calculation"""

    # Cleanup bad data
    polars_df = polars_df.drop("GRID").filter(
        (polars_df["ZONE"] != "Totals")
        & (polars_df["REGION"] != "Totals")
        & (polars_df["FACIES"] != "Totals")
    )

    # Filter on indexes
    filters = []
    for index_filter in index_filters:
        filters.append(pl.col(index_filter.name).is_in(index_filter.values))
    if filters:
        polars_df = polars_df.filter(pl.all_horizontal(filters))

    # Perform a groupby and sum
    per_group_with_real = ["REAL"] + groupby_cols
    per_group_summed = polars_df.group_by(per_group_with_real).agg(
        [pl.sum("*").exclude(per_group_with_real)]
    )

    # Calculate some properties
    calculated_columns = get_calculated_properties_expression(
        polars_df.columns, response_cols
    )
    if calculated_columns:
        per_group_summed = per_group_summed.with_columns(calculated_columns)

    # Define aggregation expressions
    agg_expressions = get_aggregation_expressions(response_cols, drop_nans)

    # Perform the groupby and aggregation
    per_group_stats = (
        per_group_summed.select(*groupby_cols, *response_cols)
        .group_by(groupby_cols)
        .agg(agg_expressions)
    )

    return per_group_stats



def get_calculated_properties_expression(df_cols: list[str], response_cols: list[str]):
    """Return the calculated properties expressions."""
    calculated_columns = []
    if "HCPV_OIL" in df_cols and "PORV_OIL" in df_cols and "SW_OIL" in response_cols:
        calculated_columns.append(
            (1 - (pl.col("HCPV_OIL") / pl.col("PORV_OIL"))).alias("SW_OIL")
        )
    if "PORV_OIL" in df_cols and "BULK_OIL" in df_cols and "PORO_OIL" in response_cols:
        calculated_columns.append(
            (pl.col("PORV_OIL") / pl.col("BULK_OIL")).alias("PORO_OIL")
        )
    return calculated_columns

def get_aggregation_expressions(response_cols: list[str], drop_nans: bool = True):
    """Generate the aggregation expressions for the selected statistics."""
    agg_expressions = []
    for col in response_cols:
        for stat in ["mean", "stddev", "p10", "p90"]:
            agg_expressions.append(get_expr(stat, col, drop_nans))
    return agg_expressions

def get_stat_func(stat: str):
    """Return the appropriate statistical function based on the stat type."""
    return {
        "mean": lambda col: col.mean(),
        "median": lambda col: col.median(),
        "sum": lambda col: col.sum(),
        "min": lambda col: col.min(),
        "max": lambda col: col.max(),
        "stddev": lambda col: col.std(),
        "var": lambda col: col.var(),
        "p10": lambda col: col.quantile(0.1, "linear"),
        "p90": lambda col: col.quantile(0.9, "linear"),
    }.get(stat)


def get_expr(stat: str, col: str, drop_nans: bool = True):
    """Generate the Polars expression for the given statistic."""
    base_col = pl.col(col)
    if drop_nans:
        base_col = base_col.drop_nans()
    stat_func = get_stat_func(stat)
    if stat_func is None:
        raise ValueError(f"Unsupported statistic: {stat}")
    return stat_func(base_col).alias(f"{col}_{stat}")
