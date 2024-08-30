import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from .polars_stat import get_aggregation_expressions
from .classes import IndexFilter


def calc_grouped_statistics_arrow_and_polars(
    arrow_df: pa.Table,
    index_filters: list[IndexFilter],
    groupby_cols: list[str],
    response_cols: list[str],
) -> pl.DataFrame:
    """Pyarrow implementation of grouped statistics calculation"""

    # Cleanup bad data
    arrow_df = arrow_df.drop(["GRID"])
    filter_condition = (
        (pc.field("ZONE") != "Totals")
        & (pc.field("REGION") != "Totals")
        & (pc.field("FACIES") != "Totals")
    )
    arrow_df = arrow_df.filter(filter_condition)

    # Filter on indexes
    mask = pa.array([True] * arrow_df.num_rows)

    for index_filter in index_filters:
        identifier_mask = pc.is_in(
            arrow_df[index_filter.name], value_set=pa.array(index_filter.values)
        )
        mask = pc.and_(mask, identifier_mask)
    arrow_df = arrow_df.filter(mask)

    # Perform a groupby and sum
    columns_to_group_by_for_sum = set(list(groupby_cols) + ["REAL"])
    accumulated_table = arrow_df.group_by(columns_to_group_by_for_sum).aggregate(
        [
            (volume_name, "sum")
            for volume_name in arrow_df.column_names
            if volume_name in get_numerical_column_names(arrow_df)
        ]
    )
    suffix_to_remove = "_sum"
    column_names_with_suffix = accumulated_table.column_names
    new_column_names = [
        column_name.replace(suffix_to_remove, "")
        for column_name in column_names_with_suffix
    ]
    accumulated_table = accumulated_table.rename_columns(new_column_names)

    # Calculate some properties
    if "SW_OIL" in response_cols and set(["HCPV_OIL", "PORV_OIL"]).issubset(
        accumulated_table.column_names
    ):
        sw_array = calculate_property_from_volume_arrays(
            "SW_OIL", accumulated_table["HCPV_OIL"], accumulated_table["PORV_OIL"]
        )
        accumulated_table = accumulated_table.append_column("SW_OIL", sw_array)
    if "PORO_OIL" in response_cols and set(["BULK_OIL", "PORV_OIL"]).issubset(
        accumulated_table.column_names
    ):
        poro_array = calculate_property_from_volume_arrays(
            "PORO_OIL", accumulated_table["PORV_OIL"], accumulated_table["BULK_OIL"]
        )
        accumulated_table = accumulated_table.append_column("PORO_OIL", poro_array)

    # Calculate statistics
    valid_result_names = [
        col for col in response_cols if col not in groupby_cols + ["REAL"]
    ]
    polars_df = pl.DataFrame(accumulated_table)
    agg_expressions = get_aggregation_expressions(valid_result_names, True)
    per_group_stats = (
        polars_df.select(*groupby_cols, *response_cols)
        .group_by(groupby_cols)
        .agg(agg_expressions)
    )
    statistical_table = per_group_stats.to_arrow()
    return statistical_table


def get_numerical_column_names(table):
    numerical_types = (
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
        pa.uint64(),
        pa.float16(),
        pa.float32(),
        pa.float64(),
    )

    return [
        field.name
        for field in table.schema
        if pa.types.is_primitive(field.type) and field.type in numerical_types
    ]


def calculate_property_from_volume_arrays(
    property: str, nominator: pa.array, denominator: pa.array
) -> pa.array:
    """
    Calculate property from two arrays of volumes

    Assume equal length and dimension of arrays

    """
    safe_denominator = _create_safe_denominator_array(denominator)

    result = None
    if property == "PORO_OIL":
        result = pc.divide(nominator, safe_denominator)
    if property == "SW_OIL":
        result = pc.subtract(1, pc.divide(nominator, safe_denominator))

    if result is not None:
        # return result
        return _replace_nan_and_inf_with_null(result)

    ValueError(f"Unhandled property: {property}")


def _create_safe_denominator_array(denominator_array: pa.array) -> pa.array:
    """
    Create denominator array for safe division, i.e. replace 0 with np.nan
    """
    zero_mask = pc.equal(denominator_array, 0.0)
    safe_denominator_array = pc.if_else(zero_mask, float("nan"), denominator_array)
    return safe_denominator_array


def _replace_nan_and_inf_with_null(array: pa.array) -> pa.array:
    """
    Replace NaN and Inf with null, this is needed for pyarrow to handle null values in aggregation

    The None value is used to represent null values in pyarrow array

    NOTE: if pyarrow is removed, this replacement is probably not needed
    """
    nan_or_inf_mask = pc.or_(pc.is_nan(array), pc.is_inf(array))
    return pc.if_else(nan_or_inf_mask, None, array)
