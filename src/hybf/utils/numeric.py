"""/hybf/src/hybf/utils/numeric.py
Utilities for numeric data types
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, BinaryIO

from hybf.core.base import BinaryReader

def analyze_numeric_column(series: pd.Series) -> Tuple[Optional[np.dtype], bool]:
    """
    Analyze a pandas Series to determine optimal numeric storage type.
    
    Args:
        series: pandas Series to analyze
        
    Returns:
        Tuple of (optimal dtype or None if not numeric, has_nulls)
    """
    # Handle empty series
    if len(series) == 0:
        return None, False
        
    # Get non-null values
    non_null = series.dropna()
    if len(non_null) == 0:
        return None, True
        
    # Check if all non-null values are numeric
    try:
        numeric_values = pd.to_numeric(non_null)
    except (ValueError, TypeError):
        return None, series.isna().any()
        
    # Determine if values are integers
    is_integer = np.all(np.equal(np.mod(numeric_values, 1), 0))
    
    if is_integer:
        # Find min/max to determine optimal integer type
        min_val = numeric_values.min()
        max_val = numeric_values.max()
        
        # Check ranges for different integer types
        if min_val >= -128 and max_val <= 127:
            return np.dtype('int8'), series.isna().any()
        elif min_val >= 0 and max_val <= 255:
            return np.dtype('uint8'), series.isna().any()
        elif min_val >= -32768 and max_val <= 32767:
            return np.dtype('int16'), series.isna().any()
        elif min_val >= 0 and max_val <= 65535:
            return np.dtype('uint16'), series.isna().any()
        elif min_val >= -2147483648 and max_val <= 2147483647:
            return np.dtype('int32'), series.isna().any()
        elif min_val >= 0 and max_val <= 4294967295:
            return np.dtype('uint32'), series.isna().any()
        else:
            return np.dtype('int64'), series.isna().any()
    else:
        # For floating point, use float32 if precision allows
        float32_values = numeric_values.astype('float32')
        if np.allclose(numeric_values, float32_values):
            return np.dtype('float32'), series.isna().any()
        else:
            return np.dtype('float64'), series.isna().any()

def write_numeric_column(buffer: BinaryIO, series: pd.Series, dtype: np.dtype, has_nulls: bool) -> None:
    """
    Write numeric column data with optional null value handling.
    
    Args:
        buffer: Binary buffer to write to
        series: Series containing the data
        dtype: numpy dtype to use for storage
        has_nulls: Whether the series contains null values
    """
    if has_nulls:
        # Write a null bitmap first
        null_bitmap = ~series.isna()
        buffer.write(null_bitmap.to_numpy().tobytes('C'))
        
        # Write non-null values
        non_null_values = series.dropna().astype(dtype)
        buffer.write(non_null_values.to_numpy().tobytes('C'))
    else:
        # Write all values directly
        buffer.write(series.astype(dtype).to_numpy().tobytes('C'))

def read_numeric_column(buffer: BinaryIO, dtype: np.dtype, row_count: int, has_nulls: bool) -> np.ndarray:
    """
    Read numeric column data with optional null value handling.
    
    Args:
        buffer: Binary buffer to read from
        dtype: numpy dtype to use for reading
        row_count: Number of rows to read
        has_nulls: Whether the column contains null values
        
    Returns:
        numpy array containing the column data
    """
    reader = BinaryReader(buffer)
    if has_nulls:
        # Read null bitmap
        bitmap_size = (row_count + 7) // 8
        null_bitmap = reader.read_bytes(bitmap_size)

        #null_bitmap = np.fromfile(buffer, dtype=np.bool_, count=row_count)
        non_null_count = np.count_nonzero(null_bitmap)
        
        # Read non-null values
        # values = np.fromfile(buffer, dtype=dtype, count=non_null_count)
        values = reader.read_array(dtype=dtype, count=non_null_count)

        # Create result array with nulls
        result = np.full(row_count, np.nan, dtype=dtype)
        result[null_bitmap] = values
        return result
    else:
        return reader.read_array(dtype=dtype, count=row_count)
