"""/hybf/src/hybf/formats/raw.py
"Raw" is a bit of a misnomer. These are bit-optimized columns.
There is some code duplication with /hybf/src/hybf/core/dtypes.py
read_optimized_numeric has it's own internal dtype. Need to fix.
"""

import struct
import numpy as np
import pandas as pd
from typing import Tuple, Optional, BinaryIO

from hybf.core.base import BinaryReader
from hybf.core.dtypes import DataType

def analyze_numeric_column(series: pd.Series) -> Tuple[Optional[np.dtype], bool]:
    """
    Analyze a pandas Series to determine the optimal numeric dtype.
    
    Args:
        series: pandas Series to analyze
        
    Returns:
        Tuple of (optimal_dtype, contains_null)
        If the series cannot be stored as numeric, returns (None, contains_null)
    """
    # Drop NA values for analysis
    non_null = series.dropna()
    contains_null = len(non_null) < len(series)
    
    if len(non_null) == 0:
        return None, True
        
    # Try to convert to numeric, catching errors
    try:
        # First check if all values are integers
        if all(isinstance(x, (int, np.integer)) for x in non_null):
            values = non_null.astype(np.int64)
            min_val, max_val = values.min(), values.max()
            
            # Find the smallest integer type that can hold the values
            if min_val >= 0:  # Unsigned types
                if max_val <= np.iinfo(np.uint8).max:
                    return np.dtype('uint8'), contains_null
                elif max_val <= np.iinfo(np.uint16).max:
                    return np.dtype('uint16'), contains_null
                elif max_val <= np.iinfo(np.uint32).max:
                    return np.dtype('uint32'), contains_null
            
            # Signed types
            if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                return np.dtype('int8'), contains_null
            elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                return np.dtype('int16'), contains_null
            elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                return np.dtype('int32'), contains_null
            return np.dtype('int64'), contains_null
            
        # Check for float values
        values = pd.to_numeric(non_null)
        if values.dtype == np.float64:
            # Check if float32 precision is sufficient
            float32_values = values.astype(np.float32)
            if np.allclose(values, float32_values, rtol=1e-6):
                return np.dtype('float32'), contains_null
            return np.dtype('float64'), contains_null
            
    except (ValueError, TypeError):
        # If conversion fails, the series contains non-numeric data
        pass
        
    return None, contains_null

def write_optimized_numeric(buffer: BinaryIO, series: pd.Series, optimal_dtype: np.dtype) -> None:
    """
    Write numeric data with optimal dtype and null handling.
    
    Args:
        buffer: Binary buffer to write to
        series: Series to write
        optimal_dtype: The optimal dtype determined by analyze_numeric_column
    """
    # Write the dtype code
    dtype_map = {
        'uint8': 1, 'uint16': 2, 'uint32': 3,
        'int8': 4, 'int16': 5, 'int32': 6, 'int64': 7,
        'float32': 8, 'float64': 9
    }
    buffer.write(struct.pack('B', dtype_map[optimal_dtype.name]))
    
    # Write null bitmap if needed
    null_mask = series.isna()
    if null_mask.any():
        bitmap = bytearray((len(series) + 7) // 8)
        for i, is_null in enumerate(null_mask):
            if is_null:
                bitmap[i // 8] |= (1 << (i % 8))
        buffer.write(bitmap)
    
    # Write non-null values with optimal dtype
    non_null_values = series.dropna().astype(optimal_dtype)
    buffer.write(non_null_values.tobytes())

class RawWriter:
    """Handles optimized writing of raw column data."""
    
    @staticmethod
    def write(buffer: BinaryIO, series: pd.Series) -> None:
        """Write column data with numeric optimization when possible."""
        if series.dtype != object:
            # For non-object types, write directly
            buffer.write(series.to_numpy().tobytes('C'))
            return
            
        # Try to optimize numeric storage
        optimal_dtype, contains_null = analyze_numeric_column(series)
        
        if optimal_dtype is not None:
            # Write format marker (1 for optimized numeric)
            buffer.write(struct.pack('B', 1))
            write_optimized_numeric(buffer, series, optimal_dtype)
        else:
            # Write format marker (0 for string)
            buffer.write(struct.pack('B', 0))
            # Write null bitmap
            null_bitmap = bytearray((len(series) + 7) // 8)
            for i, val in enumerate(series):
                if pd.isna(val):
                    null_bitmap[i // 8] |= (1 << (i % 8))
            buffer.write(null_bitmap)
            
            # Write non-null values
            for val in series:
                if not pd.isna(val):
                    val_bytes = str(val).encode('utf-8')
                    buffer.write(struct.pack('>H', len(val_bytes)))
                    buffer.write(val_bytes)

class RawReader:
    """Handles optimized reading of raw column data."""
    
    @staticmethod
    def read(buffer: BinaryIO, dtype: DataType, row_count: int) -> np.ndarray:
        """Read column data with numeric optimization support."""
        if dtype != DataType.STRING:
            reader = BinaryReader(buffer)
            return reader.read_array(dtype.to_numpy(), row_count)
            
        # Read format marker
        format_type = struct.unpack('B', buffer.read(1))[0]
        
        if format_type == 1:  # Optimized numeric
            return read_optimized_numeric(buffer, row_count)
        else:  # String data
            # Read null bitmap
            bitmap_size = (row_count + 7) // 8
            null_bitmap = buffer.read(bitmap_size)
            
            values = []
            for i in range(row_count):
                is_null = bool(null_bitmap[i // 8] & (1 << (i % 8)))
                if is_null:
                    values.append(None)
                else:
                    length = struct.unpack('>H', buffer.read(2))[0]
                    val = buffer.read(length).decode('utf-8')
                    values.append(val)
            
            return np.array(values, dtype='O')

def read_optimized_numeric(buffer: BinaryIO, row_count: int) -> np.ndarray:
    """
    Read numeric data with optimal dtype and null handling.
    
    Args:
        buffer: Binary buffer to read from
        row_count: Number of rows to read
        
    Returns:
        numpy array with the data
    """
    # Read dtype code
    dtype_map = {
        1: np.dtype('uint8'), 2: np.dtype('uint16'), 3: np.dtype('uint32'),
        4: np.dtype('int8'), 5: np.dtype('int16'), 6: np.dtype('int32'),
        7: np.dtype('int64'), 8: np.dtype('float32'), 9: np.dtype('float64')
    }
    dtype_code = struct.unpack('B', buffer.read(1))[0]
    dtype = dtype_map[dtype_code]
    
    # Create array for result
    result = np.empty(row_count, dtype=dtype)
    
    # Read null bitmap if present
    if dtype_code > 0:
        bitmap_size = (row_count + 7) // 8
        null_bitmap = buffer.read(bitmap_size)
        
        # Read non-null values
        non_null_count = row_count - sum(
            bin(byte).count('1') for byte in null_bitmap[:bitmap_size-1]
        ) - bin(null_bitmap[-1]).count('1')
        
        # Read the actual data
        data = np.frombuffer(buffer.read(non_null_count * dtype.itemsize), dtype=dtype)
        
        # Fill result array
        curr_idx = 0
        for i in range(row_count):
            is_null = bool(null_bitmap[i // 8] & (1 << (i % 8)))
            if is_null:
                result[i] = np.nan if dtype.kind == 'f' else None
            else:
                result[i] = data[curr_idx]
                curr_idx += 1
    
    return result