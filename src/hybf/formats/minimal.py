# src/hybf/formats/minimal.py
"""Reader and Writer for the minimal HYBF format.

The minimal format is optimized for small files (<4KB) with low overhead.
It stores data in a simple column-major format with minimal metadata.
"""

import struct
import pandas as pd
import numpy as np
from typing import List, BinaryIO

from hybf import BaseWriter, BaseReader, BinaryReader
from hybf.core.types import (
    LogicalType, 
    StorageType,
    ColumnTypeInfo,
    TypeAnalyzer,
    TypeConverter
)
from hybf.core.dtypes import FormatType
from hybf.constants import MAGIC_NUMBER, VERSION

class MinimalWriter(BaseWriter):
    """Writer implementation for the minimal format."""
    
    def write(self, df: pd.DataFrame, file: BinaryIO) -> None:
        """
        Write DataFrame to file in minimal format.
        
        Args:
            df: DataFrame to write
            file: Binary file object to write to
        """
        # Analyze column types
        columns = [
            TypeAnalyzer.analyze_series(df[name])
            for name in df.columns
        ]
        
        # Write header
        self._write_header(file, len(columns))
        
        # Write column definitions
        self._write_column_definitions(file, columns)
        
        # Write row count
        file.write(struct.pack('>I', len(df)))
        
        # Write data in column-major format
        for col_info, name in zip(columns, df.columns):
            self._write_column(file, df[name], col_info)
    
    def _write_header(self, file: BinaryIO, num_columns: int) -> None:
        """Write file header with format information."""
        self.write_header(file, FormatType.MINIMAL, num_columns)
    
    def _write_column_definitions(self, file: BinaryIO, columns: List[ColumnTypeInfo]) -> None:
        """Write column metadata."""
        for col in columns:
            # Write logical type
            file.write(struct.pack('B', col.logical_type.value))
            # Write column name
            name_bytes = col.name.encode('utf-8')
            file.write(struct.pack('B', len(name_bytes)))
            file.write(name_bytes)
            # Write nullability
            file.write(struct.pack('?', col.nullable))
    
    def _write_column(self, file: BinaryIO, series: pd.Series, col_info: ColumnTypeInfo) -> None:
        """Write a single column of data."""
        data = series.to_numpy()
        
        if col_info.logical_type == LogicalType.STRING:
            self._write_string_column(file, data, col_info.nullable)
        else:
            self._write_numeric_column(file, data, col_info)
    
    def _write_string_column(self, file: BinaryIO, data: np.ndarray, nullable: bool) -> None:
        """Write string column data."""
        for val in data:
            if pd.isna(val):
                file.write(struct.pack('B', 0))  # Zero length indicates null
            else:
                val_bytes = str(val).encode('utf-8')
                file.write(struct.pack('B', len(val_bytes)))
                file.write(val_bytes)
    
    def _write_numeric_column(self, file: BinaryIO, data: np.ndarray, col_info: ColumnTypeInfo) -> None:
        """Write numeric column data."""
        # Convert to storage type
        storage_data = TypeConverter.to_storage_type(data, col_info)
        
        if col_info.nullable:
            # Write null bitmap
            null_mask = pd.isna(data)
            bitmap = bytearray((len(data) + 7) // 8)
            for i, is_null in enumerate(null_mask):
                if is_null:
                    bitmap[i // 8] |= (1 << (i % 8))
            file.write(bitmap)
            
            # Write non-null values
            non_null_data = storage_data[~null_mask]
            file.write(non_null_data.tobytes())
        else:
            # Write all values directly
            file.write(storage_data.tobytes())

class MinimalReader(BaseReader):
    """Reader implementation for the minimal format."""
    
    def read(self, file: BinaryIO) -> pd.DataFrame:
        """
        Read DataFrame from file in minimal format.
        
        Args:
            file: Binary file object to read from
            
        Returns:
            pandas DataFrame containing the data
        """
        # Read and validate header
        version, format_type, num_columns = self._read_header(file)
        if format_type != FormatType.MINIMAL:
            raise ValueError("Not a minimal format file")
        
        # Read column definitions
        columns = self._read_column_definitions(file, num_columns)
        
        # Read row count
        row_count = struct.unpack('>I', file.read(4))[0]
        
        # Read data
        data = {}
        for col in columns:
            data[col.name] = self._read_column(file, col, row_count)
            
        return pd.DataFrame(data)
    
    def _read_header(self, file: BinaryIO) -> tuple[int, FormatType, int]:
        """Read and validate file header."""
        return self.read_header(file)
    
    def _read_column_definitions(self, file: BinaryIO, num_columns: int) -> List[ColumnTypeInfo]:
        """Read column metadata."""
        columns = []
        for _ in range(num_columns):
            # Read logical type
            logical_type = LogicalType(struct.unpack('B', file.read(1))[0])
            # Read column name
            name_length = struct.unpack('B', file.read(1))[0]
            name = file.read(name_length).decode('utf-8')
            # Read nullability
            nullable = struct.unpack('?', file.read(1))[0]
            
            # Create column info (storage type will be determined during reading)
            columns.append(ColumnTypeInfo(name, logical_type, StorageType.STRING, nullable))
        return columns
    
    def _read_column(self, file: BinaryIO, col_info: ColumnTypeInfo, row_count: int) -> np.ndarray:
        """Read a single column of data."""
        if col_info.logical_type == LogicalType.STRING:
            return self._read_string_column(file, row_count)
        else:
            return self._read_numeric_column(file, col_info, row_count)
    
    def _read_string_column(self, file: BinaryIO, row_count: int) -> np.ndarray:
        """Read string column data."""
        values = []
        for _ in range(row_count):
            length = struct.unpack('B', file.read(1))[0]
            if length == 0:
                values.append(None)
            else:
                values.append(file.read(length).decode('utf-8'))
        return np.array(values, dtype='O')
    
    def _read_numeric_column(self, file: BinaryIO, col_info: ColumnTypeInfo, row_count: int) -> np.ndarray:
        """Read numeric column data."""
        reader = BinaryReader(file)
        
        if col_info.nullable:
            # Read null bitmap
            bitmap_size = (row_count + 7) // 8
            null_bitmap = file.read(bitmap_size)
            
            # Count non-null values
            non_null_count = row_count
            for i in range(row_count):
                if null_bitmap[i // 8] & (1 << (i % 8)):
                    non_null_count -= 1
            
            # Read non-null values
            dtype = col_info.logical_type.to_numpy()
            data = reader.read_array(dtype, non_null_count)
            
            # Create result array
            result = np.empty(row_count, dtype=dtype)
            curr_idx = 0
            
            for i in range(row_count):
                if null_bitmap[i // 8] & (1 << (i % 8)):
                    result[i] = np.nan if dtype.kind == 'f' else None
                else:
                    result[i] = data[curr_idx]
                    curr_idx += 1
            
            return result
        else:
            # Read all values directly
            return reader.read_array(col_info.logical_type.to_numpy(), row_count)