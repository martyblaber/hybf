import struct
import pandas as pd
import numpy as np
from typing import BinaryIO

from hybf import BaseWriter, BaseReader, BinaryReader
from hybf import DataType, ColumnInfo
from hybf.core.dtypes import FormatType


class MinimalWriter(BaseWriter):
    """Writer for the minimal format."""
    
    def write(self, df: pd.DataFrame, file: BinaryIO) -> None:
        # Write header
        self.write_header(file, FormatType.MINIMAL, len(df.columns))
        
        # Write column definitions
        columns = [
            ColumnInfo(name, DataType.from_numpy(df[name].dtype))
            for name in df.columns
        ]
        self.write_column_definitions(file, columns)
        
        # Write row count
        file.write(struct.pack('>I', len(df)))
        
        # Write data in column-major format
        for col in columns:
            self._write_column(file, df[col.name], col.dtype)
    
    def _write_column(self, file: BinaryIO, series: pd.Series, dtype: DataType) -> None:
        """Write a single column of data."""
        if dtype == DataType.STRING:
            # Write strings as length-prefixed UTF-8
            for val in series:
                if pd.isna(val):
                    file.write(struct.pack('B', 0))
                else:
                    val_bytes = str(val).encode('utf-8')
                    file.write(struct.pack('B', len(val_bytes)))
                    file.write(val_bytes)
        else:
            # Write numeric data directly
            file.write(series.to_numpy().tobytes('C'))

class MinimalReader(BaseReader):
    """Reader for the minimal format."""
    
    def read(self, file: BinaryIO) -> pd.DataFrame:
        # Read header
        version, format_type, num_columns = self.read_header(file)
        if format_type != FormatType.MINIMAL:
            raise ValueError("Not a minimal format file")
            
        # Read column definitions
        columns = self.read_column_definitions(file, num_columns)
        
        # Read row count
        row_count = struct.unpack('>I', file.read(4))[0]
        
        # Read data
        data = {}
        for col in columns:
            data[col.name] = self._read_column(file, col.dtype, row_count)
            
        return pd.DataFrame(data)
    
    def _read_column(self, file: BinaryIO, dtype: DataType, row_count: int) -> np.ndarray:
        """Read a single column of data."""
        reader = BinaryReader(file)
        
        if dtype == DataType.STRING:
            # Read strings as length-prefixed UTF-8
            values = []
            for _ in range(row_count):
                length = struct.unpack('B', file.read(1))[0]
                if length == 0:
                    values.append(None)
                else:
                    values.append(file.read(length).decode('utf-8'))
            return np.array(values, dtype='O')
        else:
            # Read numeric data using our helper
            return reader.read_array(dtype.to_numpy(), row_count)