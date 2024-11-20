from abc import ABC, abstractmethod
from enum import Enum, auto
import struct
import pandas as pd
import numpy as np
import io
from typing import Any, Dict, List, BinaryIO

from ..constants import MAGIC_NUMBER, VERSION
from .dtypes import DataType, ColumnInfo
from .enums import FormatType, CompressionType

class BaseWriter(ABC):
    """Abstract base class for format writers."""
    
    @abstractmethod
    def write(self, df: pd.DataFrame, file: BinaryIO) -> None:
        """Write dataframe to file."""
        pass
    
    def write_header(self, file: BinaryIO, format_type: FormatType, num_columns: int) -> None:
        """Write the common file header."""
        file.write(MAGIC_NUMBER)
        file.write(struct.pack('BB', VERSION, format_type.value))
        file.write(struct.pack('>H', num_columns))  # big-endian uint16
        
    def write_column_definitions(self, file: BinaryIO, columns: List[ColumnInfo]) -> None:
        """Write column metadata."""
        for col in columns:
            file.write(struct.pack('B', col.dtype.value))
            name_bytes = col.name.encode('utf-8')
            file.write(struct.pack('B', len(name_bytes)))
            file.write(name_bytes)

class BaseReader(ABC):
    """Abstract base class for format readers."""
    
    @abstractmethod
    def read(self, file: BinaryIO) -> pd.DataFrame:
        """Read dataframe from file."""
        pass
    
    def read_header(self, file: BinaryIO) -> tuple[int, FormatType, int]:
        """Read and validate file header."""
        magic = file.read(4)
        if magic != MAGIC_NUMBER:
            raise ValueError("Invalid file format")
        
        version, format_type = struct.unpack('BB', file.read(2))
        if version != VERSION:
            raise ValueError(f"Unsupported version: {version}")
            
        num_columns = struct.unpack('>H', file.read(2))[0]
        return version, FormatType(format_type), num_columns
    
    def read_column_definitions(self, file: BinaryIO, num_columns: int) -> List[ColumnInfo]:
        """Read column metadata."""
        columns = []
        for _ in range(num_columns):
            dtype_value = struct.unpack('B', file.read(1))[0]
            name_length = struct.unpack('B', file.read(1))[0]
            name = file.read(name_length).decode('utf-8')
            columns.append(ColumnInfo(name, DataType(dtype_value)))
        return columns

