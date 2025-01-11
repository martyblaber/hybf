"""/hybf/src/hybf/core/base.py
Abstract classes for readers/writers.
Adds the MAGIC_NUMBER and VERSION to the beginning of the file.
"""

from abc import ABC, abstractmethod
import struct
import pandas as pd
import numpy as np
import io
from typing import List, BinaryIO

from ..constants import MAGIC_NUMBER, VERSION
from .dtypes import DataType, ColumnInfo, FormatType

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

class BinaryReader:
    """Helper class for reading binary data from both files and buffers."""
    
    def __init__(self, source: BinaryIO):
        """Initialize reader with a file-like object."""
        self.source = source
        self._is_buffer = isinstance(source, io.BytesIO)
        
    def read_bytes(self, count: int) -> bytes:
        """
        Read a specified byte count from the source.
        Provided for uniformity of api experince.

        Args:
            count: number of bytes to read
            
        Returns:
            bytes
        """
        return self.source.read(count)

    def read_array(self, dtype: np.dtype, count: int) -> np.ndarray:
        """
        Read a numpy array of specified dtype and count from the source.
        
        Args:
            dtype: numpy dtype of the array to read
            count: number of elements to read
            
        Returns:
            numpy array of the requested type and size
        """
        # Calculate bytes to read
        bytes_to_read = dtype.itemsize * count
        
        if self._is_buffer:
            # For BytesIO, read exact number of bytes and use frombuffer
            data = self.source.read(bytes_to_read)
            if len(data) != bytes_to_read:
                raise EOFError(f"Insufficient data: expected {bytes_to_read} bytes, got {len(data)}")
            return np.frombuffer(data, dtype=dtype)
        else:
            # For files, use efficient fromfile
            # Save current position in case we need to retry
            pos = self.source.tell()
            try:
                result = np.fromfile(self.source, dtype=dtype, count=count)
                if len(result) != count:
                    raise EOFError(f"Insufficient data: expected {count} elements, got {len(result)}")
                return result
            except (AttributeError, TypeError):
                # If fromfile fails (e.g., with non-standard file objects),
                # fall back to read+frombuffer approach
                self.source.seek(pos)
                data = self.source.read(bytes_to_read)
                if len(data) != bytes_to_read:
                    raise EOFError(f"Insufficient data: expected {bytes_to_read} bytes, got {len(data)}")
                return np.frombuffer(data, dtype=dtype)