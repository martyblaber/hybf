# src/hybf/core/columns.py
from abc import ABC, abstractmethod
from typing import BinaryIO, Optional
import numpy as np
from .types import ColumnType

class Column(ABC):
    """Abstract base class for all column implementations."""
    
    def __init__(self, type_info: ColumnType, data: Optional[np.ndarray] = None):
        self.type_info = type_info
        self._data = data

    @abstractmethod
    def write(self, buffer: BinaryIO) -> None:
        """Write column data to buffer."""
        pass

    @abstractmethod
    def read(self, buffer: BinaryIO, row_count: int) -> np.ndarray:
        """Read column data from buffer."""
        pass

    @abstractmethod
    def get_size(self) -> int:
        """Get size in bytes when written."""
        pass

class RawColumn(Column):
    """Direct storage with optional bit-width reduction."""
    
    def write(self, buffer: BinaryIO) -> None:
        if self._data is None:
            raise ValueError("No data to write")
            
        # Write data using numpy's built-in binary format
        np.save(buffer, self._data)

    def read(self, buffer: BinaryIO, row_count: int) -> np.ndarray:
        return np.load(buffer)

    def get_size(self) -> int:
        if self._data is None:
            return 0
        return self._data.nbytes + 128  # Include numpy header overhead