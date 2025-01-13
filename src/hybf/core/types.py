# src/hybf/core/types.py
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np

class DataType(Enum):
    """Logical data types supported by HYBF."""
    INT32 = 1
    INT64 = 2
    FLOAT32 = 3
    FLOAT64 = 4
    STRING = 5
    BOOLEAN = 6

    @classmethod
    def from_numpy_dtype(cls, dtype: np.dtype) -> 'DataType':
        """Convert numpy dtype to HYBF DataType."""
        mapping = {
            np.int32: cls.INT32,
            np.int64: cls.INT64,
            np.float32: cls.FLOAT32,
            np.float64: cls.FLOAT64,
            np.bool_: cls.BOOLEAN,
            np.object_: cls.STRING
        }
        return mapping.get(dtype.type, cls.STRING)

@dataclass
class StorageType:
    """Physical storage representation of data."""
    base_type: DataType
    bit_width: int

    @classmethod
    def analyze(cls, data: np.ndarray) -> 'StorageType':
        """Determine optimal storage type for data."""
        dtype = DataType.from_numpy_dtype(data.dtype)
        
        if dtype in (DataType.STRING, DataType.BOOLEAN):
            return cls(dtype, 8)
            
        if dtype in (DataType.INT32, DataType.INT64):
            min_val = np.min(data)
            max_val = np.max(data)
            
            if min_val >= 0:
                if max_val < 256:
                    return cls(dtype, 8)
                elif max_val < 65536:
                    return cls(dtype, 16)
                elif max_val < 4294967296:
                    return cls(dtype, 32)
            return cls(dtype, 64)
            
        # Float types maintain their original precision
        return cls(dtype, 32 if dtype == DataType.FLOAT32 else 64)

@dataclass
class ColumnType:
    """Complete type information for a column."""
    name: str
    logical_type: DataType
    storage_type: StorageType