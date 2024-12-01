from enum import Enum
from dataclasses import dataclass
import numpy as np

class DataType(Enum):
    """Supported data types in the binary format."""
    INT32 = 1
    INT64 = 2
    FLOAT32 = 3
    FLOAT64 = 4
    STRING = 5
    BOOLEAN = 6

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> 'DataType':
        """Convert numpy dtype to our DataType enum."""
        mapping = {
            np.dtype('int32'): cls.INT32,
            np.dtype('int64'): cls.INT64,
            np.dtype('float32'): cls.FLOAT32,
            np.dtype('float64'): cls.FLOAT64,
            np.dtype('bool'): cls.BOOLEAN,
            np.dtype('O'): cls.STRING,  # Object type usually means string in pandas
        }
        return mapping.get(dtype, cls.STRING)

    def to_numpy(self) -> np.dtype:
        """Convert DataType to numpy dtype."""
        mapping = {
            self.INT32: np.dtype('int32'),
            self.INT64: np.dtype('int64'),
            self.FLOAT32: np.dtype('float32'),
            self.FLOAT64: np.dtype('float64'),
            self.BOOLEAN: np.dtype('bool'),
            self.STRING: np.dtype('O'),
        }
        return mapping[self]

@dataclass
class ColumnInfo:
    """Metadata for a column."""
    name: str
    dtype: DataType


class FormatType(Enum):
    """Available format types."""
    MINIMAL = 1
    COMPRESSED = 2


class CompressionType(Enum):
    """Available compression strategies."""
    RAW = 1
    RLE = 2
    DICTIONARY = 3
    SINGLE_VALUE = 4
    NULL = 5
