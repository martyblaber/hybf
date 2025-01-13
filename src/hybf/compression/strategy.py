# src/hybf/compression/strategy.py
from abc import ABC, abstractmethod
from typing import Type
import numpy as np
from ..core.columns import Column, RawColumn
from ..core.types import ColumnType

class CompressionStrategy(ABC):
    """Base class for compression strategies."""
    
    @abstractmethod
    def can_compress(self, data: np.ndarray) -> bool:
        """Determine if this strategy can compress the data."""
        pass

    @abstractmethod
    def compress(self, data: np.ndarray, type_info: ColumnType) -> Column:
        """Create compressed column from data."""
        pass

    @abstractmethod
    def estimate_size(self, data: np.ndarray) -> int:
        """Estimate compressed size without actually compressing."""
        pass

class DictionaryStrategy(CompressionStrategy):
    """Dictionary encoding for low-cardinality data."""
    
    def can_compress(self, data: np.ndarray) -> bool:
        if not data.size:
            return False
        unique_count = len(np.unique(data))
        return unique_count <= len(data) * 0.1  # 10% threshold

    def compress(self, data: np.ndarray, type_info: ColumnType) -> Column:
        # Basic dictionary encoding implementation
        unique_values = np.unique(data)
        value_to_id = {val: idx for idx, val in enumerate(unique_values)}
        encoded = np.array([value_to_id[val] for val in data], dtype=np.uint32)
        return RawColumn(type_info, encoded)

    def estimate_size(self, data: np.ndarray) -> int:
        unique_count = len(np.unique(data))
        return unique_count * 8 + len(data)  # Rough estimate