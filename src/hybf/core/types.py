# src/hybf/core/types.py
"""Core type system for HYBF.

This module provides the type system infrastructure for the HYBF format,
including logical types, storage types, and type conversion utilities.
"""

from enum import Enum, auto
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional, Tuple

class LogicalType(Enum):
    """Represents the logical (user-facing) data type."""
    INT32 = auto()
    INT64 = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()
    STRING = auto()
    BOOLEAN = auto()
    
    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> 'LogicalType':
        """Convert numpy dtype to LogicalType."""
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
        """Convert LogicalType to numpy dtype."""
        mapping = {
            self.INT32: np.dtype('int32'),
            self.INT64: np.dtype('int64'),
            self.FLOAT32: np.dtype('float32'),
            self.FLOAT64: np.dtype('float64'),
            self.BOOLEAN: np.dtype('bool'),
            self.STRING: np.dtype('O'),
        }
        return mapping[self]

class StorageType(Enum):
    """Represents the physical storage format."""
    UINT8 = auto()
    UINT16 = auto()
    UINT32 = auto()
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
    INT64 = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()
    BOOL = auto()
    STRING = auto()
    
    def get_numpy_dtype(self) -> np.dtype:
        """Get the corresponding numpy dtype."""
        mapping = {
            self.UINT8: np.dtype('uint8'),
            self.UINT16: np.dtype('uint16'),
            self.UINT32: np.dtype('uint32'),
            self.INT8: np.dtype('int8'),
            self.INT16: np.dtype('int16'),
            self.INT32: np.dtype('int32'),
            self.INT64: np.dtype('int64'),
            self.FLOAT32: np.dtype('float32'),
            self.FLOAT64: np.dtype('float64'),
            self.BOOL: np.dtype('bool'),
            self.STRING: np.dtype('O'),
        }
        return mapping[self]
    
    def get_bit_width(self) -> int:
        """Get the bit width of the type."""
        mapping = {
            self.UINT8: 8,
            self.UINT16: 16,
            self.UINT32: 32,
            self.INT8: 8,
            self.INT16: 16,
            self.INT32: 32,
            self.INT64: 64,
            self.FLOAT32: 32,
            self.FLOAT64: 64,
            self.BOOL: 1,
            self.STRING: 0,  # Variable width
        }
        return mapping[self]

@dataclass
class ColumnTypeInfo:
    """Full type information for a column."""
    name: str
    logical_type: LogicalType
    storage_type: StorageType
    nullable: bool = True

class TypeAnalyzer:
    """Analyzes data to determine optimal type information."""
    
    @staticmethod
    def analyze_series(series: pd.Series) -> ColumnTypeInfo:
        """
        Analyze a pandas Series to determine optimal type information.
        
        Args:
            series: The pandas Series to analyze
            
        Returns:
            ColumnTypeInfo with optimal type settings
        """
        # Get logical type from series dtype
        logical_type = LogicalType.from_numpy(series.dtype)
        
        # Check for nulls
        nullable = series.isna().any()
        
        # Get optimal storage type
        if logical_type in (LogicalType.INT32, LogicalType.INT64):
            storage_type = TypeAnalyzer._analyze_integer_storage(series)
        elif logical_type in (LogicalType.FLOAT32, LogicalType.FLOAT64):
            storage_type = TypeAnalyzer._analyze_float_storage(series)
        else:
            # For boolean and string, storage matches logical type
            storage_type = {
                LogicalType.BOOLEAN: StorageType.BOOL,
                LogicalType.STRING: StorageType.STRING
            }[logical_type]
        
        return ColumnTypeInfo(
            name=series.name or '',
            logical_type=logical_type,
            storage_type=storage_type,
            nullable=nullable
        )
    
    @staticmethod
    def _analyze_integer_storage(series: pd.Series) -> StorageType:
        """Determine optimal integer storage type."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return StorageType.INT32  # Default for empty series
        
        min_val = non_null.min()
        max_val = non_null.max()
        
        # Check unsigned ranges first
        if min_val >= 0:
            if max_val <= 255:
                return StorageType.UINT8
            elif max_val <= 65535:
                return StorageType.UINT16
            elif max_val <= 4294967295:
                return StorageType.UINT32
        
        # Check signed ranges
        if min_val >= -128 and max_val <= 127:
            return StorageType.INT8
        elif min_val >= -32768 and max_val <= 32767:
            return StorageType.INT16
        elif min_val >= -2147483648 and max_val <= 2147483647:
            return StorageType.INT32
        
        return StorageType.INT64
    
    @staticmethod
    def _analyze_float_storage(series: pd.Series) -> StorageType:
        """Determine optimal float storage type."""
        # Try float32 and check if precision is adequate
        float32_series = series.astype(np.float32)
        if series.isna().equals(float32_series.isna()) and \
           np.allclose(series[~series.isna()], float32_series[~series.isna()], rtol=1e-6):
            return StorageType.FLOAT32
        return StorageType.FLOAT64

class TypeConverter:
    """Handles conversion between storage and logical types."""
    
    @staticmethod
    def to_storage_type(data: np.ndarray, type_info: ColumnTypeInfo) -> np.ndarray:
        """Convert data to its storage type."""
        if type_info.storage_type == StorageType.STRING:
            # Convert to string, handling None/NaN correctly
            return np.array([
                None if pd.isna(x) else str(x) 
                for x in data
            ], dtype='O')
        
        # For numeric types, convert using numpy dtype
        return data.astype(type_info.storage_type.get_numpy_dtype())
    
    @staticmethod
    def from_storage_type(data: np.ndarray, type_info: ColumnTypeInfo) -> np.ndarray:
        """Convert data from storage type to logical type."""
        if type_info.logical_type == LogicalType.STRING:
            # Keep strings as objects
            return data
        
        # Convert numeric data
        return data.astype(type_info.logical_type.to_numpy())