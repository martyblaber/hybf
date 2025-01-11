"""/hybf/tests/test_types.py
Test suite for the HYBF type system."""

import numpy as np
import pandas as pd
import pytest
from hybf.core.types import (
    LogicalType,
    StorageType,
    ColumnTypeInfo,
    TypeAnalyzer,
    TypeConverter
)

class TestLogicalType:
    """Test cases for LogicalType enum."""
    
    def test_numpy_conversion(self):
        """Test conversion between numpy dtypes and LogicalType."""
        test_cases = [
            (np.dtype('int32'), LogicalType.INT32),
            (np.dtype('int64'), LogicalType.INT64),
            (np.dtype('float32'), LogicalType.FLOAT32),
            (np.dtype('float64'), LogicalType.FLOAT64),
            (np.dtype('bool'), LogicalType.BOOLEAN),
            (np.dtype('O'), LogicalType.STRING),
        ]
        
        for numpy_dtype, logical_type in test_cases:
            assert LogicalType.from_numpy(numpy_dtype) == logical_type
            assert logical_type.to_numpy() == numpy_dtype

class TestStorageType:
    """Test cases for StorageType enum."""
    
    def test_bit_widths(self):
        """Test bit width calculations."""
        test_cases = [
            (StorageType.UINT8, 8),
            (StorageType.INT16, 16),
            (StorageType.INT32, 32),
            (StorageType.FLOAT64, 64),
            (StorageType.BOOL, 1),
            (StorageType.STRING, 0),
        ]
        
        for storage_type, expected_width in test_cases:
            assert storage_type.get_bit_width() == expected_width
    
    def test_numpy_dtype_conversion(self):
        """Test conversion to numpy dtypes."""
        test_cases = [
            (StorageType.UINT8, np.dtype('uint8')),
            (StorageType.INT16, np.dtype('int16')),
            (StorageType.FLOAT32, np.dtype('float32')),
            (StorageType.STRING, np.dtype('O')),
        ]
        
        for storage_type, expected_dtype in test_cases:
            assert storage_type.get_numpy_dtype() == expected_dtype

class TestTypeAnalyzer:
    """Test cases for TypeAnalyzer."""
    
    def test_integer_analysis(self):
        """Test analysis of integer series."""
        test_cases = [
            # (data, expected_storage_type, expected_nullable)
            (pd.Series([1, 2, 3]), StorageType.UINT8, False),
            (pd.Series([-5, 0, 5]), StorageType.INT8, False),
            (pd.Series([0, 1000]), StorageType.UINT16, False),
            (pd.Series([1, None, 3]), StorageType.UINT8, True),
            (pd.Series([1000000]), StorageType.UINT32, False),
        ]
        
        for data, expected_storage, expected_nullable in test_cases:
            type_info = TypeAnalyzer.analyze_series(data)
            assert type_info.storage_type == expected_storage
            assert type_info.nullable == expected_nullable
            assert type_info.logical_type in (LogicalType.INT32, LogicalType.INT64)
    
    def test_float_analysis(self):
        """Test analysis of float series."""
        # Small float that can be represented in float32
        small_float = pd.Series([1.1, 2.2, 3.3])
        type_info = TypeAnalyzer.analyze_series(small_float)
        assert type_info.storage_type == StorageType.FLOAT32
        
        # Float requiring float64 precision
        large_float = pd.Series([1.1111111111111111])
        type_info = TypeAnalyzer.analyze_series(large_float)
        assert type_info.storage_type == StorageType.FLOAT64
    
    def test_string_analysis(self):
        """Test analysis of string series."""
        strings = pd.Series(['a', 'b', None])
        type_info = TypeAnalyzer.analyze_series(strings)
        assert type_info.logical_type == LogicalType.STRING
        assert type_info.storage_type == StorageType.STRING
        assert type_info.nullable == True

class TestTypeConverter:
    """Test cases for TypeConverter."""
    
    def test_numeric_conversion(self):
        """Test conversion of numeric data."""
        # Create test data
        data = np.array([1, 2, 3], dtype=np.int64)
        type_info = ColumnTypeInfo(
            name='test',
            logical_type=LogicalType.INT64,
            storage_type=StorageType.INT16,
            nullable=False
        )
        
        # Convert to storage type
        storage_data = TypeConverter.to_storage_type(data, type_info)
        assert storage_data.dtype == np.dtype('int16')
        
        # Convert back to logical type
        logical_data = TypeConverter.from_storage_type(storage_data, type_info)
        assert pd.isna(logical_data[1])
        assert logical_data[0] == 1
        assert logical_data[2] == 3
    
    def test_edge_cases(self):
        """Test handling of edge cases."""
        test_cases = [
            # Empty series
            (
                pd.Series([], dtype=float),
                LogicalType.FLOAT64,
                StorageType.FLOAT64,
                True
            ),
            # Series with only nulls
            (
                pd.Series([None, None]),
                LogicalType.STRING,
                StorageType.STRING,
                True
            ),
            # Mixed null types (None, np.nan, pd.NA)
            (
                pd.Series([None, np.nan, pd.NA]),
                LogicalType.STRING,
                StorageType.STRING,
                True
            ),
            # Integers that just fit in smaller types
            (
                pd.Series([127, -128]),  # Boundary of int8
                LogicalType.INT32,
                StorageType.INT8,
                False
            ),
            # Mixed numeric types
            (
                pd.Series([1, 2.5, 3]),
                LogicalType.FLOAT64,
                StorageType.FLOAT64,
                False
            ),
        ]
        
        for data, expected_logical, expected_storage, expected_nullable in test_cases:
            type_info = TypeAnalyzer.analyze_series(data)
            assert type_info.logical_type == expected_logical
            assert type_info.storage_type == expected_storage
            assert type_info.nullable == expected_nullable_info)
        assert logical_data.dtype == np.dtype('int64')
        np.testing.assert_array_equal(logical_data, data)
    
    def test_string_conversion(self):
        """Test conversion of string data."""
        # Test mixed type conversion to strings
        data = np.array([1, 'two', 3.0], dtype='O')
        type_info = ColumnTypeInfo(
            name='test',
            logical_type=LogicalType.STRING,
            storage_type=StorageType.STRING,
            nullable=False
        )
        
        # Convert to storage type (strings)
        storage_data = TypeConverter.to_storage_type(data, type_info)
        assert all(isinstance(x, str) for x in storage_data)
        assert list(storage_data) == ['1', 'two', '3.0']
    
    def test_null_handling(self):
        """Test handling of null values during conversion."""
        # Create data with nulls
        data = np.array([1, None, 3], dtype='O')
        type_info = ColumnTypeInfo(
            name='test',
            logical_type=LogicalType.INT32,
            storage_type=StorageType.INT16,
            nullable=True
        )
        
        # Convert to storage type
        storage_data = TypeConverter.to_storage_type(data, type_info)
        assert pd.isna(storage_data[1])
        
        # Convert back
        logical_data = TypeConverter.from_storage_type(storage_data, type