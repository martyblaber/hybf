# tests/test_minimal.py
"""Test suite for minimal format implementation."""

import io
import numpy as np
import pandas as pd
import pytest
from typing import Any, Dict

from hybf.formats.minimal import MinimalWriter, MinimalReader
from hybf.constants import MAGIC_NUMBER, VERSION
from hybf.core.dtypes import FormatType

class TestMinimalFormat:
    """Test cases for minimal format implementation."""
    
    def test_basic_roundtrip(self):
        """Test basic write and read functionality."""
        # Create test data
        data = {
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        }
        df = pd.DataFrame(data)
        
        # Write to buffer
        buffer = io.BytesIO()
        writer = MinimalWriter()
        writer.write(df, buffer)
        
        # Read back
        buffer.seek(0)
        reader = MinimalReader()
        df_read = reader.read(buffer)
        
        # Compare
        pd.testing.assert_frame_equal(df, df_read, check_dtype=False)
    
    def test_null_handling(self):
        """Test handling of null values."""
        # Create test data with nulls in different types
        data = {
            'int_nulls': [1, None, 3],
            'float_nulls': [1.1, None, 3.3],
            'str_nulls': ['a', None, 'c'],
            'bool_nulls': [True, None, False],
            'all_nulls': [None, None, None]
        }
        df = pd.DataFrame(data)
        
        # Write to buffer
        buffer = io.BytesIO()
        writer = MinimalWriter()
        writer.write(df, buffer)
        
        # Read back
        buffer.seek(0)
        reader = MinimalReader()
        df_read = reader.read(buffer)
        
        # Compare
        pd.testing.assert_frame_equal(df, df_read, check_dtype=False)
    
    def test_edge_cases(self):
        """Test edge cases and special values."""
        # Create test data with edge cases
        data = {
            # Empty strings
            'empty_str': ['', 'normal', ''],
            
            # Unicode strings
            'unicode': ['Hello', '世界', '🌍'],
            
            # Special numeric values
            'special_float': [np.inf, -np.inf, np.nan],
            
            # Extreme integers
            'big_int': [np.iinfo(np.int32).max, 0, np.iinfo(np.int32).min],
            
            # Boolean edge cases
            'sparse_bool': [True, None, False]
        }
        df = pd.DataFrame(data)
        
        # Write to buffer
        buffer = io.BytesIO()
        writer = MinimalWriter()
        writer.write(df, buffer)
        
        # Read back
        buffer.seek(0)
        reader = MinimalReader()
        df_read = reader.read(buffer)
        
        # Compare
        pd.testing.assert_frame_equal(df, df_read, check_dtype=False)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        # Empty DataFrame with columns
        df = pd.DataFrame(columns=['a', 'b', 'c'])
        
        # Write to buffer
        buffer = io.BytesIO()
        writer = MinimalWriter()
        writer.write(df, buffer)
        
        # Read back
        buffer.seek(0)
        reader = MinimalReader()
        df_read = reader.read(buffer)
        
        # Compare
        pd.testing.assert_frame_equal(df, df_read, check_dtype=False)
    
    def test_format_validation(self):
        """Test format validation in reader."""
        # Test invalid magic number
        buffer = io.BytesIO()
        buffer.write(b'INVALID')
        buffer.seek(0)
        reader = MinimalReader()
        with pytest.raises(ValueError, match="Invalid file format"):
            reader.read(buffer)
        
        # Test invalid version
        buffer = io.BytesIO()
        buffer.write(MAGIC_NUMBER)
        buffer.write(struct.pack('BB', 255, FormatType.MINIMAL.value))  # Invalid version
        buffer.seek(0)
        with pytest.raises(ValueError, match="Unsupported version"):
            reader.read(buffer)
        
        # Test invalid format type
        buffer = io.BytesIO()
        buffer.write(MAGIC_NUMBER)
        buffer.write(struct.pack('BB', VERSION, FormatType.COMPRESSED.value))  # Wrong format
        buffer.seek(0)
        with pytest.raises(ValueError, match="Not a minimal format file"):
            reader.read(buffer)
    
    def test_mixed_types(self):
        """Test handling of mixed type data."""
        # Mixed numeric types that should be converted
        data = {
            'mixed_ints': [1, 2.0, 3],  # Should stay as int
            'mixed_floats': [1, 2.5, 3],  # Should become float
            'mixed_strings': [1, 'two', 3.0],  # Should become string
        }
        df = pd.DataFrame(data)
        
        # Write to buffer
        buffer = io.BytesIO()
        writer = MinimalWriter()
        writer.write(df, buffer)