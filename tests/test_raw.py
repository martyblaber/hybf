# tests/test_raw.py
"""Test suite for raw data format handlers."""

import io
import numpy as np
import pandas as pd
import pytest

from hybf.formats.raw import RawWriter, RawReader
from hybf.core.types import LogicalType

class TestRawFormat:
    """Test cases for raw format handlers."""
    
    def test_numeric_roundtrip(self):
        """Test writing and reading numeric data."""
        test_cases = [
            # Integers
            pd.Series([1, 2, 3]),  # Small integers
            pd.Series([1000000, 2000000]),  # Large integers
            pd.Series([-5, 0, 5]),  # Signed integers
            pd.Series([1, None, 3]),  # Integers with null
            
            # Floats
            pd.Series([1.1, 2.2, 3.3]),  # Simple floats
            pd.Series([1.1111111111111111]),  # High precision float
            pd.Series([float('nan'), 1.0, 2.0]),  # Float with NaN
            
            # Boolean
            pd.Series([True, False, True]),
            pd.Series([True, None, False]),  # Boolean with null
        ]
        
        for series in test_cases:
            # Write to buffer
            buffer = io.BytesIO()
            RawWriter.write(buffer, series)
            
            # Read back
            buffer.seek(0)
            result = RawReader.read(
                buffer,
                LogicalType.from_numpy(series.dtype),
                len(series)
            )
            
            # Compare
            pd.testing.assert_series_equal(
                pd.Series(result),
                series,
                check_dtype=False  # Allow for type optimization
            )
    
    def test_string_roundtrip(self):
        """Test writing and reading string data."""
        test_cases = [
            # Basic strings
            pd.Series(['a', 'b', 'c']),
            
            # Strings with null
            pd.Series(['a', None, 'c']),
            
            # Empty strings
            pd.Series(['', 'b', '']),
            
            # Unicode strings
            pd.Series(['Hello', '世界', '🌍']),
            
            # Mixed length strings
            pd.Series(['a', 'bb', 'ccc']),
            
            # All nulls
            pd.Series([None, None, None]),
        ]
        
        for series in test_cases:
            # Write to buffer
            buffer = io.BytesIO()
            RawWriter.write(buffer, series)
            
            # Read back
            buffer.seek(0)
            result = RawReader.read(
                buffer,
                LogicalType.STRING,
                len(series)
            )
            
            # Compare
            pd.testing.assert_series_equal(
                pd.Series(result),
                series,
                check_dtype=False
            )
    
    def test_mixed_types(self):
        """Test handling of mixed type data."""
        # Mixed numeric types
        series = pd.Series([1, 2.5, 3])
        buffer = io.BytesIO()
        RawWriter.write(buffer, series)
        buffer.seek(0)
        result = RawReader.read(buffer, LogicalType.FLOAT64, len(series))
        pd.testing.assert_series_equal(
            pd.Series(result),
            series,
            check_dtype=False
        )
        
        # Mixed types that should convert to string
        series = pd.Series([1, 'two', 3.0])
        buffer = io.BytesIO()
        RawWriter.write(buffer, series)
        buffer.seek(0)
        result = RawReader.read(buffer, LogicalType.STRING, len(series))
        pd.testing.assert_series_equal(
            pd.Series(result).astype(str),
            series.astype(str),
            check_dtype=False
        )
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty series
        series = pd.Series([], dtype=float)
        buffer = io.BytesIO()
        RawWriter.write(buffer, series)
        buffer.seek(0)
        result = RawReader.read(buffer, LogicalType.FLOAT64, 0)
        assert len(result) == 0
        
        # Series with only nulls
        series = pd.Series([None, None, None])
        buffer = io.BytesIO()
        RawWriter.write(buffer, series)
        buffer.seek(0)
        result = RawReader.read(buffer, LogicalType.STRING, len(series))
        pd.testing.assert_series_equal(
            pd.Series(result),
            series,
            check_dtype=False
        )
        
        # Boundary value tests
        series = pd.Series([127, -128])  # int8 boundaries
        buffer = io.BytesIO()
        RawWriter.write(buffer, series)
        buffer.seek(0)
        result = RawReader.read(buffer, LogicalType.INT32, len(series))
        pd.testing.assert_series_equal(
            pd.Series(result),
            series,
            check_dtype=False
        )