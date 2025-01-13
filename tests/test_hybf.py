# tests/test_hybf.py
import pytest
import numpy as np
import pandas as pd
import io
from hybf import write_dataframe, read_dataframe

def test_round_trip():
    """Test that data survives a round trip through HYBF format."""
    # Create test DataFrame
    df = pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        'str_col': ['a', 'b', 'c', 'd', 'e'],
        'bool_col': [True, False, True, False, True]
    })
    
    # Write to buffer
    buffer = io.BytesIO()
    writer = HYBFWriter()
    writer.write(df, buffer)
    
    # Read back
    buffer.seek(0)
    reader = HYBFReader()
    df_read = reader.read(buffer)
    
    # Compare
    pd.testing.assert_frame_equal(df, df_read)

def test_compression_threshold():
    """Test that large datasets use compression."""
    # Create large DataFrame
    large_df = pd.DataFrame({
        'int_col': np.random.randint(0, 100, 1000),
        'float_col': np.random.random(1000),
        'str_col': ['value'] * 1000  # Should trigger dictionary compression
    })
    
    buffer = io.BytesIO()
    writer = HYBFWriter()
    writer.write(large_df, buffer)
    
    # Verify compressed format was used
    buffer.seek(0)
    assert buffer.read(4) == b'HYBF'
    assert buffer.read(1) == bytes([1])  # version
    assert buffer.read(1) == bytes([2])  # COMPRESSED_FORMAT

if __name__ == '__main__':
    pytest.main([__file__])