"""/hybf/test/conftest.py
PyTest configuration and shared fixtures.
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data that persists across the test session."""
    temp_dir = tmp_path_factory.mktemp("test_data")
    return temp_dir

@pytest.fixture
def temp_file(test_data_dir):
    """Provide a temporary file path for each test that needs it."""
    temp_file = test_data_dir / "test.hybf"
    yield temp_file
    # Cleanup
    if temp_file.exists():
        temp_file.unlink()

@pytest.fixture
def sample_small_df(test_data_dir):
    """Provide a small DataFrame for testing minimal format."""
    from data.generators import DataGenerator
    return DataGenerator.create_minimal_dataset()

@pytest.fixture
def sample_large_df(test_data_dir):
    """Provide a large DataFrame for testing compressed format."""
    from data.generators import DataGenerator
    return DataGenerator.create_compressed_dataset()

@pytest.fixture
def edge_cases_df(test_data_dir):
    """Provide a DataFrame with edge cases."""
    from data.generators import DataGenerator
    return DataGenerator.create_edge_cases_dataset()

@pytest.fixture
def assert_frame_equal():
    """Provide a custom DataFrame comparison function."""
    def _assert_frame_equal(df1, df2):
        """Assert that two DataFrames are equal, handling NaN and null values."""
        import pandas as pd
        import numpy as np
        
        # Check column names and order
        assert list(df1.columns) == list(df2.columns)
        
        # Check data types
        for col in df1.columns:
            # Allow for some type flexibility (e.g., int32 vs int64)
            assert (np.issubdtype(df1[col].dtype, np.number) == 
                   np.issubdtype(df2[col].dtype, np.number))
        
        # Check values
        for col in df1.columns:
            s1, s2 = df1[col], df2[col]
            
            # Handle numeric columns
            if np.issubdtype(s1.dtype, np.number):
                pd.testing.assert_series_equal(
                    s1, s2, check_dtype=False, check_exact=False
                )
            else:
                # Handle string and other columns
                pd.testing.assert_series_equal(
                    s1, s2, check_dtype=False
                )
    
    return _assert_frame_equal

def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )