"""/hybf/test/data/generators.py
Test suite for the hybrid binary format implementation.
Includes utilities for generating test data and comprehensive test cases.
"""

import unittest
import tempfile
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any

class DataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def create_minimal_dataset() -> pd.DataFrame:
        """Create a small dataset suitable for minimal format."""
        return pd.DataFrame({
            'int_col': [1, 2],
            'float_col': [1.1, 2.2],
            'str_col': ['a', 'b'],
            'bool_col': [True, False],
            'null_col': [None, None]
        })
    
    @staticmethod
    def create_compressed_dataset(rows: int = 1000) -> pd.DataFrame:
        """Create a larger dataset with various compression opportunities."""
        # Generate base data
        data = {
            # Single value column
            'constant': ['const_value'] * rows,
            
            # High cardinality string column
            'unique_strings': [f'unique_value_{i}' for i in range(rows)],
            
            # Low cardinality string column (good for dictionary encoding)
            'categorical': np.random.choice(['cat_a', 'cat_b', 'cat_c'], rows),
            
            # Low cardinality string column (good for dictionary encoding), with Nan
            'categorical_with_nan': np.random.choice(['cat_a', 'cat_b', 'cat_c', np.nan], rows),

            # Numeric column with runs (good for RLE)
            'runs': np.repeat(np.arange(rows // 100), 100),
            
            # Regular numeric columns
            'random_ints': np.random.randint(0, 1000, rows),
            'random_floats': np.random.randn(rows),

            # Null column
            'nulls': [None] * rows,
            
            # Mixed nulls
            'sparse': np.where(np.random.random(rows) < 0.8, None, 'value'),

            # Mixed int and null
            'random_int_and_null': np.where(np.random.random(rows) < 0.1, [None]*rows, np.random.randint(0, 1000, rows)),

            'random_int_and_multinull': np.where(np.random.random(rows) < 0.1, 
                                                 np.random.choice([None, pd.NA, np.nan], rows), 
                                                 np.random.randint(0, 1000, rows)),

            # Mixed int and null
            'random_float_and_null': np.where(np.random.random(rows) < 0.1, [None]*rows, np.random.randn(rows))

        }
        return pd.DataFrame(data)
    
    @staticmethod
    def create_edge_cases_dataset() -> pd.DataFrame:
        """Create a dataset with edge cases to test robustness."""
        return pd.DataFrame({
            # Empty strings
            'empty_strings': ['', 'normal', ''],
            
            # Unicode strings
            'unicode': ['Hello', '世界', '🌍'],
            
            # Mixed types (will be converted to strings)
            'mixed': ['str', 123, 45.6],
            
            # Special numeric values
            'special_nums': [np.inf, -np.inf, np.nan],
            
            # # Various null representations
            # 'nulls': [None, np.nan, pd.NA]
        })

