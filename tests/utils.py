"""
Shared test utilities and base classes.
"""

import unittest
import tempfile
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any

class BaseFormatTest(unittest.TestCase):
    """Base class for format tests with common utilities."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, 'test.hybf')
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.test_dir)
    
    def assertDataFrameEqual(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """Assert that two DataFrames are equal, handling NaN and null values."""
        # Check column names and order
        self.assertEqual(list(df1.columns), list(df2.columns))
        
        # Check data types
        for col in df1.columns:
            # Allow for some type flexibility (e.g., int32 vs int64)
            self.assertTrue(
                np.issubdtype(df1[col].dtype, np.number) == 
                np.issubdtype(df2[col].dtype, np.number)
            )
        
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
