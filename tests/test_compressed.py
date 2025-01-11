"""/hybf/tests/test_compressed.py
"""

import os

from hybf import MinimalWriter, MinimalReader
from hybf import CompressedWriter, CompressedReader

from data.generators import DataGenerator
from utils import BaseFormatTest


class CompressedFormatTest(BaseFormatTest):
    """Test cases for compressed format."""
    
    def test_basic_roundtrip(self):
        """Test basic write and read functionality."""
        df = DataGenerator.create_compressed_dataset()
        
        with open(self.test_file, 'wb') as f:
            writer = CompressedWriter()
            writer.write(df, f)
        
        with open(self.test_file, 'rb') as f:
            reader = CompressedReader()
            df_read = reader.read(f)
        
        self.assertDataFrameEqual(df, df_read)
    
    def test_compression_strategies(self):
        """Test that appropriate compression strategies are selected."""
        df = DataGenerator.create_compressed_dataset(1000)
        
        # Record file size with different columns
        original_sizes = {col: df[col].memory_usage(deep=True) 
                        for col in df.columns}
        
        with open(self.test_file, 'wb') as f:
            writer = CompressedWriter()
            writer.write(df, f)
        
        file_size = os.path.getsize(self.test_file)
        
        # Verify compression ratio is reasonable
        total_original = sum(original_sizes.values())
        self.assertLess(file_size, total_original * 0.8)  # At least 20% compression
    
    # def test_edge_cases(self):
    #     """Test handling of edge cases."""
    #     df = DataGenerator.create_edge_cases_dataset()
        
    #     with open(self.test_file, 'wb') as f:
    #         writer = CompressedWriter()
    #         writer.write(df, f)
        
    #     with open(self.test_file, 'rb') as f:
    #         reader = CompressedReader()
    #         df_read = reader.read(f)
        
    #     self.assertDataFrameEqual(df, df_read)
        
    def test_edge_cases(self):
        """Test handling of edge cases, including mixed type conversion to strings."""
        df = DataGenerator.create_edge_cases_dataset()
        
        with open(self.test_file, 'wb') as f:
            writer = CompressedWriter()
            writer.write(df, f)
        
        with open(self.test_file, 'rb') as f:
            reader = CompressedReader()
            df_read = reader.read(f)
        
        # Convert mixed type columns to strings in the original dataframe
        # to match the expected behavior of the compressed format
        mixed_columns = ['mixed']
        df_expected = df.copy()
        for col in mixed_columns:
            df_expected[col] = df_expected[col].astype(str)
        
        # # Special handling for numeric columns with NaN values
        # numeric_columns = ['special_nums']
        # for col in numeric_columns:
        #     # Keep NaN as NaN, convert other values to strings
        #     mask = df_expected[col].notna()
        #     df_expected.loc[mask, col] = df_expected.loc[mask, col].astype(str)
        
        # # Handle various null representations
        # null_columns = ['nulls']
        # for col in null_columns:
        #     # Convert all null representations (None, np.nan, pd.NA) to None
        #     df_expected[col] = df_expected[col].where(df_expected[col].notna(), None)
        #     df_read[col] = df_read[col].where(df_read[col].notna(), None)
        
        self.assertDataFrameEqual(df_expected, df_read)