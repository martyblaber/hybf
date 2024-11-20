import pandas as pd

from hybf import MinimalWriter, MinimalReader

from tests.utils import BaseFormatTest
from tests.data.generators import DataGenerator

class MinimalFormatTest(BaseFormatTest):
    """Test cases for minimal format."""
    
    def test_basic_roundtrip(self):
        """Test basic write and read functionality."""
        df = DataGenerator.create_minimal_dataset()
        
        with open(self.test_file, 'wb') as f:
            writer = MinimalWriter()
            writer.write(df, f)
        
        with open(self.test_file, 'rb') as f:
            reader = MinimalReader()
            df_read = reader.read(f)
        
        self.assertDataFrameEqual(df, df_read)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=['a', 'b', 'c'])
        
        with open(self.test_file, 'wb') as f:
            writer = MinimalWriter()
            writer.write(df, f)
        
        with open(self.test_file, 'rb') as f:
            reader = MinimalReader()
            df_read = reader.read(f)
        
        self.assertDataFrameEqual(df, df_read)
