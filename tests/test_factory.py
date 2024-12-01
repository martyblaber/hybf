from hybf import FormatFactory
from hybf import MinimalWriter, MinimalReader
from hybf import CompressedWriter, CompressedReader

from utils import BaseFormatTest
from data.generators import DataGenerator

class FormatFactoryTest(BaseFormatTest):
    """Test cases for format factory."""
    
    def test_format_selection(self):
        """Test that appropriate format is selected based on data size."""
        # Test with small dataset
        df_small = DataGenerator.create_minimal_dataset()
        writer = FormatFactory.create_writer(df_small)
        self.assertIsInstance(writer, MinimalWriter)
        
        # Test with large dataset
        df_large = DataGenerator.create_compressed_dataset()
        writer = FormatFactory.create_writer(df_large)
        self.assertIsInstance(writer, CompressedWriter)
    
    def test_reader_selection(self):
        """Test that appropriate reader is selected based on file format."""
        # Write and read with minimal format
        df_small = DataGenerator.create_minimal_dataset()
        with open(self.test_file, 'wb') as f:
            writer = MinimalWriter()
            writer.write(df_small, f)
        
        with open(self.test_file, 'rb') as f:
            reader = FormatFactory.create_reader(f)
            self.assertIsInstance(reader, MinimalReader)
        
        # Write and read with compressed format
        df_large = DataGenerator.create_compressed_dataset()
        with open(self.test_file, 'wb') as f:
            writer = CompressedWriter()
            writer.write(df_large, f)
        
        with open(self.test_file, 'rb') as f:
            reader = FormatFactory.create_reader(f)
            self.assertIsInstance(reader, CompressedReader)
