"""/hybf/src/hybf/factory.py
Provide FormatFactory to choose between a minimalistic format and a compressed one.
"""

import pandas as pd
import struct
from typing import Tuple, List, Dict, Any, BinaryIO

from hybf import BaseWriter, BaseReader
from hybf.core.dtypes import FormatType

from hybf import MinimalWriter, MinimalReader
from hybf import CompressedWriter, CompressedReader

# Factory for creating appropriate reader/writer based on data characteristics
class FormatFactory:
    @staticmethod
    def create_writer(df: pd.DataFrame) -> BaseWriter:
        """Create appropriate writer based on DataFrame characteristics."""
        estimated_size = (
            df.memory_usage(deep=True).sum() +  # Data size
            sum(len(name.encode('utf-8')) for name in df.columns) +  # Column names
            8 +  # Header
            2 * len(df.columns)  # Column definitions
        )
        
        return MinimalWriter() if estimated_size <= 4096 else CompressedWriter()
    
    @staticmethod
    def create_reader(file: BinaryIO) -> BaseReader:
        """Create appropriate reader based on file format."""
        # Save current position
        pos = file.tell()
        
        # Read format type from header
        file.seek(5)  # Skip magic number and version
        format_type = FormatType(struct.unpack('B', file.read(1))[0])
        
        # Restore position
        file.seek(pos)
        
        return MinimalReader() if format_type == FormatType.MINIMAL else CompressedReader()
    
    
