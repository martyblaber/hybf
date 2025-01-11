"""/hybf/src/hybf/formats/compressed.py
Reader and Writer and CompressionSelector for Compressed files. Multiple compression styles supported.
"""
import struct
import pandas as pd
import numpy as np
import io
from typing import Tuple, List, Dict, Any, BinaryIO

from hybf import BaseWriter
from hybf import BaseReader, BinaryReader
from hybf.core.encoding import BitPackedDictionaryWriter, BitPackedDictionaryReader 
from hybf import DataType
from hybf import ColumnInfo
from hybf.core.dtypes import CompressionType, FormatType
from hybf.utils.numeric import analyze_numeric_column, read_numeric_column, write_numeric_column
from hybf.formats.raw import RawWriter, RawReader

class CompressionSelector:
    """Analyzes columns to determine optimal compression strategy."""
    
    def __init__(self, uniqueness_threshold: float = 0.1, redundancy_threshold: float = 0.5):
        self.uniqueness_threshold = uniqueness_threshold  # For dictionary encoding
        self.redundancy_threshold = redundancy_threshold  # For RLE
        
    def select_strategy(self, series: pd.Series) -> Tuple[CompressionType, Any]:
        """
        Analyze a column and return the best compression strategy and any needed metadata.
        Returns tuple of (compression_type, metadata)
        """
        # Check for null column
        if series.isna().all():
            return CompressionType.NULL, None
        
        if len(series[series.isna()].drop_duplicates()) > 1:
            Warning("Series has more than one type of null value. These will be converted to None.")
        
        # Check for single value, properly handling NaN
        non_null_values = series.dropna()
        if len(non_null_values) > 0 and non_null_values.nunique() == 1 and series.isna().sum() == 0:
            return CompressionType.SINGLE_VALUE, series.iloc[0]

        #What is the type of the non-null values?
        #if non_null_values.infer_objects().dtypes == object:
            
        # Calculate value frequencies
        value_counts = series.value_counts()
        unique_ratio = len(value_counts) / len(series)
        
        # For string columns, consider dictionary encoding
        if series.dtype == object and unique_ratio <= self.uniqueness_threshold:
            return CompressionType.DICTIONARY, dict(enumerate(value_counts.index))
            
        # For numeric columns, check for run-length encoding potential
        if np.issubdtype(series.dtype, np.number):
            # Calculate runs
            runs = self._calculate_runs(series)
            if len(runs) / len(series) <= self.redundancy_threshold:
                return CompressionType.RLE, None
                
        # Default to raw storage
        return CompressionType.RAW, None
    
    def _calculate_runs(self, series: pd.Series) -> List[Tuple[Any, int]]:
        """Calculate run-length encoding runs for a series."""
        runs = []
        current_value = series.iloc[0]
        current_count = 1
        
        for value in series.iloc[1:]:
            if value == current_value:
                current_count += 1
            else:
                runs.append((current_value, current_count))
                current_value = value
                current_count = 1
                
        runs.append((current_value, current_count))
        return runs



class CompressedWriter(BaseWriter):
    """Writer implementation for the compressed format."""
    
    def __init__(self):
        self.compression_selector = CompressionSelector()
    
    def write(self, df: pd.DataFrame, file: BinaryIO) -> None:
        # Write header
        self.write_header(file, FormatType.COMPRESSED, len(df.columns))
        
        # Write column definitions
        columns = [
            ColumnInfo(name, DataType.from_numpy(df[name].dtype))
            for name in df.columns
        ]
        self.write_column_definitions(file, columns)
        
        # Write row count
        file.write(struct.pack('>I', len(df)))
        
        # Process and write each column
        for col in columns:
            series = df[col.name]
            compression_type, metadata = self.compression_selector.select_strategy(series)
            self._write_compressed_column(file, series, compression_type, metadata)
    
    def _write_compressed_column(
        self, 
        file: BinaryIO, 
        series: pd.Series, 
        compression_type: CompressionType,
        metadata: Any
    ) -> None:
        """Write a compressed column to the file."""
        # Write compression type
        file.write(struct.pack('B', compression_type.value))
        
        # Use a temporary buffer to build compressed data
        with io.BytesIO() as buffer:
            if compression_type == CompressionType.RAW:
                self._write_raw(buffer, series)
            elif compression_type == CompressionType.RLE:
                self._write_rle(buffer, series)
            elif compression_type == CompressionType.DICTIONARY:
                self._write_dictionary(buffer, series, metadata)
            elif compression_type == CompressionType.SINGLE_VALUE:
                self._write_single_value(buffer, metadata, len(series))
            elif compression_type == CompressionType.NULL:
                self._write_null_column(buffer, len(series))
            
            # Write compressed size and data
            compressed_data = buffer.getvalue()
            file.write(struct.pack('>I', len(compressed_data)))
            file.write(compressed_data)
   
    def _write_raw(self, buffer: BinaryIO, series: pd.Series) -> None:
        RawWriter.write(buffer, series)

    def _write_rle(self, buffer: BinaryIO, series: pd.Series) -> None:
        """Write column data using run-length encoding."""
        runs = self.compression_selector._calculate_runs(series)
        
        # Write number of runs
        buffer.write(struct.pack('>I', len(runs)))
        
        # Write each run
        for value, count in runs:
            if pd.isna(value):
                # Special handling for null values
                buffer.write(struct.pack('B', 0))
            elif isinstance(value, (int, np.integer)):
                buffer.write(struct.pack('>Bq', 1, value))
            elif isinstance(value, (float, np.floating)):
                buffer.write(struct.pack('>Bd', 2, value))
            else:  # String
                val_bytes = str(value).encode('utf-8')
                buffer.write(struct.pack('>BB', 3, len(val_bytes)))
                buffer.write(val_bytes)
            buffer.write(struct.pack('>I', count))
    
    def _write_dictionary(self, buffer: BinaryIO, series: pd.Series, value_dict: Dict) -> None:
        writer = BitPackedDictionaryWriter()
        writer.write_dictionary(buffer, series, value_dict)

    def _write_single_value(self, buffer: BinaryIO, value: Any, length: int) -> None:
        """Write a column containing a single value repeated."""
        # Write the value
        if pd.isna(value):
            buffer.write(struct.pack('B', 0))
        elif isinstance(value, (int, np.integer)):
            buffer.write(struct.pack('>Bq', 1, value))
        elif isinstance(value, (float, np.floating)):
            buffer.write(struct.pack('>Bd', 2, value))
        else:  # String
            val_bytes = str(value).encode('utf-8')
            buffer.write(struct.pack('>BB', 3, len(val_bytes)))
            buffer.write(val_bytes)
        
        # Write the length
        buffer.write(struct.pack('>I', length))
    
    def _write_null_column(self, buffer: BinaryIO, length: int) -> None:
        """Write a column containing only null values."""
        buffer.write(struct.pack('>I', length))

class CompressedReader(BaseReader):
    """Reader implementation for the compressed format."""
    
    def read(self, file: BinaryIO) -> pd.DataFrame:
        # Read header
        version, format_type, num_columns = self.read_header(file)
        if format_type != FormatType.COMPRESSED:
            raise ValueError("Not a compressed format file")
            
        # Read column definitions
        columns = self.read_column_definitions(file, num_columns)
        
        # Read row count
        row_count = struct.unpack('>I', file.read(4))[0]
        
        # Read each column
        data = {}
        for col in columns:
            data[col.name] = self._read_compressed_column(file, col.dtype, row_count)
            
        return pd.DataFrame(data)
    
    def _read_compressed_column(
        self, 
        file: BinaryIO, 
        dtype: DataType,
        row_count: int
    ) -> np.ndarray:
        """Read a compressed column from the file."""
        # Read compression type
        compression_type = CompressionType(struct.unpack('B', file.read(1))[0])
        
        # Read compressed size
        compressed_size = struct.unpack('>I', file.read(4))[0]
        
        # Read compressed data into buffer
        compressed_data = file.read(compressed_size)
        with io.BytesIO(compressed_data) as buffer:
            if compression_type == CompressionType.RAW:
                return self._read_raw(buffer, dtype, row_count)
            elif compression_type == CompressionType.RLE:
                return self._read_rle(buffer, dtype, row_count)
            elif compression_type == CompressionType.DICTIONARY:
                return self._read_dictionary(buffer, row_count)
            elif compression_type == CompressionType.SINGLE_VALUE:
                return self._read_single_value(buffer, dtype, row_count)
            elif compression_type == CompressionType.NULL:
                return self._read_null_column(buffer, row_count)
            else:
                raise ValueError(f"Unknown compression type: {compression_type}")

    def _read_raw(self, buffer: BinaryIO, dtype: DataType, row_count: int) -> np.ndarray:
        """Read raw column data with optimized numeric handling."""
        return RawReader.read(buffer, dtype, row_count)
    

    def _read_rle(self, buffer: BinaryIO, dtype: DataType, row_count: int) -> np.ndarray:
        """Read run-length encoded column data."""
        # Read number of runs
        num_runs = struct.unpack('>I', buffer.read(4))[0]
        
        values = []
        for _ in range(num_runs):
            # Read value type and value
            value_type = struct.unpack('B', buffer.read(1))[0]
            if value_type == 0:  # null
                value = None
            elif value_type == 1:  # integer
                value = struct.unpack('>q', buffer.read(8))[0]
            elif value_type == 2:  # float
                value = struct.unpack('>d', buffer.read(8))[0]
            elif value_type == 3:  # string
                length = struct.unpack('B', buffer.read(1))[0]
                value = buffer.read(length).decode('utf-8')
            else:
                raise ValueError(f"Unknown RLE value type: {value_type}")
            
            # Read run length
            count = struct.unpack('>I', buffer.read(4))[0]
            values.extend([value] * count)
        
        return np.array(values, dtype=dtype.to_numpy())
    
    def _read_dictionary(self, buffer: BinaryIO, row_count: int) -> np.ndarray:
        reader = BitPackedDictionaryReader()
        return reader.read_dictionary(buffer, row_count)
    
    def _read_single_value(self, buffer: BinaryIO, dtype: DataType, row_count: int) -> np.ndarray:
        """Read a column containing a single repeated value."""
        # Read value type and value
        value_type = struct.unpack('B', buffer.read(1))[0]
        
        if value_type == 0:  # null
            value = None
        elif value_type == 1:  # integer
            value = struct.unpack('>q', buffer.read(8))[0]
        elif value_type == 2:  # float
            value = struct.unpack('>d', buffer.read(8))[0]
        elif value_type == 3:  # string
            length = struct.unpack('B', buffer.read(1))[0]
            value = buffer.read(length).decode('utf-8')
        else:
            raise ValueError(f"Unknown single value type: {value_type}")
        
        # Read length (for verification)
        stored_length = struct.unpack('>I', buffer.read(4))[0]
        if stored_length != row_count:
            raise ValueError(f"Length mismatch in single-value column: expected {row_count}, got {stored_length}")
        
        return np.full(row_count, value, dtype=dtype.to_numpy())
    
    def _read_null_column(self, buffer: BinaryIO, row_count: int) -> np.ndarray:
        """Read a column containing only null values."""
        # Read length (for verification)
        stored_length = struct.unpack('>I', buffer.read(4))[0]
        if stored_length != row_count:
            raise ValueError(f"Length mismatch in null column: expected {row_count}, got {stored_length}")
        
        return np.full(row_count, None, dtype='O')

