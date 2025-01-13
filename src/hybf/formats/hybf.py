# src/hybf/formats/hybf.py
from typing import BinaryIO, List
import numpy as np
import pandas as pd
from ..core.types import DataType, StorageType, ColumnType
from ..core.columns import Column, RawColumn
from ..compression.strategy import CompressionStrategy, DictionaryStrategy

MAGIC = b'HYBF'
VERSION = 1
MINIMAL_FORMAT = 1
COMPRESSED_FORMAT = 2
SIZE_THRESHOLD = 4096  # 4KB

class HYBFWriter:
    """Writer for HYBF format."""
    
    def __init__(self, compression_strategies: List[CompressionStrategy] = None):
        self.compression_strategies = compression_strategies or [DictionaryStrategy()]

    def write(self, df: pd.DataFrame, buffer: BinaryIO) -> None:
        """Write DataFrame to HYBF format."""
        # Estimate size to choose format
        estimated_size = sum(col.nbytes for col in df.values.T)
        format_type = MINIMAL_FORMAT if estimated_size < SIZE_THRESHOLD else COMPRESSED_FORMAT
        
        # Write header
        buffer.write(MAGIC)
        buffer.write(bytes([VERSION]))
        buffer.write(bytes([format_type]))
        buffer.write(len(df.columns).to_bytes(2, 'big'))
        
        # Write columns
        for col_name, col_data in df.items():
            self._write_column(col_name, col_data, format_type, buffer)

    def _write_column(self, name: str, data: pd.Series, format_type: int, buffer: BinaryIO) -> None:
        np_data = data.to_numpy()
        logical_type = DataType.from_numpy_dtype(np_data.dtype)
        storage_type = StorageType.analyze(np_data)
        col_type = ColumnType(name, logical_type, storage_type)
        
        # Write column metadata
        buffer.write(len(name).to_bytes(1, 'big'))
        buffer.write(name.encode('utf-8'))
        buffer.write(bytes([logical_type.value]))
        buffer.write(bytes([storage_type.base_type.value]))
        buffer.write(bytes([storage_type.bit_width]))
        
        if format_type == COMPRESSED_FORMAT:
            # Try compression strategies
            best_strategy = None
            min_size = float('inf')
            
            for strategy in self.compression_strategies:
                if strategy.can_compress(np_data):
                    size = strategy.estimate_size(np_data)
                    if size < min_size:
                        min_size = size
                        best_strategy = strategy
            
            if best_strategy:
                column = best_strategy.compress(np_data, col_type)
            else:
                column = RawColumn(col_type, np_data)
        else:
            column = RawColumn(col_type, np_data)
            
        # Write column data
        column.write(buffer)

class HYBFReader:
    """Reader for HYBF format."""
    
    def read(self, buffer: BinaryIO) -> pd.DataFrame:
        """Read DataFrame from HYBF format."""
        # Verify header
        magic = buffer.read(4)
        if magic != MAGIC:
            raise ValueError("Invalid HYBF file")
            
        version = int.from_bytes(buffer.read(1), 'big')
        if version != VERSION:
            raise ValueError(f"Unsupported version: {version}")
            
        format_type = int.from_bytes(buffer.read(1), 'big')
        col_count = int.from_bytes(buffer.read(2), 'big')
        
        # Read columns
        columns = {}
        for _ in range(col_count):
            name_len = int.from_bytes(buffer.read(1), 'big')
            name = buffer.read(name_len).decode('utf-8')
            
            logical_type = DataType(int.from_bytes(buffer.read(1), 'big'))
            base_type = DataType(int.from_bytes(buffer.read(1), 'big'))
            bit_width = int.from_bytes(buffer.read(1), 'big')
            
            storage_type = StorageType(base_type, bit_width)
            col_type = ColumnType(name, logical_type, storage_type)
            
            # Read column data
            column = RawColumn(col_type)
            data = column.read(buffer, None)  # row_count not needed for numpy save/load
            columns[name] = data
            
        return pd.DataFrame(columns)