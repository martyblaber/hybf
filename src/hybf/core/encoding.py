"""/hybf/src/hybf/core/encoding.py
Special Writer for BitPackedDictionaryWriter. Maybe needs to be in a new file like /hybf/src/hybf/formats/dictionary.py?
"""
import math
from typing import BinaryIO, Dict
import struct
import pandas as pd
import numpy as np

class BitPackedDictionaryWriter:
    """Writes dictionary-encoded data using minimum necessary bits per value."""
    
    def write_dictionary(self, buffer: BinaryIO, series: pd.Series, value_dict: Dict) -> None:
        """Write dictionary-encoded column using minimum bits per value.
        
        Args:
            buffer: Binary buffer to write to
            series: Data to encode
            value_dict: Dictionary mapping indexes to values
        """
        dict_size = len(value_dict)
        bits_needed = max(1, math.ceil(math.log2(dict_size + 1)))  # +1 for null value
        
        # Write dictionary metadata
        buffer.write(struct.pack('>HB', dict_size, bits_needed))
        reverse_dict = {v: k for k, v in value_dict.items()}
        
        # Write dictionary values
        for value in value_dict.values():
            val_bytes = str(value).encode('utf-8')
            buffer.write(struct.pack('B', len(val_bytes)))
            buffer.write(val_bytes)
        
        # Pack values into bits
        current_byte = 0
        bits_in_byte = 0
        null_value = (1 << bits_needed) - 1  # Use maximum value for null
        
        for value in series:
            # Get the index (using max value for null)
            if pd.isna(value):
                index = null_value
            else:
                index = reverse_dict[value]
                
            # Add bits to current byte
            current_byte = (current_byte << bits_needed) | index
            bits_in_byte += bits_needed
            
            # Write complete bytes
            while bits_in_byte >= 8:
                out_byte = current_byte >> (bits_in_byte - 8)
                buffer.write(bytes([out_byte & 0xFF]))
                current_byte &= (1 << (bits_in_byte - 8)) - 1
                bits_in_byte -= 8
        
        # Write final partial byte if any
        if bits_in_byte > 0:
            # Left-align remaining bits in final byte
            current_byte = current_byte << (8 - bits_in_byte)
            buffer.write(bytes([current_byte & 0xFF]))

class BitPackedDictionaryReader:
    """Reads dictionary-encoded data that uses minimum bits per value."""
    
    def read_dictionary(self, buffer: BinaryIO, row_count: int) -> np.ndarray:
        """Read dictionary-encoded column using minimum bits per value.
        
        Args:
            buffer: Binary buffer to read from
            row_count: Number of rows to read
            
        Returns:
            numpy array containing decoded values
        """
        # Read dictionary metadata
        dict_size, bits_needed = struct.unpack('>HB', buffer.read(3))
        
        # Read dictionary values
        value_dict = {}
        for i in range(dict_size):
            length = struct.unpack('B', buffer.read(1))[0]
            value = buffer.read(length).decode('utf-8')
            value_dict[i] = value
        
        # Calculate total bytes needed
        total_bits = row_count * bits_needed
        total_bytes = (total_bits + 7) // 8
        
        # Read all packed bytes
        packed_data = buffer.read(total_bytes)
        
        # Unpack values
        values = []
        current_byte = 0
        bits_in_byte = 0
        null_value = (1 << bits_needed) - 1
        mask = (1 << bits_needed) - 1
        
        byte_index = 0
        for _ in range(row_count):
            # Ensure we have enough bits
            while bits_in_byte < bits_needed and byte_index < len(packed_data):
                current_byte = (current_byte << 8) | packed_data[byte_index]
                bits_in_byte += 8
                byte_index += 1
            
            # Extract value
            if bits_in_byte < bits_needed:
                # Handle partial byte at end if needed
                shift = bits_in_byte - bits_needed
                if shift < 0:
                    current_byte = current_byte << -shift
                    shift = 0
            else:
                shift = bits_in_byte - bits_needed
                
            index = (current_byte >> shift) & mask
            
            # Update bit buffer
            current_byte &= (1 << shift) - 1
            bits_in_byte = shift
            
            # Convert index to value
            values.append(None if index == null_value else value_dict[index])
        
        return np.array(values, dtype='O')