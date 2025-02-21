# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:54:13 2025

@author: 512051
"""
from pprint import pprint
import math
from enum import Enum
from dataclasses import dataclass
from typing import BinaryIO, Any, Type
from io import BytesIO
import pandas as pd
import numpy as np
import struct
from abc import ABC, abstractmethod
#from collections.abc import abstractmethod

from hybf_types import HYBF_DType, HYBFTypes, determine_best_hybf_dtype

class InsufficientBitsError(ValueError):
    pass


def write_struct(buffer:BytesIO, value:Any, hybf_dtype:HYBF_DType):
    if hybf_dtype.struct_char is None:
        raise ValueError(f"Type hybf_dtype {hybf_dtype} is not directly writable. Encode first")
    if not hybf_dtype.fixed_length:
        raise ValueError(f"Type hybf_dtype {hybf_dtype} doesn't have a fixed length. Encode first")
    out_bytes = struct.pack(hybf_dtype.struct_char, value)
    buffer.write(out_bytes)
    
def read_struct(buffer:BytesIO, hybf_dtype:HYBF_DType) -> Any:
    if hybf_dtype.struct_char is None:
        raise ValueError(f"Type hybf_dtype {hybf_dtype} is not directly readable. Why is it here?")
    if not hybf_dtype.fixed_length:
        raise ValueError(f"Type hybf_dtype {hybf_dtype} doesn't have a fixed length. Decode first")
        
    in_bytes = buffer.read(hybf_dtype.byte_count)
    values = struct.unpack(hybf_dtype.struct_char, in_bytes)
    if len(values) != 1:
        raise ValueError("Wrong number of items returned from struct.unpack({hybf_dtype.struct_char}.. ")
    return values[0]


def read_varint(buffer: BinaryIO) -> int:
    """Read a variable-length integer.
    
    Format: Each byte uses 7 bits for data and 1 bit to indicate if more bytes follow.
    MSB=1 means more bytes follow, MSB=0 means this is the last byte.
    Can encode integers up to 2^64 - 1.
    """
    result = 0
    shift = 0
    while True:
        byte = buffer.read(1)
        if not byte:
            raise EOFError("Unexpected end of buffer during varint read")
            
        byte_val = ord(byte)
        # Add the 7 data bits to our result
        result |= (byte_val & 0x7F) << shift
        # Check if there are more bytes (MSB = 1)
        if not (byte_val & 0x80):
            break
        shift += 7
        if shift > 63:
            raise ValueError("Varint is too long")
    return result

def write_varint(buffer: BinaryIO, value: int) -> None:
    """Write a variable-length integer.
    
    Format: Each byte uses 7 bits for data and 1 bit to indicate if more bytes follow.
    MSB=1 means more bytes follow, MSB=0 means this is the last byte.
    """
    if value < 0:
        raise ValueError("Varint encoding only supports non-negative integers")
    if value > (1 << 64) - 1:
        raise ValueError("Varint value too large")
        
    while value >= 0x80:  # While we need more bytes
        # Write 7 bits of data + MSB=1
        buffer.write(bytes([(value & 0x7F) | 0x80]))
        value >>= 7
    # Write final byte with MSB=0
    buffer.write(bytes([value]))

def test_varint():
    
    for x in [50, 100, 1_000_000_000_000]:
        buffer = BytesIO()
        write_varint(buffer, x)
        buffer_len = buffer.seek(0, 2)
        buffer.seek(0)
        xout = read_varint(buffer)
        
        print(f"In: {x}, nbytes:{buffer_len}, Out: {xout}")

def write_string(buffer: BinaryIO, value: str) -> None:
    val_bytes = value.encode('utf-8')
    # Length of the string
    write_varint(buffer, len(val_bytes))
    # String itself.
    buffer.write(val_bytes)
    
def read_string(buffer: BinaryIO) -> str:
    # Length of the string
    length = read_varint(buffer)
    # String itself.
    value = buffer.read(length).decode('utf-8')
    return value

def write_strings(buffer: BinaryIO, values: list[str]) -> None:
    # Number of strings
    write_varint(buffer, len(values))
    # For each string
    for value in values:
        write_string(buffer, value)
        
def read_strings(buffer: BinaryIO) -> list[str]:    
    string_list =  []
    # Number of strings
    count = read_varint(buffer)
    # For each string
    for i in range(count):
        value = read_string(buffer)
        string_list.append(value)
    return string_list

def read_bits(buffer: BinaryIO, bit_widths: list[int]) -> list[int]:
    """Read multiple fields with specified bit widths.
       Always uses integer number of bytes to store the bits.
   """
    total_bits = sum(bit_widths)
    num_bytes = (total_bits + 7) // 8
    data = buffer.read(num_bytes)
    
    if not data:
        raise EOFError("Attempted to read past end of buffer")
    
    # Convert bytes to generic integer
    value = int.from_bytes(data, byteorder='big')
    
    result = []
    current_pos = 0
    
    for width in bit_widths:
        mask = (1 << width) - 1
        field_value = (value >> (num_bytes * 8 - current_pos - width)) & mask
        result.append(field_value)
        current_pos += width
        
    return result
    
def write_bits(buffer:BinaryIO, values: list[int], bit_widths: list[int]) -> None:
    """Write multiple fields with specified bit widths. 
       Always uses integer number of bytes to store the bits.
   """
    if len(values) != len(bit_widths):
        raise ValueError("Must have same number of values and widths")
        
    total_bits = sum(bit_widths)
    num_bytes = (total_bits + 7) // 8
    
    result = 0
    current_pos = 0
    
    for value, width in zip(values, bit_widths):
        max_value = (1 << width) - 1
        if value > max_value:
            raise ValueError(f"Value {value} exceeds maximum for {width} bits")
        if value < 0:
            raise ValueError("Values must be non-negative")
            
        result |= (value << (num_bytes * 8 - current_pos - width))
        current_pos += width

    out_bytes = result.to_bytes(num_bytes, byteorder='big')
    buffer.write(out_bytes)



def write_bitpacked_integers(buffer: BinaryIO, series: pd.Series):
    """
    Test pd.Series([-10,0,10,pd.NA,20])
    
    """
    series = series.copy()
    
    # Get HYBF type
    hybf_type = HYBFTypes.get_hybf_type_for_object(series)
    
    if not hybf_type.is_int:
        raise TypeError("Series must be of integer type, not {str(series)}")
    
    # Series Offset from 0
    offset = series.min()
    
    if offset != 0:
        series = series - offset 
        
        # Calculate bytes needed for offset and bits for values
        # min_bytes is 1 for -120, 2 for -130, 2 for -32768 and 3 for -32769
        min_bytes_needed_for_offset = max(1, math.ceil((1 + math.log2(abs(offset)))/8)) # +1 for sign
        # bytes_needed is *4* for -32769, i.e. int32
        bytes_needed_for_offset = 2**math.ceil(math.log2(min_bytes_needed_for_offset))
    else:
        bytes_needed_for_offset = 0
        
    # Write metadata: type (5 bits), offset bytes (3 bits)
    write_bits(buffer, 
               [hybf_type.type_id, bytes_needed_for_offset],
               [5, 3])
    
    if bytes_needed_for_offset != 0:
        # Write offset and length
        offset_dtype = HYBFTypes.signed[bytes_needed_for_offset]
        write_struct(buffer, offset, offset_dtype)
        
    write_bitpacked_uints(buffer, series)
    
    
def write_bitpacked_uints(buffer: BinaryIO, series:pd.Series) -> None:
    
    encode_nulls = int(series.isnull().values.any())

    max_val = series.max()

    # Check we can fit the data. Unlikely to have 2^32 uniques.
    # max_val +1 is there to ensure a max_val of 4 requires 3 bits, not 2.
    bits_needed_for_longest_field = max(1, math.ceil(math.log2(max_val+1)))
    if encode_nulls:
        bits_needed_for_longest_field += 1
    if bits_needed_for_longest_field > 32:
        raise InsufficientBitsError("Use a dictionary to encode the data first")
    
    #print(f"Need {bits_needed_for_longest_field} bits to encode int={max_val}")
    
    # Calculate bytes needed for length
    series_len = len(series)

    #Write the number of bits per field and whether we're encoding nulls or not.
    write_bits(buffer, 
               [bits_needed_for_longest_field, encode_nulls],
               [5,1])
    #Write the length
    write_varint(buffer, series_len)

    # Pack values into bits
    current_byte = 0
    bits_in_byte = 0
    
    if encode_nulls:
        null_value = (1 << bits_needed_for_longest_field) - 1  # Use maximum value for null
    
    for value in series:
        # Get the index (using max value for null)
        if encode_nulls and pd.isna(value):
            index = null_value
        else:
            index = value
            
        # Add bits to current byte
        current_byte = (current_byte << bits_needed_for_longest_field) | index
        bits_in_byte += bits_needed_for_longest_field
        
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


def read_bitpacked_integers(buffer: BinaryIO) -> pd.Series:
    """Read a series of integers that were packed into bits."""
    # Read metadata
    [type_id, bytes_needed_for_offset] = \
        read_bits(buffer, [5, 3])
    
    if bytes_needed_for_offset != 0:
        # Get types
        hybf_type = HYBFTypes.decode_id(type_id)
        offset_dtype = HYBFTypes.signed[bytes_needed_for_offset]
        
        # Read offset and length
        offset = read_struct(buffer, offset_dtype)
        print("Offset read", offset)
    else:
        offset = 0
        print("No Offset")
        
    values = read_bitpacked_uints(buffer)
    series = pd.Series(values)
    
    if offset:
        series += offset
    
    return series
    
    
def read_bitpacked_uints(buffer: BinaryIO) -> pd.Series:
    
    # How many bits per entry? 
    [bits_needed_for_longest_field, encode_nulls] = read_bits(buffer, [5, 1])

    # Read series length
    series_len = read_varint(buffer)

    # Calculate how many bytes we need to read for the values
    total_value_bits = series_len * bits_needed_for_longest_field
    value_bytes_needed = (total_value_bits + 7) // 8

    # Read exactly the number of bytes we need.
    data = buffer.read(value_bytes_needed)
    
    # Define the null value if needed.
    if encode_nulls:
        null_value = (1 << bits_needed_for_longest_field) - 1
    else:
        null_value = None
        
    values = []
    
    current_value = 0
    available_bits = 0
    byte_pos = 0
    
    while len(values) < series_len:
        while available_bits < bits_needed_for_longest_field and byte_pos < len(data):
            current_value = (current_value << 8) | data[byte_pos]
            available_bits += 8
            byte_pos += 1
            
        if available_bits >= bits_needed_for_longest_field:
            mask = (1 << bits_needed_for_longest_field) - 1
            value = (current_value >> (available_bits - bits_needed_for_longest_field)) & mask
            current_value &= (1 << (available_bits - bits_needed_for_longest_field)) - 1
            available_bits -= bits_needed_for_longest_field
            
            if encode_nulls & (value == null_value):
                values.append(pd.NA)
            else:
                values.append(value)
    
    # Round the number of bits up to the closest pandas dtype. 
    if encode_nulls:
        data_bits = bits_needed_for_longest_field - 1
    else:
        data_bits = bits_needed_for_longest_field

    dtype_bytes = 2**math.ceil(math.log2(max(1, math.ceil(data_bits/8))))
    if encode_nulls:
        hybf_type = HYBFTypes.null_ints[dtype_bytes]
    else:
        hybf_type = HYBFTypes.signed[dtype_bytes]

    series = pd.Series(values, dtype=hybf_type.pd)

    return series


def encode_runs(series: pd.Series) -> list[tuple[Any, int]]:
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

def decode_runs(runs:list[tuple[Any, int]]) -> pd.Series:
    """Convert run-length encoding back into a series."""
    values = []    
    for run in runs:
        value, count = run
        values.extend([value] * count)
    return pd.Series(values)

def write_int_tuples(buffer: BinaryIO, tuples: list[tuple[int, int]]) -> None:
    """Write a list of (value, count) tuples to the buffer.
    
    Args:
        buffer: Binary buffer to write to
        tuples: List of (value, count) tuples from run-length encoding
        
    The format is:
    - Split tuples into separate value and count arrays
    - Write values using bitpacked integers
    - Write counts using bitpacked unsigned integers
    """
    if not tuples:
        raise ValueError("Empty tuple list")
        
    # Split tuples into separate arrays
    values = [t[0] for t in tuples]
    counts = [t[1] for t in tuples]
    
    # Convert to series for writing
    value_series = pd.Series(values)
    count_series = pd.Series(counts)
    
    # Write the values using bitpacked integers
    write_bitpacked_integers(buffer, value_series)
    
    # Write the counts using bitpacked unsigned integers
    # Counts are always positive so we can use unsigned
    write_bitpacked_uints(buffer, count_series)

def read_int_tuples(buffer: BinaryIO) -> list[tuple[int, int]]:
    """Read a list of (value, count) tuples from the buffer.
    
    Args:
        buffer: Binary buffer to read from
        
    Returns:
        List of (value, count) tuples for run-length decoding
        
    The format matches write_int_tuples:
    - Read values using bitpacked integers
    - Read counts using bitpacked unsigned integers
    - Combine into tuples
    """
    # Read the values
    values = read_bitpacked_integers(buffer)
    
    # Read the counts
    counts = read_bitpacked_uints(buffer)
    
    # Combine into tuples
    tuples = list(zip(values, counts))
    
    return tuples
    

def write_vc_tuple(buffer: BinaryIO, vc_tuple: tuple[Any, int]) -> None:
    """Write a value-count tuple to the buffer.
    
    Args:
        buffer: Binary buffer to write to
        vc_tuple: A (value, count) tuple from run-length encoding
        
    The format is:
    - Write type_id (5 bits) using write_bits
    - Write count using write_varint
    - Write value using appropriate writer for the type
    """
    value, count = vc_tuple
    
    # Convert value to series to get type
    series = pd.Series([value])
    hybf_type, series = determine_best_hybf_dtype(series)
    
    if hybf_type is None:
        raise ValueError(f"Unsupported value type: {type(value)}")
    
    # Write the type ID (5 bits)
    write_bits(buffer, [hybf_type.type_id], [5])
    
    # Write the count
    write_varint(buffer, count)
    
    # Write the value based on its type
    if hybf_type == HYBFTypes.string:
        write_string(buffer, value)
    elif hybf_type.fixed_length:
        if hybf_type.is_nullable and hybf_type.is_int:
            non_nullable_type = HYBFTypes.signed(hybf_type.byte_count)
            non_nullable_value = value.astype(non_nullable_type.pd)
            write_struct(buffer, non_nullable_value, non_nullable_type)
        else:
            write_struct(buffer, value, hybf_type)
    elif hybf_type.is_null:
        pass #Nothing to write
    else:
        raise ValueError(f"Unsupported HYBF type: {hybf_type}")

def read_vc_tuple(buffer: BinaryIO) -> tuple[Any, int]:
    """Read a value-count tuple from the buffer.
    
    Args:
        buffer: Binary buffer to read from
        
    Returns:
        A (value, count) tuple for run-length decoding
        
    The format matches write_vc_tuple:
    - Read type_id (5 bits)
    - Read count using read_varint
    - Read value using appropriate reader for the type
    """
    # Read the type ID
    [type_id] = read_bits(buffer, [5])
    hybf_type = HYBFTypes.decode_id(type_id)
    
    # Read the count
    count = read_varint(buffer)
    
    # Read the value based on its type
    if hybf_type == HYBFTypes.string:
        value = read_string(buffer)
    elif hybf_type.fixed_length:
        value = read_struct(buffer, hybf_type)
    elif hybf_type.is_null:
        value = pd.NA
    else:
        raise ValueError(f"Unsupported HYBF type: {hybf_type}")
        
    return (value, count)


def write_value_list(buffer: BinaryIO, values: list[Any]) -> None:
    """Write a list of values of the same type to a binary buffer.
    
    Args:
        buffer: Binary buffer to write to
        values: List of values (all same type)
        
    The format is:
    - Convert list to series to determine type
    - Write type_id (5 bits)
    - Write length using varint
    - Write values using most efficient method for the type
    """
    if not values:
        raise ValueError("Empty value list")
        
    # Convert to series to get type
    series = pd.Series(values)
    hybf_type, series = determine_best_hybf_dtype(series)
    
    if hybf_type is None:
        raise ValueError(f"Unsupported value type: {type(values[0])}")
    
    # Write the type ID (5 bits)
    write_bits(buffer, [hybf_type.type_id], [5])
    
    # Write values based on type
    if hybf_type == HYBFTypes.string:
        write_strings(buffer, values)
    elif hybf_type.is_int:
        # For integers, use bit packing for efficiency
        write_bitpacked_integers(buffer, series)
    elif hybf_type.is_float:
        # For floats
        # Write the length
        write_varint(buffer, len(values))
        for value in values:
            write_struct(buffer, value, hybf_type)
    elif hybf_type.is_null:
        pass  # Nothing to write for null values
    else:
        raise ValueError(f"Unsupported HYBF type: {hybf_type}")

def read_value_list(buffer: BinaryIO) -> list[Any]:
    """Read a list of values from a binary buffer.
    
    Args:
        buffer: Binary buffer to read from
        
    Returns:
        List of values of the same type
        
    The format matches write_value_list:
    - Read type_id (5 bits)
    - Read length using varint
    - Read values using appropriate method for the type
    """
    # Read the type ID
    [type_id] = read_bits(buffer, [5])
    hybf_type = HYBFTypes.decode_id(type_id)
    
    # Read the length
    length = read_varint(buffer)
    
    # Read values based on type
    if hybf_type == HYBFTypes.string:
        return read_strings(buffer)
    elif hybf_type.fixed_length:
        values = []
        for _ in range(length):
            value = read_struct(buffer, hybf_type)
            values.append(value)
        return values
    elif hybf_type.is_int:
        series = read_bitpacked_integers(buffer)
        return series.tolist()
    elif hybf_type.is_null:
        return [pd.NA] * length
    else:
        raise ValueError(f"Unsupported HYBF type: {hybf_type}")

def get_series_size_info(series: pd.Series, label: str) -> dict[str, int]:
    
    buffer = BytesIO()
    series = series.copy()

    series_length = len(series)
    
    size_info = {"label": label,
                 "length": series_length,
                 "bytes": None,
                 "pd type": str(series.dtype),
                 "hybf_type": None}    
 
    write_varint(buffer, series_length)

    if series_length == 0:
        size_info["bytes"] = buffer.seek(0,2)
        return size_info
        
    if isinstance(series[0],tuple):
        values = [t[0] for t in series]
        counts = [t[1] for t in series]
        series = pd.Series(values+counts)

    hybf_type, dummy = determine_best_hybf_dtype(series)
    size_info["hybf_type"] = str(hybf_type)
    
    # Write values based on type
    if hybf_type == HYBFTypes.string:
        write_strings(buffer, series.tolist())
        size_info["bytes"] = buffer.seek(0,2)
        return size_info
    
    if hybf_type.is_int:
        if hybf_type.is_nullable:
            #Convert type to non nullable
            hybf_type = HYBFTypes.signed[hybf_type.byte_count]
            #simulate null bitmap
            series = series.replace(pd.NA, 1)
            simulate_null_bitmap = 'A'*(series_length // 8 + 1)
            buffer.write(simulate_null_bitmap.encode('utf-8'))
    
    #reset the type from stuff above.
    hybf_type = HYBFTypes.get_hybf_type_for_object(series)
    size_info["hybf_type"] = str(hybf_type)
    
    for value in series:
        print(value)
        write_struct(buffer, value, hybf_type)
   
    return {"length": series_length,
            "bytes": buffer.seek(0,2),
            "pd type": str(series.dtype),
            "hybf_type": str(hybf_type)}    

def write_basic(buffer: BinaryIO, hybf_type: HYBF_DType, value:Any) -> None:
    """Write a value to a binary buffer.
    
    Args:
        buffer: Binary buffer to write to
        hybf_type: HYBF_DType to write. Only basic types are supported:
            string, float, int, uint. Not Int.
        value: the thing to encode and write.
    
    Raises:
        TypeError: If hybf_type has unsupported type
    """
    if not hybf_type.is_basic:
        raise TypeError(f"hybf_type must be basic(str,int,uint,float). Not {hybf_type}")
    
    if hybf_type.is_string:
        write_string(buffer, value)
    else:
        write_struct(buffer, value, hybf_type)
    
def read_basic(buffer: BinaryIO, hybf_type: HYBF_DType) -> Any:
    """Read a basic hybf_Type from a binary buffer.
    
    Args:
        buffer: Binary buffer to write to
        hybf_type: HYBF_DType to write. Only basic types are supported:
            string, float, int, uint. Not Int.
    
    Raises:
        TypeError: If hybf_type has unsupported type
    """
    if not hybf_type.is_basic:
        raise TypeError(f"hybf_type must be basic(str,int,uint,float). Not {hybf_type}")
    
    if hybf_type.is_string:
        value = read_string(buffer)
    else:
        value = read_struct(buffer, hybf_type)
    
    return value
    
    

def write_series(buffer: BinaryIO, series: pd.Series) -> None:
    """Write a pandas Series to a binary buffer.
    
    Args:
        buffer: Binary buffer to write to
        series: Pandas Series to write
        
    The format is:
    - Write type_id (5 bits)
    - Write length using varint
    - Write values using most efficient method for the type
    
    Raises:
        ValueError: If series is empty or has unsupported type
    """
        
    # Determine most efficient type
    hybf_type, series = determine_best_hybf_dtype(series)
    
    if hybf_type is None:
        raise ValueError(f"Unsupported series type: {series.dtype}")
    
    # Write the type ID (5 bits)
    write_bits(buffer, [hybf_type.type_id], [5])
    
    #No need to write_nothing
    if hybf_type.is_null:
        return
    
    # Write values based on type
    if hybf_type.is_string:
        write_strings(buffer, series.tolist())
    elif hybf_type.is_int:
        # For integers, use bit packing for efficiency
        write_bitpacked_integers(buffer, series)
    elif hybf_type.is_float:
        # For fixed length types, write length then values
        write_varint(buffer, len(series))
        for value in series:
            write_struct(buffer, value, hybf_type)
    else:
        raise ValueError(f"Unsupported HYBF type: {hybf_type}")

def read_series(buffer: BinaryIO) -> pd.Series:
    """Read a pandas Series from a binary buffer.
    
    Args:
        buffer: Binary buffer to read from
        
    Returns:
        Pandas Series
        
    The format matches write_series:
    - Read type_id (5 bits)
    - Read length using varint
    - Read values using appropriate method for the type
    
    Raises:
        ValueError: If unsupported type is encountered
    """
    # Read the type ID
    [type_id] = read_bits(buffer, [5])
    hybf_type = HYBFTypes.decode_id(type_id)
    
    if hybf_type == HYBFTypes.null:
        return pd.Series([])
    elif hybf_type == HYBFTypes.string:
        values = read_strings(buffer)
        return pd.Series(values, dtype=hybf_type.pd)
    elif hybf_type.is_int:
        return read_bitpacked_integers(buffer)
    elif hybf_type.is_float:
        # Read length
        length = read_varint(buffer)
        values = []
        
        for _ in range(length):
            value = read_struct(buffer, hybf_type)
            values.append(value)
                
        return pd.Series(values, dtype=hybf_type.pd)
    else:
        raise ValueError(f"Unsupported HYBF type: {hybf_type}")

###############################################################################
# Compressor Interface (for reference)
###############################################################################
class Metadata(ABC):
    """
    Abstract base class for metadata
    """
    
class Compressor(ABC):
    """
    Abstract base class defining the interface for compression steps.
    """

    @abstractmethod
    def compress(self, series: pd.Series) -> tuple[pd.Series, Metadata]:
        """
        Compress the input data (a Pandas Series) into an in-memory representation
        (e.g., np.ndarray) plus a metadata object needed to decompress.
        """
        pass

    @abstractmethod
    def decompress(self, compressed_series: pd.Series, metadata: Metadata) -> pd.Series:
        """
        Decompress from the given in-memory representation (pandas Series) using
        the provided metadata, returning the original data as a Pandas Series.
        """
        pass

class Writer(ABC):
    """
    Abstract base class defining the interface for writing and reading data.
    The end of a Pipeline is always a Writer.
    """
    @abstractmethod
    def write(self, buffer: BinaryIO, series: pd.Series) -> None:
        """
        Write the input data (a Pandas Series) into a buffer using some binary encoding.
        """
        pass

    @abstractmethod
    def read(self, buffer: BinaryIO) -> pd.Series:
        """
        Read a Pandas Series from the buffer.
        """
        pass

###############################################################################
# DictionaryMetadata
###############################################################################

@dataclass
class DictionaryMetadata(Metadata):
    """
    Holds the reverse mapping (id -> original value) and a version number.
    For example, if we have id_to_val = ["SN_AA1", "SN_AA2"], 
    then ID 0 corresponds to "SN_AA1", and ID 1 corresponds to "SN_AA2".
    """
    id_to_val: pd.Series

    def encode(self) -> bytes:
        """
        Serialize this metadata to a custom binary format:
        
        Layout:
          - 1 Series: id_to_val
        """
        buffer = BytesIO()
        write_series(buffer, self.id_to_val)
        buffer.seek(0)
        return buffer.read()

    @classmethod
    def decode(cls, buffer: BinaryIO) -> "DictionaryMetadata":
        """
        Inverse of encode(): read the series, 
        and reconstruct the metadata.
        """
        id_to_val = read_series(buffer)

        return cls(id_to_val=id_to_val)

###############################################################################
# DictionaryEncoder
###############################################################################

class DictionaryEncoder(Compressor):
    """
    Maps each unique value in the column to an integer ID.
    The output is an integer np.ndarray and the metadata 
    is DictionaryMetadata (including id->value mapping).
    """
    @classmethod
    def compress(cls, series: pd.Series) -> tuple[pd.Series, DictionaryMetadata]:
        # 1) extract unique values
        unique_vals = series.unique()
        # create val -> ID map
        val_to_id = {val: i for i, val in enumerate(unique_vals)}

        # 2) encode data to integer series
        compressed_series = pd.Series([val_to_id[val] for val in series])

        # 3) prepare metadata that allows us to decode
        #    "id_to_val" is simply the list of unique values in the same order
        #    that we assigned IDs in the dictionary
        id_to_val = pd.Series(list(unique_vals))
        metadata = DictionaryMetadata(id_to_val=id_to_val)

        return compressed_series, metadata

    @classmethod
    def decompress(self, compressed_series: pd.Series, metadata: DictionaryMetadata) -> pd.Series:
        # We have an integer array of IDs, so map them back to strings
        # ID i corresponds to metadata.id_to_val[i]
        id_to_val = metadata.id_to_val
        
        # build the original values
        # compressed_series is e.g. np.array([0, 0, 1, 1, 0, ...])
        # we do: for each ID, get the original string
        series = pd.Series([id_to_val[i] for i in compressed_series])
        return series

###############################################################################
# RLEMetadata
###############################################################################

@dataclass
class RLEMetadata:
    """
    Holds minimal metadata for RLE-encoded data.
    - version: a format version, allowing you to evolve the RLE format over time.
    - num_runs: how many run-length pairs are in the compressed output.
    """

    def encode(self) -> bytes:
        """
            RLE Doesn't have any
        """
        return b''

    @classmethod
    def decode(cls, buffer: BinaryIO) -> "RLEMetadata":
        """
        Inverse of encode(). Does Nothing.
        """
        return cls()

###############################################################################
# RLEEncoder
###############################################################################


class RLEEncoder(Compressor):
    """
    Encodes a 1D array of Strings or ints using run-length encoding.
    Output (compressed_series) is a series of tuples.
    where each row is (value, run_length).
    """
    @classmethod
    def compress(cls, series: np.ndarray) -> tuple[pd.Series, RLEMetadata]:
        """
        series: a 1D NumPy array of integers (e.g. from DictionaryEncoder).
        Returns:
          - compressed_series: a 2D array of shape (num_runs, 2).
            Each row is [value, run_length].
          - RLEMetadata: metadata (empty).
        """
        runs = encode_runs(series)
        compressed_series = pd.Series(runs)
        metadata = RLEMetadata()
        return compressed_series, metadata

    @classmethod
    def decompress(cls, compressed_series: np.ndarray, metadata: RLEMetadata) -> np.ndarray:
        """
        compressed_series: series of (value, run_length).
        metadata: RLEMetadata, empty
        Returns the full decompressed series.
        """
        series = pd.Series(decode_runs(list(compressed_series)))
        return series
    
    
class IntTupleWriter(Writer):
    """
    Writes tuples of integers (probably from RLEEncoder.
    """
    @classmethod
    def write(cls, buffer: BinaryIO, series: pd.Series) -> None:
        """
        Write the tuples to the buffer.
        """
        tuples = list(series)
        write_int_tuples(buffer, tuples)
    
    @classmethod
    def read(cls, buffer: BinaryIO) -> pd.Series:
        """
        Read a Pandas Series full of tuples of ints
        """
        tuples = read_int_tuples(buffer)
        return pd.Series(tuples)
    
class ValueCountTupleWriter(Writer):
    """
    Writes tuples of (Any, int) (probably from RLEEncoder).
    """
    @classmethod
    def write(cls, buffer: BinaryIO, series: pd.Series) -> None:
        """
        Write the tuples to the buffer.
        """
        tuples = list(series)
        write_vc_tuples(buffer, tuples)
    
    @classmethod
    def read(cls, buffer: BinaryIO) -> pd.Series:
        """
        Read a Pandas Series full of tuples of ints
        """
        tuples = read_vc_tuples(buffer)
        return pd.Series(tuples)

    

class StepID(Enum):
    DICT_COMPRESSOR = 1
    RLE_COMPRESSOR = 2
    
    INT_TUPLE_WRITER = 100
    VALUE_COUNT_TUPLE_WRITER = 101
    BITPACK_WRITER = 102

# Map from the enum to the actual metadata class
STEP_CLASSES: dict[StepID, tuple[Compressor, Metadata] | tuple[Writer]] = {
    StepID.DICT_COMPRESSOR: (DictionaryEncoder, DictionaryMetadata, None),
    StepID.RLE_COMPRESSOR: (RLEEncoder, RLEMetadata, None),
    StepID.BITPACK_WRITER: (None, None, IntTupleWriter),
}

@dataclass
class Step:
    step_id: StepID
    compressor: Compressor | None
    metadata: Metadata | None
    writer: Writer | None
    
    @classmethod
    def from_id(cls, step_id: StepID) -> "Step":
        compressor, metadata, writer = STEP_CLASSES[step_id]
        return cls(step_id=step_id, compressor=compressor, metadata=metadata, writer=writer)

    @property
    def is_writer(self):
        return self.writer is not None
    
    @property
    def is_reader(self):
        return self.writer is not None
    
    @property
    def is_compressor(self):
        return self.compressor is not None
    
    
if True:
    # 1) Some sample data with repeated values
    series = pd.Series([
        "SN_AA1", "SN_AA1", "SN_AA1", 
        "SN_AA2", "SN_AA2", "SN_AA2", "SN_AA2",
        "SN_AA1", "SN_AA1"
    ])

    step_ids = [StepID.DICT_COMPRESSOR, StepID.RLE_COMPRESSOR, StepID.BITPACK_WRITER]

    size_tracking = []
    size_info = get_series_size_info(series,"Start")
    print(size_info)
    size_tracking.append(size_info)

    buffer = BytesIO()
    md_nbytes = 0
    metadata_list = []

    for step_id in step_ids:
        #step_id = step_ids[0]
        step = Step.from_id(step_id)
        
        if step.is_compressor:
            series, metadata = step.compressor.compress(series)
            metadata_list.append(metadata)
            md_bytes = metadata.encode()
            md_nbytes = len(md_bytes)
            buffer.write(md_bytes)
            size_info = get_series_size_info(series,step.step_id)
            size_info['metadata_bytes'] = md_nbytes
            size_info['size'] = size_info['bytes'] + size_info['metadata_bytes']
            pprint(size_info)
            size_tracking.append(size_info)
   
        if step.is_writer:
           step.writer.write(buffer, series)
           byte_count = buffer.seek(0,2)
           size_info = {'size': byte_count}
           pprint(size_info)
           size_tracking.append(size_info)
     
           
    buffer.seek(0)
    steps = []
    metadatas = []
    for step_id in step_ids:
        #step_id = step_ids[0]
        step = Step.from_id(step_id)
        
        if step.is_compressor:
            metadatas.append(step.metadata.decode(buffer))
        
        if step.is_reader:
            metadatas.append(None) #Just to keep the step_ids and metadatas the same length.
            
    for step_id, metadata in zip(reversed(step_ids),reversed(metadatas)):
        step = Step.from_id(step_id)
        if step.is_reader:
            out_series = step.writer.read(buffer)
        if step.is_compressor:
            out_series = step.compressor.decompress(out_series, metadata)
        print(step_id)
        print(out_series)


if False:
   # ---------------------------
    #  Step 1: Dictionary Encode
    # ---------------------------
    dict_encoder = DictionaryEncoder
    dict_encoded_array, dict_meta = dict_encoder.compress(data)
    print("\n--- Dictionary Encoding ---")
    print("Dictionary-Encoded Array:", dict_encoded_array)
    print("Dictionary Metadata:", dict_meta)

    # --------------------
    #  Step 2: RLE Encode
    # --------------------
    rle_encoder = RLEEncoder(version=1)
    rle_encoded_array, rle_meta = rle_encoder.compress(dict_encoded_array)
    print("\n--- RLE Encoding ---")
    print("RLE-Encoded Pairs:\n", rle_encoded_array)
    print("RLE Metadata:", rle_meta)    

    # -----------------------
    #  Step 3: BitPack Encode
    # -----------------------
    bitpack_encoder = BitPackEncoder(version=1)
    bitpack_bytes, bitpack_meta = bitpack_encoder.compress(rle_encoded_array)
    print("\n--- BitPack Encoding ---")
    print("Bit-packed bytes (hex):", bitpack_bytes.hex())
    print("BitPack Metadata:", bitpack_meta)

    # =====================================================================
    # DECOMPRESS in Reverse Order: BitPack -> RLE -> Dictionary
    # =====================================================================

    # 1) BitPack decode
    rle_decoded_array = bitpack_encoder.decompress(bitpack_bytes, bitpack_meta)
    
    # 2) RLE decode
    dict_decoded_array = rle_encoder.decompress(rle_decoded_array, rle_meta)

    # 3) Dictionary decode
    recovered_series = dict_encoder.decompress(dict_decoded_array, dict_meta)

    # Display final result
    print("\n--- Final Decompression ---")
    print("Recovered Series:", recovered_series.tolist())

    # Validate correctness
    assert all(recovered_series == data), "Mismatch after round-trip compression!"
    print("Success: data round-trip confirmed!")

    
    
    # data = pd.Series(['SN_AA1', 'SN_AA1', 'SN_AA1', 'SN_AA2', 'SN_AA2', 'SN_AA2', 'SN_AA2'])

    # pipeline = PipelineCompressor([
    #     DictionaryEncoder(),
    #     RLEEncoder(),
    #     BitPackEncoder(),
    # ])

    # compressed_bytes, metadatas = pipeline.compress(data)
    
    # DictionaryEncoder().compress(data)
    # data_c1, metadata_c1 = DictionaryEncoder().compress(data)
    
    # # Later, to decompress:
    # recovered_data = pipeline.decompress(compressed_bytes, metadatas)
    # assert all(data == recovered_data), "Data mismatch!"

