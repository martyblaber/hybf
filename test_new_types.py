# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:54:13 2025

@author: 512051
"""
import math
from dataclasses import dataclass
from typing import BinaryIO, Any
from io import BytesIO
import pandas as pd
import numpy as np
import struct
from collections.abc import Callable

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
    # Get HYBF type
    hybf_type = HYBFTypes.get_hybf_type_for_object(series)
    
    if not hybf_type.is_int:
        raise TypeError("Series must be of integer type, not {str(series)}")
    
    # Series Offset from 0
    offset = series.min()
    series = series.copy() - offset 
    
    # Calculate bytes needed for offset and bits for values
    # min_bytes is 1 for -120, 2 for -130, 2 for -32768 and 3 for -32769
    min_bytes_needed_for_offset = max(1, math.ceil((1 + math.log2(abs(offset)))/8)) # +1 for sign
    # bytes_needed is *4* for -32769, i.e. int32
    bytes_needed_for_offset = 2**math.ceil(math.log2(min_bytes_needed_for_offset))
    
    # Write metadata: type (5 bits), offset bytes (3 bits)
    write_bits(buffer, 
               [hybf_type.type_id, bytes_needed_for_offset],
               [5, 3])
    
    # Write offset and length
    offset_dtype = HYBFTypes.signed[bytes_needed_for_offset]
    write_struct(buffer, offset, offset_dtype)
    
    print(f"Header len = {buffer.seek(0,2)}")
    
    write_bitpacked_uints(buffer, series)
    
    
def write_bitpacked_uints(buffer: BinaryIO, series:pd.Series) -> None:
    
    encode_nulls = int(series.isnull().values.any())

    if encode_nulls:
        max_val = series.max() + 1
    else:
        max_val = series.max()

    # Check we can fit the data. Unlikely to have 2^32 uniques.
    bits_needed_for_longest_field = max(1, math.ceil(math.log2(max_val))) # +1 for null
    if bits_needed_for_longest_field > 32:
        raise InsufficientBitsError("Use a dictionary to encode the data first")
    
    print(f"Need {bits_needed_for_longest_field} bits to encode int={max_val}")
    
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
    
    # Get types
    hybf_type = HYBFTypes.decode_id(type_id)
    offset_dtype = HYBFTypes.signed[bytes_needed_for_offset]
    
    # Read offset and length
    offset = read_struct(buffer, offset_dtype)
    print("Offset read", offset)
    
    values = read_bitpacked_uints(buffer)
    series = pd.Series(values, dtype=hybf_type.np)
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
    elif hybf_type.fixed_length:
        # For fixed length types, we can write directly, but need to include length.
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

def test_value_list_read_and_write() -> None:
    """Test round-trip encoding and decoding of value lists."""
    test_cases = [
        # Strings
        ["apple", "banana", "cherry"],
        
        # Integers (small)
        [1, 2, 3, 4, 5],
        
        # Integers (large)
        [1000000, 2000000, 3000000],
        
        # Integers (mixed sign)
        [-10, 0, 10, 20],
        
        # Integers with nulls
        [-10, 0, 10, None, 20],
        
        # Single value repeated
        [42] * 100,
        
        # All nulls
        [None] * 5
    ]
    
    for test_case in test_cases:
        buffer = BytesIO()
        
        # Write values
        write_value_list(buffer, test_case)
        
        # Get buffer size
        buffer_size = buffer.tell()
        
        # Reset buffer for reading
        buffer.seek(0)
        
        # Read values
        result = read_value_list(buffer)
        
        # Convert to series for comparison (handles NA values better)
        original = pd.Series(test_case)
        result = pd.Series(result)
        
        # Check if equal
        if not original.equals(result):
            raise AssertionError(
                f"Test failed for {test_case}\n"
                f"Original: {original}\n"
                f"Result: {result}"
            )
            
        print(f"Successfully encoded and decoded {len(test_case)} values "
              f"of type {type(test_case[0])} in {buffer_size} bytes")


#def test_bitpacked_int():
if True:
    
    s1 = pd.Series([50, 100, 1_000_000_000])
    s2 = pd.Series([-50, 50, pd.NA, 0, 0, 0, 1])
    s3 = pd.Series([-3, 3, 0, 0, 0, 1], dtype='Int8')
    for series_in in [s1, s2, s3]:
        dtype, series_in = determine_best_hybf_dtype(series_in)
        if dtype is None:
            raise TypeError("Bad Type")
        buffer = BytesIO()
        buffer.seek(0)
        write_bitpacked_integers(buffer, series_in)    
        buffer.seek(0)
        print(buffer.read())
        print(f"Length = {buffer.seek(0,2)}")
        buffer.seek(0)
        series_out = read_bitpacked_integers(buffer)
        print("Series in\n", series_in)
        print("Series out\n", series_out)
        # buffer = BytesIO()
        # write_varint(buffer, x)
        # buffer_len = buffer.seek(0, 2)
        # buffer.seek(0)
        # xout = read_varint(buffer)
        
        # print(f"In: {x}, nbytes:{buffer_len}, Out: {xout}")


if False:

    
    @dataclass
    class RLEMetadata:
        version: int
        num_runs: int
    
        def encode(self) -> bytes:
            import struct
            out = bytearray()
            # version (2 bytes)
            out += struct.pack("<H", self.version)
            # num_runs (4 bytes)
            out += struct.pack("<I", self.num_runs)
            return bytes(out)
    
        @classmethod
        def decode(cls, data: bytes) -> "RLEMetadata":
            import struct
            version, num_runs = struct.unpack("<HI", data)  # 2 bytes + 4 bytes
            return cls(version, num_runs)
    
    
    
    
    
    @dataclass
    class StepMetadata:
        step_name: str
        version: int
        data: Any  # e.g., the dataclass for dictionary or RLE
    
        def encode(self) -> bytes:
            # Convert step_name to bytes, store the length, version, etc.
            import struct
            
            name_bytes = self.step_name.encode("utf-8")
            block = bytearray()
    
            # Step name length + step name
            block += struct.pack("<I", len(name_bytes))
            block += name_bytes
            
            # Step version (4 bytes)
            block += struct.pack("<I", self.version)
            
            # Then the step-specific metadata
            # We assume `self.data` is something like DictionaryMetadata, which also has an encode() method
            data_bytes = self.data.encode()
            
            # Next, store the length of data_bytes (so we know how much to read)
            block += struct.pack("<I", len(data_bytes))
            block += data_bytes
            
            return bytes(block)
    
        @classmethod
        def decode(cls, data: bytes) -> "StepMetadata":
            import struct
            
            offset = 0
            
            # read step_name
            (name_len,) = struct.unpack_from("<I", data, offset)
            offset += 4
            name_bytes = data[offset:offset+name_len]
            offset += name_len
            step_name = name_bytes.decode("utf-8")
            
            # read version
            (version,) = struct.unpack_from("<I", data, offset)
            offset += 4
            
            # read data_bytes
            (data_len,) = struct.unpack_from("<I", data, offset)
            offset += 4
            data_bytes = data[offset:offset+data_len]
            offset += data_len
            
            # Now we must know *which* metadata class to invoke for decode, based on step_name or version
            if step_name == "DictionaryEncoder":
                meta_obj = DictionaryMetadata.decode(data_bytes)
            elif step_name == "RLEEncoder":
                meta_obj = RLEMetadata.decode(data_bytes)
            else:
                raise ValueError(f"Unknown step: {step_name}")
            
            return cls(step_name, version, meta_obj)
    
    
    
    if True:
        data = pd.Series(['SN_AA1', 'SN_AA1', 'SN_AA1', 'SN_AA2', 'SN_AA2', 'SN_AA2', 'SN_AA2'])
    
        pipeline = PipelineCompressor([
            DictionaryEncoder(),
            RLEEncoder(),
            BitPackEncoder(),
        ])
    
        compressed_bytes, metadatas = pipeline.compress(data)
        
        DictionaryEncoder().compress(data)
        data_c1, metadata_c1 = DictionaryEncoder().compress(data)
        
        # Later, to decompress:
        recovered_data = pipeline.decompress(compressed_bytes, metadatas)
        assert all(data == recovered_data), "Data mismatch!"
    
