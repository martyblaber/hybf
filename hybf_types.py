# -*- coding: utf-8 -*-
"""
hybf_types.py

"""
from dataclasses import dataclass
from typing import Type
import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_float_dtype

@dataclass
class HYBF_DType():
    type_id: int
    struct_char: str
    byte_count: int | None
    np: Type | None
    
    @property
    def fixed_length(self) -> bool:
        return self.byte_count is not None

    @property
    def pd(self) -> Type | None:
        return self.np

    @property
    def is_int(self) -> bool:
        return is_integer_dtype(self.np)

    @property
    def is_float(self) -> bool:
        return is_float_dtype(self.np)

    @property
    def is_string(self) -> bool:
        return (self.type_id == 2)
    
    @property
    def is_null(self) -> bool:
        return self.type_id == 10

    @property
    def is_nullable(self) -> bool:
        if self.type_id >= 23:
            return True
        else:
            return False
    
    @property
    def is_basic_int(self) -> bool:
        """Basic ints are not nullable. i.e. int and uint only. Not Int"""
        return (self.is_int and not self.is_nullable)    
     
    @property
    def is_basic(self) -> bool:
        """string, int, uint and float only"""
        return self.is_string or self.is_basic_int or self.is_float

dataframe =     HYBF_DType(0, None, None, pd.DataFrame)
series =        HYBF_DType(1, None, None, pd.Series)

# Strings
string  =        HYBF_DType(2, None, None, np.str_)

# Packed Types: 
varint =        HYBF_DType(3, None, None, None)
bitpackedint =  HYBF_DType(4, None, None, None)

# Other
dictionary   =  HYBF_DType(5, None, None, None)
array        =  HYBF_DType(6, None, None, list)
ndarray      =  HYBF_DType(7, None, None, np.ndarray)
int_tuples   =  HYBF_DType(8, None, None, None)
vc_tuple    =  HYBF_DType(9, None, None, None)
null        = HYBF_DType(10, None, None, None)

# Unsigned integers
uint8 =  HYBF_DType(15, '>B', 1, np.uint8)
uint16 = HYBF_DType(16, '>H', 2, np.uint16)
uint32 = HYBF_DType(17, '>I', 4, np.uint32)
uint64 = HYBF_DType(18, '>Q', 8, np.uint64)    

# Signed integers
int8 =  HYBF_DType(19, '>b', 1, np.int8)
int16 = HYBF_DType(20, '>h', 2, np.int16)
int32 = HYBF_DType(21, '>i', 4, np.int32)
int64 = HYBF_DType(22, '>q', 8, np.int64)

## Nullable Numeric Types

# Floating point
# Save for 8 bit float.
float8  = HYBF_DType(23, None, 1, None)
float16 = HYBF_DType(24, '>e', 2, np.float16)
float32 = HYBF_DType(25, '>f', 4, np.float32)
float64 = HYBF_DType(26, '>d', 8, np.float64)

# Nullable Signed integers
Int8 =  HYBF_DType(27, None, 1, pd.Int8Dtype())
Int16 = HYBF_DType(28, None, 2, pd.Int16Dtype())
Int32 = HYBF_DType(29, None, 4, pd.Int32Dtype())
Int64 = HYBF_DType(30, None, 8, pd.Int64Dtype())


    
class HYBFTypes():
    """ """
    null = null
    dataframe = dataframe
    series = series
    
    string = string    
    
    varint = varint
    bitpackedint = bitpackedint
    
    dictionary = dictionary
    array = array
    ndarray = ndarray
    int_tuples = int_tuples
    vc_tuple = vc_tuple
    
    uint8 = uint8
    uint16 = uint16
    uint32 = uint32
    uint64 = uint64

    int8 = int8
    int16 = int16
    int32 = int32
    int64 = int64
    
    float8 = float8
    float16 = float16
    float32 = float32
    float64 = float64
    
    #Pandas extension types.
    Int8 = Int8
    Int16 = Int16
    Int32 = Int32
    Int64 = Int64
    
    dtypes = [dataframe, series, string,
           varint, bitpackedint, 
           dictionary, 
           array, ndarray, 
           int_tuples, vc_tuple,
           null,
           uint8, uint16, uint32, uint64,
           int8, int16, int32, int64,
           float8, float16, float32, float64,
           Int8, Int16, Int32, Int64]
    
    decoder = {}
    
    unsigned = { 1: uint8, 2: uint16, 4: uint32, 8: uint64}
    signed = { 1: int8, 2: int16, 4: int32, 8: int64}
    null_ints = { 1: Int8, 2: Int16, 4: Int32, 8: Int64}
    floats = { 1: float8, 2: float16, 4: float32, 8: float64}
    
    for hybf_dtype in dtypes:
        decoder[hybf_dtype.type_id] = hybf_dtype
    
    @classmethod
    def decode_id(cls, type_id) -> HYBF_DType:
        return cls.decoder[type_id]

    @classmethod
    def get_hybf_type_for_object(cls, obj: pd.Series | pd.DataFrame) -> HYBF_DType:
        """Map pandas type to HYBF type"""

        #Check for dataframe first - it doesn't have a dtype.
        if isinstance(obj, pd.DataFrame):
            return cls.dataframe

        if is_string_dtype(obj):
            return cls.string
        
        dtype = obj.dtype
        
        # Handle pandas extension dtypes
        if isinstance(dtype, pd.Int8Dtype): return cls.Int8
        if isinstance(dtype, pd.Int16Dtype): return cls.Int16
        if isinstance(dtype, pd.Int32Dtype): return cls.Int32
        if isinstance(dtype, pd.Int64Dtype): return cls.Int64
        
        # Convert numpy dtype to string representation
        dtype_str = str(dtype)
        
        # Handle numpy-backed dtypes
        if dtype_str == 'bool': return cls.uint8
        
        if dtype_str == 'int8': return cls.int8
        if dtype_str == 'int16': return cls.int16
        if dtype_str == 'int32': return cls.int32
        if dtype_str == 'int64': return cls.int64
        
        if dtype_str == 'uint8': return cls.uint8
        if dtype_str == 'uint16': return cls.uint16
        if dtype_str == 'uint32': return cls.uint32
        if dtype_str == 'uint64': return cls.uint64
        
        if dtype_str == 'float16': return cls.float16
        if dtype_str == 'float32': return cls.float32
        if dtype_str == 'float64': return cls.float64
        
        raise ValueError(f"Unsupported pandas dtype: {dtype} ({type(dtype)})")


def determine_best_hybf_dtype(series: pd.Series) -> tuple[HYBF_DType, pd.Series]:
    """
    Determines the most memory-efficient HYBF dtype for a pandas Series while preserving data integrity.
    
    This function analyzes the input Series and attempts to find the most appropriate HYBF dtype by:
    1. Handling string data types
    2. Converting to numeric when possible
    3. Managing null values appropriately
    4. Downcasting integers to the smallest possible size
    5. Downcasting floats to float32 when possible without loss of precision
    
    Args:
        series (pd.Series): Input pandas Series to analyze
        
    Returns:
        tuple[HYBF_DType, pd.Series]: A tuple containing:
            - The determined HYBF dtype
            - A new Series converted to the appropriate type
            
    Raises:
        ValueError: If the series cannot be converted to a supported type
        TypeError: If the series cannot be converted to numeric
        
    Examples:
        >>> s = pd.Series([1.0, 2.0, 3.0])
        >>> dtype, converted = determine_best_hybf_dtype(s)
        >>> print(dtype)  # Will return float32 if values fit precision
        
        >>> s = pd.Series([1, 2, None, 4])
        >>> dtype, converted = determine_best_hybf_dtype(s)
        >>> print(dtype)  # Will return appropriate nullable integer type
    """
    # Create a copy to avoid modifying the original
    series = series.copy()
    
    if len(series)==0:
        return HYBFTypes.null, series
    
    # Handle string dtypes
    if is_string_dtype(series):
        return HYBFTypes.string, series
    
    # Attempt numeric conversion
    try:
        _ = pd.to_numeric(series, errors='raise')
    except (TypeError, ValueError):
        return None, series

    # Check for null values
    has_nulls = series.isna().any()
    
    # Handle all-null series
    if has_nulls and series.isna().all():
        return HYBFTypes.null, series
    
    # Convert non-null values to most efficient numeric type
    # Try unsigned first, then signed if there are negative values
    nn_series = pd.to_numeric(series.dropna(), errors='raise')
    if (nn_series < 0).any():
        nn_series = pd.to_numeric(nn_series, downcast='signed')
    else:
        nn_series = pd.to_numeric(nn_series, downcast='unsigned')
    
    # Get the HYBF type for the non-null series
    nn_dtype = HYBFTypes.get_hybf_type_for_object(nn_series)
    
    # Handle float types
    if nn_dtype.is_float:
        
        # Check if values are within float32 range
        if ((nn_series.abs() <= np.finfo(np.float32).max).all() and 
            (nn_series.abs()[nn_series != 0] >= np.finfo(np.float32).tiny).all()):
            
            # Convert to float32
            float32_series = series.astype(np.float32)
            
            # Check relative and absolute differences
            nn_series_32 = float32_series.dropna()
            abs_diff = np.abs(nn_series - nn_series_32)
            rel_diff = abs_diff / np.abs(nn_series)
            
            # Use relative difference for values above 1e-4, absolute for smaller values
            rel_threshold = 1e-8  # Slightly above 2^-23 to account for rounding
            abs_threshold = 1e-8  # For small numbers, absolute difference matters more
            near_zero = nn_series.abs() < 1e-4
  
            # # Use relative difference for large values, absolute for values near zero
            # threshold = 1e-7  # About 7 decimal digits of precision for float32
            # near_zero = nn_series.abs() < 1e-6
            
            if (
                (rel_diff[~near_zero] < rel_threshold).all() and  # Check relative difference for larger values
                (abs_diff[near_zero] < abs_threshold).all()           # Check absolute difference for values near zero
            ):
                return HYBFTypes.float32, float32_series
                
        # If any check fails, use original float type
        return nn_dtype, series.astype(nn_dtype.pd)
    
    
    # Handle integer types
    if nn_dtype.is_int:
        if has_nulls:
            # Convert to nullable integer type if nulls are present
            new_type = HYBFTypes.null_ints[nn_dtype.byte_count]
            return new_type, series.astype(new_type.pd)
        else:
            return nn_dtype, series.astype(nn_dtype.pd)
            
    raise ValueError(f"Unknown Type {series.dtype}")










# def determine_best_hybf_dtype(series: pd.Series) -> tuple[HYBF_DType, pd.Series]:
    
#     #Don't modify the original
#     series = series.copy()
    
#     if is_string_dtype(series):
#         return HYBFTypes.string, series
    
#     try:
#         _ = pd.to_numeric(series, errors='raise')
#     except TypeError:
#         return None, series

#     has_nulls = series.isna().any()
    
#     if has_nulls:
#         if series.isna().all():
#             return HYBFTypes.null, series
    
#     nn_series = pd.to_numeric(series.dropna(), errors='raise', downcast='unsigned')
    
#     nn_dtype = HYBFTypes.get_hybf_type_for_object(nn_series)
    
#     if nn_dtype.is_float:
#         return nn_dtype, series.astype(nn_dtype.pd)
    
#     if nn_dtype.is_int:
#         if has_nulls:
#             # Translate to type with nulls
#             new_type = HYBFTypes.null_ints[nn_dtype.byte_count]
#             return new_type, series.astype(new_type.pd)
#         else:
#             return nn_dtype, series.astype(nn_dtype.pd)
#     raise ValueError(f"Unknown Type {series.dtype}")


if True and __name__ == "__main__":
    s6 = pd.Series([-50, 50, pd.NA, 0, 0, 0, 1])

    print(f"Type: {s6.dtype}")
    x = determine_best_hybf_dtype(s6)
    
    
if False and __name__ == "__main__":
    s1 = pd.Series([1, 2, 3], dtype='int8')
    s2 = pd.Series([1, 2, 3], dtype='int64')
    s3 = pd.Series([1, 2, None], dtype='Int64')
    s4 = pd.Series([1.0, 2.0], dtype='float32')
    s5 = pd.Series([1.0, 2.0], dtype='float64')
    s6 = pd.Series([-50, 50, pd.NA, 0, 0, 0, 1])
    d1 = pd.DataFrame(s1)
    for o in [s1, s2, s3, s4, s5, s6, d1]:
        print(o, HYBFTypes.get_hybf_type_for_object(o))
    