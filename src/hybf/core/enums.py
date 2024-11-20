from enum import Enum

class FormatType(Enum):
    """Available format types."""
    MINIMAL = 1
    COMPRESSED = 2

class CompressionType(Enum):
    """Available compression strategies."""
    RAW = 1
    RLE = 2
    DICTIONARY = 3
    SINGLE_VALUE = 4
    NULL = 5
