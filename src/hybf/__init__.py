# src/hybf/__init__.py
from .formats.hybf import HYBFWriter, HYBFReader

def write_dataframe(df, path):
    """Write DataFrame to HYBF file."""
    with open(path, 'wb') as f:
        writer = HYBFWriter()
        writer.write(df, f)

def read_dataframe(path):
    """Read DataFrame from HYBF file."""
    with open(path, 'rb') as f:
        reader = HYBFReader()
        return reader.read(f)

__all__ = ['write_dataframe', 'read_dataframe', 'HYBFWriter', 'HYBFReader']