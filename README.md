# HYBF (Hybrid Binary Format)

A Python library for efficient storage of tabular data, optimizing for both small and large datasets. HYBF automatically selects between a minimal format for small files and a compressed format for larger files, providing optimal storage efficiency across different use cases.

## Features

- **Adaptive Format Selection**: Automatically chooses between minimal and compressed formats based on data characteristics
- **Minimal Overhead**: Small files (< 4KB) use a lightweight format with minimal metadata
- **Efficient Compression**: Larger files benefit from column-specific compression strategies:
  - Dictionary encoding for categorical data
  - Run-length encoding for repetitive values
  - Single-value optimization for constant columns
  - Null column optimization
- **Fast Reading**: Optimized for quick loading into pandas DataFrames
- **Python-First**: Designed specifically for use with pandas and numpy

## Installation

```bash
pip install hybf
```

For development installation:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from hybf import FormatFactory

# Writing data
df = pd.DataFrame({
    'id': range(1000),
    'category': ['A', 'B', 'C'] * 333 + ['A'],
    'value': np.random.randn(1000)
})

with open('data.hybf', 'wb') as f:
    writer = FormatFactory.create_writer(df)
    writer.write(df, f)

# Reading data
with open('data.hybf', 'rb') as f:
    reader = FormatFactory.create_reader(f)
    df_read = reader.read(f)
```

## When to Use HYBF

HYBF is particularly useful when:
- You have a mix of small and large tabular datasets
- Your data contains many repeated values
- You need a simple, single-file format
- You want to minimize storage overhead for small files
- Your data is primarily accessed through pandas DataFrames

## Format Details

### Minimal Format
Used for small datasets (< 4KB):
- 8-byte header
- Simple column definitions
- Raw data storage
- No compression overhead

### Compressed Format
Used for larger datasets:
- Column-specific compression
- Dictionary encoding for categorical data
- Run-length encoding for repetitive numeric data
- Special handling for constant and null columns

## Performance

Compared to other formats:
- Small files (1-2 rows): 90% smaller than Feather/Parquet
- Medium files: Comparable to Feather
- Large files with repetitive data: Comparable to Parquet

## Development

Clone the repository:
```bash
git clone https://github.com/martyblaber/hybf.git
cd hybf
```

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/name`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -am 'Add some feature'`)
6. Push to the branch (`git push origin feature/name`)
7. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Created in collaboration with Claude 3.5 Sonnet, Nov 2024 version.
- Inspired by the Apache Arrow project
- Built with pandas and numpy