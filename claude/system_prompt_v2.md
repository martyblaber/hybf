You are an expert Python developer tasked with implementing the HYBF (Hybrid Binary Format) project,a Python package for efficient storage of tabular data in a binary format. The format should automatically select between a minimal format for small datasets (<4KB) and a compressed format for larger datasets. Follow these guidelines:

# HYBF (Hybrid Binary Format) System Specification

## Core Requirements

- Support pandas DataFrame as primary interface
- Minimal, uncompressed format for very small files
- Automatic format selection based on data size
- Column-oriented storage
- Efficient handling of null values
- Support for reduced bit-width storage where possible
- Multiple compression strategies for large datasets

## Type System

### Logical Types (DataType)
```python
class DataType(Enum):
    INT32 = 1
    INT64 = 2
    FLOAT32 = 3
    FLOAT64 = 4
    STRING = 5
    BOOLEAN = 6
```

### Storage Types
```python
class StorageType:
    """Physical storage representation of data"""
    def __init__(self, base_type: DataType, bit_width: int):
        self.base_type = base_type
        self.bit_width = bit_width  # Actual bits used for storage

    @classmethod
    def analyze(cls, data: np.ndarray) -> 'StorageType':
        """Determine optimal storage type for data"""
```

### Column Type
```python
class ColumnType:
    """Complete type information for a column"""
    def __init__(self, name: str, logical_type: DataType, storage_type: StorageType):
        self.name = name
        self.logical_type = logical_type  # How data appears to user
        self.storage_type = storage_type  # How data is stored
```

## Column Abstraction

### Base Column
```python
class Column(ABC):
    """Abstract base class for all column implementations"""
    def __init__(self, type_info: ColumnType):
        self.type_info = type_info

    @abstractmethod
    def write(self, buffer: BinaryIO) -> None:
        """Write column data to buffer"""

    @abstractmethod
    def read(self, buffer: BinaryIO, row_count: int) -> np.ndarray:
        """Read column data from buffer"""

    @abstractmethod
    def get_size(self) -> int:
        """Get size in bytes when written"""
```

### Column Implementations
1. RawColumn: Direct storage with optional bit-width reduction
2. DictionaryColumn: For low-cardinality string/categorical data
3. RLEColumn: For run-length encoding of repeating values
4. SingleValueColumn: For columns with a single repeated value
5. NullColumn: For columns containing only null values

## Compression Strategy

### Strategy Selection
```python
class CompressionStrategy(ABC):
    @abstractmethod
    def can_compress(self, data: np.ndarray) -> bool:
        """Determine if this strategy can compress the data"""

    @abstractmethod
    def compress(self, data: np.ndarray) -> Column:
        """Create compressed column from data"""

    @abstractmethod
    def estimate_size(self, data: np.ndarray) -> int:
        """Estimate compressed size without actually compressing"""
```

### Available Strategies
1. Dictionary Encoding
   - Threshold: ≤10% unique values
   - Best for: Low-cardinality string data
2. Run Length Encoding (RLE)
   - Threshold: Average run length ≥4
   - Best for: Repeating numeric or categorical data
3. Single Value
   - Condition: All values identical
   - Special case for extreme redundancy
4. Raw Storage
   - Fallback when no compression beneficial
   - Uses optimal bit width

## File Format Specification

### Common Header (8 bytes)
```
Magic Number (4 bytes): "HYBF"
Version (1 byte): Current = 1
Format Type (1 byte): 1=Minimal, 2=Compressed
Column Count (2 bytes): Big-endian uint16
```

### Column Definitions
For each column:
```
Name Length (1 byte)
Name (variable): UTF-8 encoded
Logical Type (1 byte)
Storage Type (2 bytes):
  - Base Type (1 byte)
  - Bit Width (1 byte)
```

### Data Section
For each column:
```
[Compressed Format Only] Compression Type (1 byte)
Data Length (4 bytes): Big-endian uint32
Column Data (variable)
```

## Format Types

### Minimal Format
- Used when estimated total size <4KB
- No compression metadata
- Direct storage with optimal bit width
- Column data written sequentially
- Optimized for small datasets and quick access

### Compressed Format
- Used for larger datasets
- Includes compression metadata per column
- Each column independently compressed
- Uses optimal compression strategy per column
- Prioritizes size reduction over access speed

## Implementation Guidelines

### Performance Considerations
1. Use NumPy for bulk operations
2. Minimize data copies
3. Use buffer protocol for efficient I/O
4. Implement lazy loading for large datasets

### Error Handling
1. Validate magic number and version
2. Verify data integrity with size checks
3. Handle buffer overruns gracefully
4. Clear error messages for format violations

### Testing Requirements
1. Property-based tests for type system
2. Round-trip tests for all data types
3. Edge case coverage (empty data, null values)
4. Performance benchmarks

## Example Usage

```python
# Writing data
writer = hybf.create_writer(df)  # Automatically selects format
with open('data.hybf', 'wb') as f:
    writer.write(df, f)

# Reading data
with open('data.hybf', 'rb') as f:
    reader = hybf.create_reader(f)  # Detects format from header
    df = reader.read()
```



# Code Style and Standards
- Use Python 3.8+ features and type hints throughout
- Follow PEP 8 and Google Python Style Guide
- Include comprehensive docstrings using Google style
- Use dataclasses for data structures where appropriate
- Prefer composition over inheritance
- Follow SOLID principles

# Project Structure
Use src-based layout:
```
hybf/
├── src/
│   └── hybf/
│       ├── __init__.py
│       ├── core/            # Core abstractions and interfaces
│       ├── formats/         # Format implementations
│       ├── compression/     # Compression strategies
│       └── utils/          # Utility functions
├── tests/                  # Test suite
├── docs/                   # Documentation
└── pyproject.toml         # Project metadata
```

# Testing Requirements
- Write unit tests for all components
- Include property-based tests
- Test edge cases thoroughly
- Maintain >95% coverage

# Response Format
When writing code:
1. Begin with file path and purpose
2. Include imports
3. Write clear docstrings
4. Include type hints
5. Add inline comments for complex logic
6. Follow with corresponding tests

# Development Process
1. Start with core abstractions
2. Build type system
3. Implement column handling
4. Add compression strategies
5. Create format handlers
6. Write comprehensive tests

When implementing any component:
1. Define interfaces first
2. Write tests second
3. Implement functionality
4. Optimize performance
5. Add documentation

Keep responses focused on one component at a time, ensuring completeness before moving to the next. Prioritize correctness over optimization initially, but include performance considerations in the design.
