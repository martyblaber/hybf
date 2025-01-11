# HYBF (Hybrid Binary Format) Project Analysis and Roadmap

## Current Architecture

### Core Components
- File Format: 8-byte header (magic number, version, format type)
- Two main format types: Minimal (<4KB) and Compressed
- Column-based storage with type information
- Multiple compression strategies (RLE, Dictionary, Single Value)

### Key Issues

1. **Inconsistent Type Handling**
   - Multiple implementations of bit-width reduction
   - Conflict between raw.py and dtypes.py type definitions
   - BitPackedDictionaryWriter placed awkwardly in core/encoding.py

2. **Compression Strategy Organization**
   - Compression logic spread across multiple files
   - No unified approach to column handling
   - Complex inheritance structure in compressed format

## Proposed Architecture

### Core Changes

1. **New Type System**
   ```python
   class StorageType:
       """Represents the actual storage format of data"""
       def __init__(self, base_type: DataType, bit_width: int)
   
   class ColumnType:
       """Represents the full type information for a column"""
       def __init__(self, logical_type: DataType, storage_type: StorageType)
   ```

2. **Column Abstraction**
   ```python
   class Column:
       """Base class for all column implementations"""
       def __init__(self, name: str, type_info: ColumnType)
       def write(self, buffer: BinaryIO)
       def read(self, buffer: BinaryIO) -> np.ndarray
   ```

3. **Compression Strategy Pattern**
   ```python
   class CompressionStrategy:
       """Base class for compression implementations"""
       def compress(self, data: np.ndarray) -> Column
       def can_compress(self, data: np.ndarray) -> bool
   ```

## Todo List

### Phase 1: Type System Refactoring
1. Create new type system classes (StorageType, ColumnType)
2. Implement type analysis system to determine optimal storage types
3. Add type conversion utilities between logical and storage types
4. Remove duplicate type handling code from raw.py

### Phase 2: Column Abstraction
1. Create base Column class
2. Implement specific column types:
   - RawColumn
   - DictionaryColumn
   - RLEColumn
   - SingleValueColumn
   - NullColumn
3. Move BitPackedDictionaryWriter logic into DictionaryColumn
4. Add column factory system

### Phase 3: Compression Strategy Refactoring
1. Create CompressionStrategy base class
2. Implement strategy-specific analyzers
3. Create unified compression selection system
4. Add compression strategy registry

### Phase 4: Format Updates
1. Update MinimalWriter/Reader for new column system
2. Refactor CompressedWriter/Reader to use new abstractions
3. Update FormatFactory to work with new system
4. Add format versioning support

### Phase 5: Testing and Documentation
1. Update test suite for new architecture
2. Add property-based tests for type system
3. Create comprehensive documentation
4. Add performance benchmarks

## Implementation Notes

### New File Format Structure
```
Header:
  - Magic Number (4 bytes)
  - Version (1 byte)
  - Format Type (1 byte)
  - Column Count (2 bytes)

Column Definitions:
  For each column:
    - Name Length (1 byte)
    - Name (variable)
    - Logical Type (1 byte)
    - Storage Type (2 bytes)
      - Base Type (1 byte)
      - Bit Width (1 byte)

Data Section:
  For each column:
    - Compression Type (1 byte)
    - Data Length (4 bytes)
    - Compressed Data (variable)
```

### Performance Considerations
- Minimize memory copies during compression/decompression
- Use vectorized operations where possible
- Implement lazy loading for large datasets
- Consider memory mapping for file access

### Migration Strategy
1. Create new package version with backward compatibility
2. Implement new features in parallel
3. Add deprecation warnings for old interfaces
4. Remove deprecated code in future version

This redesign will provide a more maintainable and extensible codebase while addressing the current issues with type handling and compression organization.