# HYBF (Hybrid Binary Format) Project Analysis

## Project Overview

HYBF is a Python library for efficient binary storage of tabular data, with intelligent format selection based on data characteristics. The project uses a hybrid approach:
- Minimal format for small datasets (<4KB)
- Compressed format with multiple compression strategies for larger datasets

### Key Components

1. **Format Selection (factory.py)**
   - Analyzes data characteristics to choose appropriate format
   - Handles format detection during reading

2. **Base Infrastructure (core/)**
   - Abstract base classes for readers/writers
   - Data type definitions and metadata structures
   - Common utilities for binary operations

3. **Format Implementations**
   - Minimal format (minimal.py) for small datasets
   - Compressed format (compressed.py) with multiple compression strategies:
     - Run Length Encoding (RLE)
     - Dictionary encoding with bit packing
     - Single value optimization
     - Raw storage with numeric optimization

## Current Issues and Analysis

### 1. Type Optimization Fragmentation

The codebase has multiple implementations of bit-width optimization:

a) In `formats/raw.py`:
- Custom dtype mapping for optimized numeric storage
- Separate implementation from core type system
- Handles null values and type conversion

b) In `core/encoding.py`:
- BitPackedDictionaryWriter implements its own bit packing
- Focused on dictionary-specific use case
- Separate from main type system

This fragmentation leads to:
- Inconsistent type handling
- Duplicate logic
- Potential bugs in type conversion
- Maintenance challenges

### 2. Compression Architecture Issues

The compressed format implementation has several architectural concerns:

a) Component Organization:
- BitPackedDictionary classes in unexpected location
- Unclear separation of concerns
- Potential circular import issues

b) Compression Strategy Management:
- No unified interface for compression methods
- Mixed responsibility in CompressionSelector
- Limited extensibility

## TODO List

### Phase 1: Type System Refactoring

1. Create Unified Type System
```python
class OptimizedType:
    source_dtype: DataType  # Original data type
    storage_dtype: DataType  # Optimized storage type
    needs_conversion: bool
    
    def optimize(data: np.ndarray) -> Tuple[np.dtype, bool]
    def encode(data: np.ndarray) -> np.ndarray
    def decode(data: np.ndarray) -> np.ndarray
```

2. Implement Type Optimizers
- Create base TypeOptimizer class
- Implement numeric optimization (IntegerOptimizer, FloatOptimizer)
- Add dictionary-specific optimization
- Add tests for each optimizer

3. Update Type Registry
- Modify DataType enum to support optimized types
- Add type conversion mapping
- Update type detection logic

### Phase 2: Column Architecture

1. Create Column Base Class
```python
class Column:
    name: str
    dtype: OptimizedType
    compression: CompressionStrategy
    
    def encode(data: np.ndarray) -> bytes
    def decode(bytes_data: bytes) -> np.ndarray
```

2. Implement Column Types
- RawColumn (no compression)
- RLEColumn (run length encoding)
- DictionaryColumn (with bit packing)
- SingleValueColumn
- NullColumn

3. Add Column Factory
- Create ColumnFactory class
- Move compression selection logic
- Add column type registry

### Phase 3: Compression Refactoring

1. Create Compression Strategy Interface
```python
class CompressionStrategy:
    def analyze(data: np.ndarray) -> CompressionMetadata
    def compress(data: np.ndarray) -> bytes
    def decompress(data: bytes) -> np.ndarray
```

2. Implement Strategy Classes
- Move existing compression logic to separate classes
- Add strategy selection metrics
- Implement strategy chaining

3. Update CompressedWriter
- Use new Column architecture
- Implement strategy selection
- Add compression metadata handling

### Phase 4: Integration and Testing

1. Update Format Implementations
- Modify MinimalWriter to use new Column types
- Update CompressedWriter to use new architecture
- Ensure backward compatibility

2. Expand Test Suite
- Add tests for type optimization
- Add tests for each Column type
- Add integration tests
- Add performance benchmarks

3. Documentation and Examples
- Update API documentation
- Add migration guide
- Create examples for common use cases

## Implementation Notes

1. **Performance Considerations**
   - Measure impact of type conversion
   - Profile compression strategy selection
   - Consider caching optimization results

2. **Future Extensions**
   - Add support for additional compression methods
   - Consider parallel processing for large datasets
   - Add streaming support for large files

## Migration Strategy

1. Create new package structure
2. Implement new components alongside existing code
3. Add deprecation warnings for old interfaces
4. Provide migration utilities
5. Update documentation with migration guide

The refactoring should be done in phases, with each phase having its own set of tests and documentation updates. This allows for incremental improvements while maintaining a working codebase.