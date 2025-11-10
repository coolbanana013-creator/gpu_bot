# Deprecated Code

This folder contains deprecated code that is no longer used in the main application but kept for reference and potential future use.

## Files

### `gpu_processor.py`
- **Original Location**: `src/data_provider/gpu_processor.py`
- **Purpose**: GPU-accelerated data validation and cleaning for market data
- **Reason for Deprecation**: GPU validation always failed, CPU validation is more reliable
- **Moved**: 2025-11-10
- **Dependencies**: Required `data_processing.cl` kernel

### `data_processing.cl`
- **Original Location**: `src/gpu_kernels/data_processing.cl`
- **Purpose**: OpenCL kernel for GPU data processing operations
- **Reason for Deprecation**: Used only by deprecated `gpu_processor.py`
- **Moved**: 2025-11-10

## Removed Dead Code

### From `src/data_provider/loader.py`
- `_validate_and_clean_data_gpu()` method
- `validate_data_cpu_only()` method
- GPU processor initialization code

### From `src/ga/evolver_compact.py`
- `release_combinations()` method (no longer needed with global tracking)
- Comments about removed legacy methods: `_cpu_mutate_bot`, `_cpu_crossover`, `_generate_children`, `_gpu_generate_children`, `_cpu_generate_children`

### From `src/ga/gpu_logging_processor.py`
- `_log_first_bot_details()` method (individual bot detail files no longer created)

## Notes

- All functionality has been replaced with more efficient CPU-based implementations
- No breaking changes - the main application continues to work as expected
- Code is preserved here in case GPU validation needs to be revisited in the future
