# C++ vs Rust Implementation Comparison Framework

This document describes the comprehensive validation framework for comparing C++ and Rust implementations of camera model optimization algorithms.

## Overview

The comparison framework ensures mathematical equivalence between C++ (using Ceres Solver) and Rust (using factrs) implementations across all supported camera model conversions:

- **KB → Double Sphere**: Advanced fisheye model conversion
- **KB → Radial-Tangential**: Standard distortion model conversion  
- **KB → UCM**: Unified camera model conversion
- **KB → EUCM**: Extended unified camera model conversion

## Architecture

### Components

1. **C++ Benchmark** (`cpp_benchmark/final_demo.cpp`)
   - Implements identical conversion algorithms using Ceres Solver
   - Loads same test data from `samples/kannala_brandt.yaml`
   - Exports results in JSON format for Rust comparison

2. **Enhanced Rust Benchmark** (`examples/final_demo.rs`)
   - Runs factrs-based optimization
   - Automatically executes C++ benchmark if available
   - Performs detailed comparison analysis
   - Generates comprehensive comparison reports

3. **Comparison Analysis Engine**
   - Side-by-side accuracy comparison
   - Performance metrics analysis
   - Statistical validation
   - Mathematical equivalence verification

## Building and Running

### Prerequisites

**System Dependencies:**
```bash
# macOS
brew install cmake yaml-cpp jsoncpp

# Ubuntu/Debian
sudo apt-get install cmake libyaml-cpp-dev libjsoncpp-dev

# CentOS/RHEL
sudo yum install cmake yaml-cpp-devel jsoncpp-devel
```

**Rust Dependencies:**
- Automatically managed by Cargo
- serde_json added for JSON parsing

### Build Process

1. **Build C++ Benchmark:**
   ```bash
   cd cpp_benchmark
   ./build.sh
   ```

2. **Run C++ Benchmark:**
   ```bash
   cd cpp_benchmark/build
   ./final_demo
   ```

3. **Run Enhanced Rust Benchmark:**
   ```bash
   cargo run --example final_demo
   ```

## Validation Methodology

### Test Data Generation

Both implementations use:
- **Identical YAML input**: `samples/kannala_brandt.yaml`
- **Fixed random seed**: Ensures reproducible test points
- **Same point count**: 500 3D-2D correspondences
- **Identical bounds checking**: Points within image boundaries

### Mathematical Equivalence Criteria

**Accuracy Tolerance:**
- Error difference < 1e-3 pixels (1 millipixel)
- Validates numerical precision equivalence

**Residual Formulation:**
- Both use identical analytical derivatives
- Same parameter bounds and constraints
- Equivalent optimization termination criteria

**Convergence Behavior:**
- Similar iteration counts (when exposed)
- Comparable optimization times
- Consistent convergence status

### Comparison Metrics

**Accuracy Comparison:**
```
Error Difference = |Rust_Error - CPP_Error|
Accuracy Match = Error_Difference < 1e-3
```

**Performance Comparison:**
```
Time Difference = Rust_Time - CPP_Time
Performance Winner = argmin(Rust_Time, CPP_Time)
```

**Statistical Analysis:**
- Average error differences across all models
- Maximum error difference detection
- Performance distribution analysis
- Convergence success rates

## Output Formats

### Console Output

**Real-time Comparison Table:**
```
📋 C++ vs RUST COMPARISON TABLE
┌─────────────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Model                   │ Rust Error (px) │ C++ Error (px)  │ Error Diff (px) │ Rust Time (ms)  │ C++ Time (ms)   │
├─────────────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Double Sphere           │      1.167647   │      1.165432   │      0.002215   │         42.00   │         38.00   │ ✅
│ Unified Camera Model    │      0.145221   │      0.144876   │      0.000345   │         33.00   │         35.00   │ ✅
└─────────────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

**Summary Statistics:**
```
📈 COMPARISON SUMMARY
====================
🎯 Accuracy Matches: 4/4 (100.0%)
⚡ Performance: Rust wins 2, C++ wins 2
🏆 EXCELLENT: All implementations produce mathematically equivalent results!
```

### Detailed Report

**File:** `cpp_vs_rust_benchmark_comparison.txt`

**Contents:**
- Timestamp and framework information
- Detailed per-model comparison
- Statistical analysis
- Validation criteria checklist
- Overall assessment and recommendations

## Expected Results

### Accuracy Expectations

**Excellent (Target):**
- All models: Error difference < 1e-6 pixels
- Perfect mathematical equivalence

**Good (Acceptable):**
- All models: Error difference < 1e-3 pixels
- Minor numerical differences due to solver variations

**Warning (Investigate):**
- Any model: Error difference > 1e-3 pixels
- Potential implementation discrepancies

### Performance Expectations

**Typical Results:**
- Rust (factrs): Generally faster for smaller problems
- C++ (Ceres): May be faster for complex optimizations
- Time differences: Usually within 10-50ms range

## Troubleshooting

### Common Issues

**C++ Build Failures:**
```bash
# Check dependencies
pkg-config --exists yaml-cpp jsoncpp

# Manual dependency installation
# See Prerequisites section above
```

**JSON Parsing Errors:**
```bash
# Verify C++ benchmark completed successfully
ls -la cpp_benchmark_results.json

# Check JSON format
cat cpp_benchmark_results.json | python -m json.tool
```

**Accuracy Mismatches:**
- Verify identical test data loading
- Check parameter initialization consistency
- Validate optimization termination criteria
- Review residual formulation implementations

### Debug Mode

**Enable Verbose Output:**
```bash
# Run with debug information
RUST_LOG=debug cargo run --example final_demo
```

**Manual C++ Execution:**
```bash
# Run C++ benchmark independently
cd cpp_benchmark/build
./final_demo > cpp_output.log 2>&1
```

## Implementation Details

### C++ Implementation

**Framework:** Ceres Solver
**Language:** C++17
**Dependencies:** yaml-cpp, jsoncpp
**Optimization:** Analytical Jacobians
**Output:** JSON format for interoperability

### Rust Implementation

**Framework:** factrs
**Language:** Rust 2021 Edition
**Dependencies:** nalgebra, serde, serde_json
**Optimization:** Analytical Jacobians
**Output:** Console + detailed text report

### Data Flow

1. **Rust loads KB model** → `samples/kannala_brandt.yaml`
2. **Rust generates test points** → Fixed seed for reproducibility
3. **Rust runs conversions** → factrs optimization
4. **Rust executes C++ benchmark** → `cpp_benchmark/build/final_demo`
5. **C++ loads same data** → Identical YAML file
6. **C++ runs conversions** → Ceres optimization
7. **C++ exports results** → `cpp_benchmark_results.json`
8. **Rust loads C++ results** → JSON parsing
9. **Rust performs comparison** → Statistical analysis
10. **Rust generates report** → Console + file output

## Validation Checklist

- [ ] Both implementations load identical test data
- [ ] Same random seed used for point generation
- [ ] Identical residual formulations implemented
- [ ] Same parameter bounds and constraints
- [ ] Analytical Jacobians used (not automatic differentiation)
- [ ] Convergence criteria properly configured
- [ ] Error differences < 1e-3 pixels for all models
- [ ] Performance within expected ranges
- [ ] Detailed comparison report generated

## Future Enhancements

**Planned Improvements:**
- Automatic C++ dependency installation
- Extended statistical analysis
- Performance profiling integration
- Continuous integration testing
- Cross-platform validation
- Additional camera model support
