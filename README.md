# fisheye-tools

A Rust library for fisheye camera model conversions.

## Overview

This library provides implementations for various camera models, focusing on fisheye lenses. It allows for projection and unprojection of points between 3D space and the 2D image plane.

Currently supported models:
- **Pinhole**: Standard pinhole camera model without distortion
- **Double Sphere**: Advanced model for wide-angle and fisheye cameras using dual sphere projection
- **Kannala-Brandt**: Fisheye camera model with polynomial radial distortion
- **Radial-Tangential (RadTan)**: Standard distortion model with radial and tangential components
- **Unified Camera Model (UCM)**: Single sphere projection model for wide-angle cameras
- **Extended Unified Camera Model (EUCM)**: Enhanced UCM with additional parameter for better modeling

## Camera Model Optimization

This library also provides capabilities for optimizing camera model parameters. This is useful for camera calibration tasks, where the goal is to find the camera parameters that best describe the relationship between 3D world points and their corresponding 2D image projections.

The optimization process typically refines the intrinsic parameters (focal length, principal point) and distortion coefficients for a given camera model.

### The `Optimizer` Trait

A common interface for camera model optimization is defined by the `Optimizer` trait (see `src/optimization/mod.rs`). Implementations of this trait for specific camera models allow users to:
- Perform non-linear optimization of camera parameters (usually using Levenberg-Marquardt).
- Optionally, perform a linear estimation for an initial guess of some parameters.
- Retrieve the current intrinsic, resolution, and distortion parameters from the model.

### Supported Models for Optimization

Optimization is currently implemented for the following camera models:
- **Double Sphere**: See `DoubleSphereOptimizationCost`
- **Kannala-Brandt**: See `KannalaBrandtOptimizationCost`
- **Radial-Tangential (RadTan)**: See `RadTanOptimizationCost`
- **Unified Camera Model (UCM)**: See `UcmOptimizationCost`
- **Extended Unified Camera Model (EUCM)**: See `EucmOptimizationCost`

These optimization tasks utilize the `factrs` crate for the underlying non-linear least squares solving.

## Performance Benchmarks

The library provides comprehensive benchmarking tools to evaluate conversion accuracy and performance across all supported camera models.

### Comprehensive Conversion Benchmark

Run the complete benchmark to test KB→target model conversions:

```bash
cargo run --example final_demo
```

This benchmark provides:
- **KB → Double Sphere**: Advanced fisheye model conversion
- **KB → Radial-Tangential**: Standard distortion model conversion
- **KB → UCM**: Unified camera model conversion
- **KB → EUCM**: Extended unified camera model conversion

**Sample Results:**
```
┌─────────────────────────┬─────────────────┬─────────────┬─────────────────┬─────────────────┐
│ Target Model            │ Reprojection    │ Iterations  │ Time (ms)       │ Convergence     │
│                         │ Error (pixels)  │             │                 │ Status          │
├─────────────────────────┼─────────────────┼─────────────┼─────────────────┼─────────────────┤
│ Double Sphere           │      1.167647   │         1   │         41.00   │ Success         │
│ Radial-Tangential       │     35.637131   │         1   │         53.00   │ Linear Only     │
│ Unified Camera Model    │      0.145221   │         1   │         32.00   │ Success         │
│ Extended Unified Camera Model │     97.193595   │         1   │         32.00   │ Linear Only     │
└─────────────────────────┴─────────────────┴─────────────┴─────────────────┴─────────────────┘
```

### Simple Camera Model Conversion

For basic model conversion workflows:

```bash
cargo run --example camera_model_conversion
```

This example demonstrates:
- Loading camera models from YAML files
- Converting between different model types
- Saving converted models
- Basic optimization workflows

### Testing and Validation

The library includes comprehensive test suites for all camera models:

```bash
# Run all tests
cargo test --all-features

# Test specific camera models
cargo test double_sphere --lib
cargo test ucm --lib
cargo test eucm --lib
cargo test rad_tan --lib

# Test optimization functionality
cargo test optimization --lib
```

### C++ vs Rust Implementation Comparison

The library includes a comprehensive validation framework that compares Rust and C++ implementations to ensure mathematical equivalence:

#### Building the C++ Benchmark

```bash
# Navigate to the fisheye-calib-adapter repository
cd /Volumes/External/Workspace/fisheye-calib-adapter

# Compile the benchmark (requires yaml-cpp)
g++ -std=c++17 -O3 -I/opt/homebrew/include -L/opt/homebrew/lib \
    -lyaml-cpp -o simple_benchmark simple_benchmark.cpp

# Run C++ benchmark
./simple_benchmark
```

#### Running the Comparison

```bash
# Run enhanced Rust benchmark with C++ comparison
cargo run --example final_demo
```

**Sample Comparison Results:**
```
📋 C++ vs RUST COMPARISON TABLE
┌─────────────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Model                   │ Rust Error (px) │ C++ Error (px)  │ Error Diff (px) │ Rust Time (ms)  │ C++ Time (ms)   │
├─────────────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Double Sphere           │      1.167647   │      1.167656   │      0.000009   │         39.00   │         52.00   │ ✅
│ Radial-Tangential       │     35.637131   │     35.637211   │      0.000080   │         49.00   │         53.00   │ ✅
│ Unified Camera Model    │      0.145221   │      0.145223   │      0.000002   │         30.00   │         35.00   │ ✅
│ Extended Unified Camera Model │     97.193595   │     97.193675   │      0.000080   │         29.00   │         39.00   │ ✅
└─────────────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘

📈 COMPARISON SUMMARY
====================
🎯 Accuracy Matches: 4/4 (100.0%)
⚡ Performance: Rust wins 4, C++ wins 0
🏆 EXCELLENT: All implementations produce mathematically equivalent results!
```

#### Validation Criteria

The comparison framework validates:

- **Mathematical Equivalence**: Error differences < 1e-3 pixels
- **Identical Test Data**: Deterministic point generation with fixed seeds
- **Same Residual Formulations**: Both use analytical derivatives
- **Identical Parameter Bounds**: Same optimization constraints
- **Convergence Behavior**: Similar optimization characteristics

#### Troubleshooting C++ Integration

**Dependencies Required:**
- CMake 3.16+
- yaml-cpp library
- jsoncpp library
- C++17 compatible compiler

**Installation on macOS:**
```bash
brew install cmake yaml-cpp jsoncpp
```

**Installation on Ubuntu:**
```bash
sudo apt-get install cmake libyaml-cpp-dev libjsoncpp-dev
```

### Optimization Verification

The implementation ensures mathematical correctness through:

- ✅ **Analytical Jacobians**: Hand-derived derivatives for optimal performance
- ✅ **C++ Compatibility**: Residual formulations match reference implementations
- ✅ **Parameter Bounds**: Proper constraints enforced (e.g., Alpha ∈ (0, 1])
- ✅ **Convergence Tracking**: Detailed optimization statistics
- ✅ **Cross-Validation**: Extensive test coverage with known ground truth
- ✅ **Implementation Equivalence**: Direct C++ vs Rust comparison framework

### 📊 Comprehensive Benchmark Results

The benchmark generates detailed result files for thorough analysis:

#### Generated Files

- **`rust_benchmark_results.txt`** - Detailed Rust implementation results with per-model analysis
- **`cpp_benchmark_results.txt`** - Detailed C++ implementation results with per-model analysis
- **`cpp_vs_rust_benchmark_comparison.txt`** - Side-by-side comparison analysis
- **`cpp_benchmark_results.json`** - C++ results in JSON format for programmatic access
- **Console output** - Real-time formatted tables and analysis

#### Side-by-Side Comparison Tool

Use the included Python script for easy side-by-side comparison:

```bash
# Compare detailed results from both implementations
python3 compare_results.py
```

**Features:**
- **Side-by-side text comparison** of detailed results
- **Key metrics summary** (accuracy, performance, framework details)
- **Mathematical equivalence assessment** (< 1e-3 pixels difference)
- **Performance comparison** (optimization times, iterations)

**Sample Output:**
```
🎯 COMPARISON ASSESSMENT
================================================================================
✅ EXCELLENT: Mathematical equivalence achieved (< 1e-3 pixels difference)
   Average error difference: 0.000043 pixels

📊 KEY METRICS COMPARISON
Framework                      Rust                      C++
Average Error                  33.535898 pixels          33.535941 pixels
Average Time                   35.75 ms                  45.25 ms
Total Time                     143.00 ms                 181.00 ms
Best Accuracy                  0.145221 pixels           0.145223 pixels
```

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
fisheye-tools = "0.3.0"
```

## Usage

```rust
use fisheye_tools::camera::{CameraModel, DoubleSphereModel, UcmModel, EucmModel};
use fisheye_tools::optimization::{Optimizer, DoubleSphereOptimizationCost};
use nalgebra::{Vector3, Vector2, Matrix3xX, Matrix2xX};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load camera parameters from a YAML file
    let model = DoubleSphereModel::load_from_yaml("samples/double_sphere.yaml")?;

    // Define a 3D point in camera coordinates
    let point_3d = Vector3::new(1.0, 2.0, 5.0);

    // Project the 3D point onto the 2D image plane
    match model.project(&point_3d) {
        Ok(point_2d) => {
            println!("Projected 2D point: ({:.2}, {:.2})", point_2d.x, point_2d.y);

            // Unproject back to 3D ray
            match model.unproject(&point_2d) {
                Ok(ray_3d) => {
                    println!("Unprojected 3D ray: ({:.3}, {:.3}, {:.3})",
                             ray_3d.x, ray_3d.y, ray_3d.z);
                }
                Err(e) => eprintln!("Unprojection failed: {}", e),
            }
        }
        Err(e) => eprintln!("Projection failed: {}", e),
    }

    // Example: Camera calibration optimization
    // Generate some sample 3D-2D correspondences
    let points_3d = Matrix3xX::from_columns(&[
        Vector3::new(1.0, 0.0, 5.0),
        Vector3::new(0.0, 1.0, 5.0),
        Vector3::new(-1.0, 0.0, 5.0),
        Vector3::new(0.0, -1.0, 5.0),
    ]);

    // Project these points to get 2D observations
    let mut points_2d_vec = Vec::new();
    for i in 0..points_3d.ncols() {
        let p3d = points_3d.column(i);
        if let Ok(p2d) = model.project(&Vector3::new(p3d[0], p3d[1], p3d[2])) {
            points_2d_vec.push(p2d);
        }
    }
    let points_2d = Matrix2xX::from_columns(&points_2d_vec);

    // Create optimization cost function
    let mut optimizer = DoubleSphereOptimizationCost::new(model, points_3d, points_2d);

    // Perform optimization
    match optimizer.optimize(true) {
        Ok(()) => println!("Camera calibration optimization completed successfully!"),
        Err(e) => eprintln!("Optimization failed: {}", e),
    }

    Ok(())
}
```

## Features

- **Multiple Camera Models**: Support for 6 different camera models including fisheye and wide-angle lenses
- **Factrs Optimization**: High-performance camera calibration using analytical Jacobians
- **YAML Configuration**: Easy parameter loading and saving in YAML format
- **Comprehensive Testing**: Full test coverage with Jacobian validation
- **Cross-Platform**: Works on Linux, macOS, and Windows

## Testing

Run the tests using:

```bash
# Run all tests
cargo test

# Run tests with all features
cargo test --all-features

# Run specific camera model tests
cargo test camera::double_sphere
cargo test optimization::ucm

# Run with verbose output
cargo test -- --nocapture
```

## Performance

This library is optimized for performance with:
- **Release Mode by Default**: Configured for maximum optimization (`opt-level = 3`, LTO enabled)
- **Analytical Jacobians**: Uses hand-derived Jacobians instead of automatic differentiation for better performance
- **Factrs Framework**: Leverages the high-performance factrs optimization library
- **Zero-Copy Operations**: Efficient memory usage with minimal allocations

## Contributing

Contributions are welcome! Please ensure your code:
- Passes all tests: `cargo test --all-features`
- Is properly formatted: `cargo fmt --all`
- Passes clippy checks: `cargo clippy --all-targets --all-features`
- Includes appropriate documentation and tests

Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.