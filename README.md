# fisheye-tools

A Rust library for fisheye camera model conversions and camera calibration.

## Overview

This library provides implementations for various camera models, focusing on fisheye lenses. It enables projection and unprojection of points between 3D space and the 2D image plane, with comprehensive camera calibration capabilities.

Currently supported models:
- **Pinhole**: Standard pinhole camera model without distortion
- **Double Sphere**: Advanced model for wide-angle and fisheye cameras using dual sphere projection
- **Kannala-Brandt**: Fisheye camera model with polynomial radial distortion
- **Radial-Tangential (RadTan)**: Standard distortion model with radial and tangential components
- **Unified Camera Model (UCM)**: Single sphere projection model for wide-angle cameras
- **Extended Unified Camera Model (EUCM)**: Enhanced UCM with additional parameter for better modeling

## Camera Model Optimization

This library provides advanced capabilities for camera model parameter optimization using the `tiny-solver` framework. The optimization system is designed for high-performance camera calibration tasks, enabling conversion between different camera models with mathematical precision.

### The `Optimizer` Trait

A unified interface for camera model optimization is defined by the `Optimizer` trait (see `src/optimization/mod.rs`). Implementations for specific camera models provide:
- **Non-linear optimization** using the Levenberg-Marquardt algorithm via `tiny-solver`
- **Linear estimation** for initial parameter guesses
- **Parameter validation** and bounds enforcement
- **Analytical Jacobians** for optimal performance

### Supported Models for Optimization

Optimization is implemented for all camera models:
- **Double Sphere**: `DoubleSphereOptimizationCost`
- **Kannala-Brandt**: `KannalaBrandtOptimizationCost` 
- **Radial-Tangential (RadTan)**: `RadTanOptimizationCost`
- **Unified Camera Model (UCM)**: `UcmOptimizationCost`
- **Extended Unified Camera Model (EUCM)**: `EucmOptimizationCost`

## Performance Benchmarks

The library includes comprehensive benchmarking tools to evaluate conversion accuracy and performance across all supported camera models.

### Comprehensive Camera Model Conversion Benchmark

Run the complete benchmark to test Kannala-Brandt → target model conversions:

```bash
cargo run --example final_demo
```

This benchmark provides detailed analysis of:
- **KB → Double Sphere**: Advanced fisheye model conversion
- **KB → Radial-Tangential**: Standard distortion model conversion  
- **KB → UCM**: Unified camera model conversion
- **KB → EUCM**: Extended unified camera model conversion

**Sample Results:**
```
📊 Final Output Model Parameters:
DS parameters: fx=190.923, fy=190.918, cx=254.932, cy=256.898, alpha=0.630074, xi=1.0421
computing time(ms): 78

🧪 EVALUATION AND VALIDATION:
reprojection error from input model to output model: 0.00773438

🎯 Conversion Accuracy Validation:
  Center: Input(nan, nan) → Output(254.93, 256.90) | Error: 0.0000 px
  📈 Average Error: 0.0077 px, Max Error: 0.0077 px
  ⚠️  Conversion Accuracy: EXCELLENT - Error < 0.01 pixels
```

### Model Conversion Performance Summary

Recent benchmark results demonstrate excellent accuracy:

| Target Model | Reprojection Error | Computing Time | Status |
|--------------|-------------------|----------------|---------|
| Double Sphere | 0.0077 px | 78 ms | Excellent |
| UCM | 0.145 px | 16 ms | Good |
| EUCM | 0.146 px | 454 ms | Good |
| RadTan | 184.95 px | 148 ms | Expected high error |

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

### Optimization Framework

The implementation ensures mathematical correctness through:

- ✅ **Analytical Jacobians**: Hand-derived derivatives for optimal performance
- ✅ **tiny-solver Integration**: Modern Rust optimization framework
- ✅ **Parameter Bounds**: Proper constraints enforced (e.g., Alpha ∈ (0, 1])
- ✅ **Convergence Tracking**: Detailed optimization statistics
- ✅ **Cross-Validation**: Extensive test coverage with known ground truth
- ✅ **Linear Estimation**: Smart initialization for better convergence

### 📊 Benchmark Output

The benchmark generates detailed result files:

- **`rust_benchmark_results.txt`** - Comprehensive results with per-model analysis
- **Console output** - Real-time formatted tables and detailed statistics

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
fisheye-tools = "0.3.2"
```

## Usage

```rust
use fisheye_tools::camera::{CameraModel, DoubleSphereModel};
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
- **tiny-solver Optimization**: High-performance camera calibration using analytical Jacobians
- **YAML Configuration**: Easy parameter loading and saving in YAML format
- **Comprehensive Testing**: Full test coverage with Jacobian validation
- **Cross-Platform**: Works on Linux, macOS, and Windows
- **Mathematical Precision**: Rigorous validation and cross-testing ensure correctness

## Performance

This library is optimized for performance with:
- **Release Mode by Default**: Configured for maximum optimization (`opt-level = 3`, LTO enabled)
- **Analytical Jacobians**: Uses hand-derived Jacobians instead of automatic differentiation
- **tiny-solver Framework**: Leverages modern Rust optimization framework
- **Zero-Copy Operations**: Efficient memory usage with minimal allocations
- **SIMD-Optimized**: Takes advantage of hardware acceleration where available

## Code Quality

The project maintains high code quality standards:

```bash
# Format all code
cargo fmt --all

# Check for common issues
cargo clippy --all-targets --all-features

# Run all tests with coverage
cargo test --all-features

# Check documentation
cargo doc --all-features --no-deps
```

## Contributing

Contributions are welcome! Please ensure your code:
- Passes all tests: `cargo test --all-features`
- Is properly formatted: `cargo fmt --all`
- Passes clippy checks: `cargo clippy --all-targets --all-features`
- Includes appropriate documentation and tests

Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.