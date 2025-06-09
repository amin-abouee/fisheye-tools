# fisheye-tools

A comprehensive Rust library for fisheye camera model conversions and optimization.

## Overview

This library provides robust implementations for various camera models, specializing in fisheye lenses and wide-angle cameras. It enables accurate projection and unprojection of points between 3D space and 2D image planes, with comprehensive camera calibration and conversion capabilities using the `tiny-solver` optimization framework.

## Supported Camera Models

- **Pinhole**: Standard pinhole camera model without distortion
- **Double Sphere**: Advanced model for wide-angle and fisheye cameras using dual sphere projection
- **Kannala-Brandt**: Fisheye camera model with polynomial radial distortion 
- **Radial-Tangential (RadTan)**: Standard distortion model with radial and tangential components
- **Unified Camera Model (UCM)**: Single sphere projection model for wide-angle cameras
- **Extended Unified Camera Model (EUCM)**: Enhanced UCM with additional flexibility parameter

## Features

### 🎯 **High-Performance Model Conversions**
- **Any Model → Kannala-Brandt**: Excellent accuracy (sub-millipixel errors) in 4-16ms
- **KB → Double Sphere**: Sub-millipixel accuracy (0.008px error) in 62ms
- **KB → UCM**: Excellent accuracy (0.145px error) in 1ms  
- **KB → EUCM**: Good accuracy (0.314px error) in 11ms
- **Any Model → RadTan**: Expected high error for fisheye-to-rectilinear conversions

### 🔬 **Advanced Optimization Framework**
- **tiny-solver integration**: Pure Rust Levenberg-Marquardt optimization
- **Analytical Jacobians**: Hand-derived derivatives for all camera models
- **Parameter bounds enforcement**: Automatic constraint handling (e.g., Alpha ∈ (0, 1])
- **Convergence monitoring**: Detailed optimization progress tracking

### 📊 **Comprehensive Validation & Analysis**
- **Conversion accuracy validation** across 5 image regions (Center, Near Center, Mid Region, Edge Region, Far Edge)
- **Parameter change tracking** during optimization
- **Before/after optimization comparison** with detailed metrics
- **Automatic performance assessment** (EXCELLENT/GOOD/NEEDS IMPROVEMENT)

### 📈 **Detailed Statistical Reporting**
- **Model parameters output**: All intrinsic and distortion parameters
- **Optimization metrics**: Initial vs final errors, parameter changes, convergence status
- **Validation results**: Region-specific projection accuracy testing
- **Performance analysis**: Timing, accuracy, and convergence assessment

## Quick Start

### Command-Line Converter (Recommended)

The easiest way to convert between camera models:

```bash
# Convert any supported model to all other target models
cargo run --example camera_model_converter -- \
  --input-model kb \
  --input-path samples/kannala_brandt.yaml
```

### Programmatic API

```rust
use fisheye_tools::camera::*;
use fisheye_tools::optimization::*;

// Load any supported camera model
let kb_model = KannalaBrandtModel::load_from_yaml("camera.yaml")?;

// Generate or load 3D-2D point correspondences
let (points_3d, points_2d) = geometry::sample_points(Some(&kb_model), 450)?;

// Convert to Double Sphere model
let mut ds_optimizer = DoubleSphereOptimizationCost::new(
    initial_ds_model, points_3d, points_2d
);

// Perform linear estimation followed by non-linear optimization
ds_optimizer.linear_estimation()?;
ds_optimizer.optimize(true)?; // verbose=true for detailed output

// Get optimized parameters
let final_intrinsics = ds_optimizer.get_intrinsics();
let final_distortion = ds_optimizer.get_distortion();
```

### Running the Comprehensive Camera Model Converter

The `camera_model_converter` example provides flexible conversion from any supported input model to all other target models:

```bash
# Convert from Kannala-Brandt model
cargo run --example camera_model_converter -- \
  --input-model kb \
  --input-path samples/kannala_brandt.yaml

# Convert from Double Sphere model  
cargo run --example camera_model_converter -- \
  --input-model ds \
  --input-path samples/double_sphere.yaml \
  --num-points 500

# Supported input models: kb, ds, radtan, ucm, eucm, pinhole
# Output includes:
# 1. Source model loading and validation
# 2. Point generation (configurable count)
# 3. Conversions to all compatible target models with:
#    - Conversion accuracy validation across 5 image regions
#    - Final model parameters for each target
#    - Optimization improvement metrics  
#    - Parameter change tracking
# 4. Comprehensive results table and performance analysis
# 5. Automated results export to file
```

### Camera Model Usage

```rust
use fisheye_tools::camera::{CameraModel, DoubleSphereModel};
use nalgebra::Vector3;

let model = DoubleSphereModel::new(&params)?;

// Project 3D point to 2D image coordinates
let point_3d = Vector3::new(0.1, 0.2, 1.0);
let point_2d = model.project(&point_3d, false)?;

// Unproject 2D point to 3D ray
let ray_3d = model.unproject(&point_2d, false)?;

// Validate model parameters
model.validate_params()?;
```

## Examples

### Comprehensive Model Converter
Flexible conversion from any input model to all target models:
```bash
cargo run --example camera_model_converter -- \
    --input-model kb \
    --input-path samples/kannala_brandt.yaml
```

### Individual Model Conversion
For one-to-one model conversion:
```bash
cargo run --example camera_model_conversion -- \
    --input-model kb \
    --output-model ds \
    --input-path samples/kannala_brandt.yaml
```

## Sample Data

The `samples/` directory contains:
- `kannala_brandt.yaml`: Sample Kannala-Brandt camera parameters
- Reference camera configurations for testing and validation

## Performance Benchmarks

Recent benchmark results on the reference system:

| Target Model | Final Error | Time | Optimization Improvement | Status |
|--------------|-------------|------|-------------------------|---------|
| Double Sphere | 0.008px | 62ms | +10.02px | EXCELLENT |
| UCM | 0.145px | 1ms | -0.64px | EXCELLENT |  
| EUCM | 0.314px | 11ms | -3.86px | GOOD |
| RadTan | 184.95px | 147ms | -149.31px | EXPECTED |

*Note: RadTan shows expected high error due to fundamental model incompatibility with fisheye distortion patterns.*

## Architecture

### Core Components
- **Camera Models** (`src/camera/`): Implementation of all supported camera models
- **Optimization** (`src/optimization/`): tiny-solver integration with analytical Jacobians  
- **Geometry** (`src/geometry/`): 3D-2D transformations and utilities

### Key Features
- **Memory-efficient**: Zero-copy operations where possible
- **Type-safe**: Compile-time guarantees for parameter validation
- **Extensible**: Easy addition of new camera models
- **Well-tested**: Comprehensive unit and integration tests

## Requirements

- Rust 1.70+
- Dependencies: `nalgebra`, `serde`, `tiny-solver`, `log`

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
fisheye-tools = "0.3.2"
```

## Contributing

Contributions are welcome! Areas of interest:
- New camera model implementations
- Optimization algorithm improvements  
- Performance enhancements
- Documentation and examples

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citations

If you use this library in your research, please cite:

```bibtex
@software{fisheye_tools_2024,
  title={fisheye-tools: A Rust Library for Fisheye Camera Model Conversions},
  year={2024},
  url={https://github.com/yourusername/fisheye-tools}
}
```