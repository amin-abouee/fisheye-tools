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