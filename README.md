# fisheye-tools

A Rust library for fisheye camera model conversions.

## Overview

This library provides implementations for various camera models, focusing on fisheye lenses. It allows for projection and unprojection of points between 3D space and the 2D image plane.

Currently supported models:
- Pinhole
- Double Sphere
- Kannala-Brandt
- Radial-Tangential (RadTan)

(Add other models here as they are implemented)

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
- **Double Sphere**: See `DoubleSphereOptimizationCost`.
- **Kannala-Brandt**: See `KannalaBrandtOptimizationCost`.
- **Radial-Tangential (RadTan)**: See `RadTanOptimizationCost`.

These optimization tasks utilize the `factrs` crate for the underlying non-linear least squares solving.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
fisheye-tools = "0.1.0" # Replace with the desired version
```

## Usage

```rust
use fisheye_tools::pinhole::PinholeModel;
use fisheye_tools::camera::CameraModel; // Assuming a common trait/enum
use nalgebra::Point3;

fn main() {
    // Load camera parameters from a file (e.g., YAML)
    let model = PinholeModel::load_from_yaml("path/to/your/pinhole_params.yaml").expect("Failed to load camera model");

    // Define a 3D point
    let point_3d = Point3::new(1.0, 2.0, 5.0);

    // Project the 3D point onto the 2D image plane
    match model.project(&point_3d) {
        Ok(point_2d) => {
            println!("Projected 2D point: {:?}", point_2d);
            // Further processing...
        }
        Err(e) => {
            eprintln!("Projection failed: {}", e);
        }
    }

    // Example for unprojection (if implemented)
    // let point_2d = Point2::new(320.0, 240.0);
    // match model.unproject(&point_2d) {
    //     Ok(point_3d) => {
    //         println!("Unprojected 3D point: {:?}", point_3d);
    //     }
    //     Err(e) => {
    //         eprintln!("Unprojection failed: {}", e);
    //     }
    // }
}
```

*(Note: Adjust the file path and usage according to your actual implementation and file structure.)*

## Testing

Run the tests using:

```bash
cargo test
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.