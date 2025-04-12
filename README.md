# fisheye-tools

A Rust library for fisheye camera model conversions.

## Overview

This library provides implementations for various camera models, focusing on fisheye lenses. It allows for projection and unprojection of points between 3D space and the 2D image plane.

Currently supported models:
- Pinhole

(Add other models here as they are implemented)

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