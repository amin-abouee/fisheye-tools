use clap::Parser;
use std::path::PathBuf; // Use PathBuf for paths

pub mod camera;
pub mod double_sphere;
pub mod pinhole;
pub mod rad_tan;
// pub mod kannala_brandt;

/// Simple program to demonstrate reading input/output model paths from args
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)] // Adds version, author, about from Cargo.toml
struct Cli {
    /// Path to the input model file
    #[arg(short = 'i', long)] // Defines a required arg: -i <PATH> or --input <PATH>
    input_model: PathBuf,

    /// Path to the output model file
    #[arg(short = 'o', long)] // Defines a required arg: -o <PATH> or --output <PATH>
    output_model: PathBuf,
}

fn main() {
    // Parse the command line arguments into the Cli struct
    // clap automatically handles errors (e.g., missing args) and --help / --version
    let cli = Cli::parse();

    // Access the parsed arguments
    println!("Input Model Path: {:?}", cli.input_model);
    println!("Output Model Path: {:?}", cli.output_model);
}

#[cfg(test)]
mod tests {
    use crate::camera::CameraModel;
    use crate::double_sphere::DoubleSphereModel;
    use crate::pinhole::PinholeModel;
    use crate::rad_tan::RadTanModel;
    use nalgebra::Point3;

    #[test]
    fn test_pinhole_camera() {
        let model = PinholeModel::load_from_yaml("src/pinhole/pinhole.yaml").unwrap();
        let point_3d = Point3::new(1.0, 1.0, 3.0);
        let point_2d = model.project(&point_3d).unwrap();
        assert!(point_2d.x > 0.0);
        assert!(point_2d.y > 0.0);
    }

    #[test]
    fn test_radtan_camera() {
        let model = RadTanModel::load_from_yaml("src/rad_tan/rad_tan.yaml").unwrap();
        let point_3d = Point3::new(1.0, 1.0, 3.0);
        let point_2d = model.project(&point_3d).unwrap();
        assert!(point_2d.x > 0.0);
        assert!(point_2d.y > 0.0);
    }

    #[test]
    fn test_double_sphere_camera() {
        let model =
            DoubleSphereModel::load_from_yaml("src/double_sphere/double_sphere.yaml").unwrap();
        let point_3d = Point3::new(1.0, 1.0, 3.0);
        let point_2d = model.project(&point_3d).unwrap();
        assert!(point_2d.x > 0.0);
        assert!(point_2d.y > 0.0);
    }
}
