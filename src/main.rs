pub mod camera;
pub mod geometry;

use crate::camera::{CameraModel, DoubleSphereModel, PinholeModel, RadTanModel};
pub use clap::Parser;
pub use std::path::PathBuf; // Use PathBuf for paths

/// Simple program to demonstrate reading input/output model paths from args
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)] // Adds version, author, about from Cargo.toml
struct Cli {
    /// Type of the input camera model
    #[arg(short = 'i', long)]
    // Defines a required arg: -i <MODEL_TYPE> or --input_model <MODEL_TYPE>
    input_model: String,

    /// Type of the output camera model
    #[arg(short = 'o', long)]
    // Defines a required arg: -o <MODEL_TYPE> or --output_model <MODEL_TYPE>
    output_model: String,

    /// Path to the input model file
    #[arg(short = 'p', long)] // Defines a required arg: -p <PATH> or --input_path <PATH>
    input_path: PathBuf,
}

fn main() {
    // Parse the command line arguments into the Cli struct
    // clap automatically handles errors (e.g., missing args) and --help / --version
    let cli = Cli::parse();

    // Access the parsed arguments
    println!("Input Model Type: {}", cli.input_model);
    println!("Output Model Type: {}", cli.output_model);
    println!("Input Path: {:?}", cli.input_path);

    // let input_model = match cli.input_model.as_str() {
    //     "pinhole" => PinholeModel::load_from_yaml(cli.input_path.to_str().unwrap()).unwrap(),
    //     "radtan" => RadTanModel::load_from_yaml(cli.input_path.to_str().unwrap()).unwrap(),
    //     "double_sphere" => {
    //         DoubleSphereModel::load_from_yaml(cli.input_path.to_str().unwrap()).unwrap()
    //     }
    //     _ => {
    //         eprintln!("Unsupported input model type: {}", cli.input_model);
    //         std::process::exit(1);
    //     }
    // };

    // match cli.output_model.as_str() {
    //     "pinhole" => {
    //         let model = PinholeModel::load_from_yaml(cli.input_path).unwrap();
    //         println!("Loaded Pinhole Model");
    //     }
    //     "radtan" => {
    //         let model = RadTanModel::load_from_yaml(cli.input_path).unwrap();
    //         println!("Loaded RadTan Model");
    //     }
    //     "double_sphere" => {
    //         let model = DoubleSphereModel::load_from_yaml(cli.input_path).unwrap();
    //         println!("Loaded Double Sphere Model");
    //     }
    //     _ => {
    //         eprintln!("Unsupported output model type: {}", cli.output_model);
    //         std::process::exit(1);
    //     }
    // }
}

#[cfg(test)]
mod tests {
    use crate::camera::{CameraModel, DoubleSphereModel, PinholeModel, RadTanModel};
    use nalgebra::Point3;

    #[test]
    fn test_pinhole_camera() {
        let model = PinholeModel::load_from_yaml("samples/pinhole.yaml").unwrap();
        let point_3d = Point3::new(1.0, 1.0, 3.0);
        let point_2d = model.project(&point_3d).unwrap();
        assert!(point_2d.x > 0.0);
        assert!(point_2d.y > 0.0);
    }

    #[test]
    fn test_radtan_camera() {
        let model = RadTanModel::load_from_yaml("samples/rad_tan.yaml").unwrap();
        let point_3d = Point3::new(1.0, 1.0, 3.0);
        let point_2d = model.project(&point_3d).unwrap();
        assert!(point_2d.x > 0.0);
        assert!(point_2d.y > 0.0);
    }

    #[test]
    fn test_double_sphere_camera() {
        let model = DoubleSphereModel::load_from_yaml("samples/double_sphere.yaml").unwrap();
        let point_3d = Point3::new(1.0, 1.0, 3.0);
        let point_2d = model.project(&point_3d).unwrap();
        assert!(point_2d.x > 0.0);
        assert!(point_2d.y > 0.0);
    }
}
