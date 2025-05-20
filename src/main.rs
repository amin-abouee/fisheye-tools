pub mod camera;
pub mod geometry;

use crate::camera::{CameraModel, DoubleSphereModel, Intrinsics, RadTanModel, Resolution};
use clap::Parser;
use flexi_logger::{colored_with_thread, FileSpec, Logger};
use log::{error, info};
use nalgebra::{Matrix2xX, Matrix3xX};
use std::path::PathBuf; // Use PathBuf for paths

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

fn create_input_model(
    input_model_type: &str,
    input_path: &str,
) -> Result<Box<dyn CameraModel>, Box<dyn std::error::Error>> {
    let input_model: Box<dyn CameraModel> = match input_model_type {
        "rad_tan" => {
            info!("Successfully loaded input model: RadTan");
            Box::new(RadTanModel::load_from_yaml(input_path)?)
        }
        "double_sphere" => {
            info!("Successfully loaded input model: DoubleSphere");
            Box::new(DoubleSphereModel::load_from_yaml(input_path)?)
        }
        _ => {
            error!("Unsupported input model type: {}", input_model_type);
            return Err("Unsupported input model type".into());
        }
    };
    Ok(input_model)
}

fn create_output_model(
    output_model_type: &str,
    input_intrinsic: &Intrinsics,
    input_resolution: &Resolution,
    points_2d: &Matrix2xX<f64>,
    points_3d: &Matrix3xX<f64>,
) -> Result<Box<dyn CameraModel>, Box<dyn std::error::Error>> {
    let output_model: Box<dyn CameraModel> = match output_model_type {
        "rad_tan" => {
            info!("Estimated init params: RadTan");
            Box::new(RadTanModel::linear_estimation(
                &input_intrinsic,
                &input_resolution,
                &points_2d,
                &points_3d,
            )?)
        }
        "double_sphere" => {
            info!("Estimated init params: DoubleSphere");
            Box::new(DoubleSphereModel::linear_estimation(
                &input_intrinsic,
                &input_resolution,
                &points_2d,
                &points_3d,
            )?)
        }
        _ => {
            error!("Unsupported input model type: {}", output_model_type);
            return Err("Unsupported input model type".into());
        }
    };

    Ok(output_model)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    Logger::try_with_str("info")? // Log level filter
        .log_to_file(FileSpec::default().directory("logs")) // Log to files in the "logs" directory
        .format(colored_with_thread) // Use the custom format including file and line number
        .start()?;
    // Parse the command line arguments into the Cli struct
    // clap automatically handles errors (e.g., missing args) and --help / --version
    let cli = Cli::parse();

    // Access the parsed arguments
    info!("Input Model Type: {}", cli.input_model);
    info!("Output Model Type: {}", cli.output_model);
    info!("Input Path: {:?}", cli.input_path);

    // Convert PathBuf to &str for loading functions
    let n = 100 as usize;
    let input_path_str = cli.input_path.to_str().ok_or("Invalid input path string")?;
    let input_model_type = cli.input_model.as_str();
    let input_model = create_input_model(input_model_type, input_path_str)?;
    let (points_2d, points_3d) = geometry::sample_points(Some(&*input_model), n).unwrap();
    info!("points_2d: {:?}", points_2d.ncols());
    info!("points_3d: {:?}", points_3d.ncols());

    let input_model_intrinsics = input_model.get_intrinsics();
    info!("input_model_intrinsics: {:?}", input_model_intrinsics);
    let input_model_resolution = input_model.get_resolution();
    info!("input_model_resolution: {:?}", input_model_resolution);
    let input_model_distortion = input_model.get_distortion();
    info!("input_model_distortion: {:?}", input_model_distortion);

    let output_model_type = cli.output_model.as_str();
    let mut output_model = create_output_model(
        output_model_type,
        &input_model_intrinsics,
        &input_model_resolution,
        &points_2d,
        &points_3d,
    )?;

    output_model.optimize(&points_3d, &points_2d, true).unwrap();

    info!("Output Model Parameters:");
    info!("Intrinsics: {:?}", output_model.get_intrinsics());
    info!("Resolution: {:?}", output_model.get_resolution());
    info!("Resolution: {:?}", output_model.get_distortion());

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::camera::{CameraModel, DoubleSphereModel, RadTanModel};
    use nalgebra::Vector3;

    #[test]
    fn test_radtan_camera() {
        let model = RadTanModel::load_from_yaml("samples/rad_tan.yaml").unwrap();
        let point_3d = Vector3::new(1.0, 1.0, 3.0);
        let (point_2d, _) = model.project(&point_3d, false).unwrap();
        assert!(point_2d.x > 0.0);
        assert!(point_2d.y > 0.0);
    }

    #[test]
    fn test_double_sphere_camera() {
        let model = DoubleSphereModel::load_from_yaml("samples/double_sphere.yaml").unwrap();
        let point_3d = Vector3::new(1.0, 1.0, 3.0);
        let (point_2d, _) = model.project(&point_3d, false).unwrap();
        assert!(point_2d.x > 0.0);
        assert!(point_2d.y > 0.0);
    }
}
