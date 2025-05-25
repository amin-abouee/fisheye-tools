pub mod camera;
pub mod geometry;
pub mod optimization;

use crate::camera::{
    CameraModel, CameraModelError, DoubleSphereModel, Intrinsics, RadTanModel, Resolution,
};
use crate::optimization::Optimizer;
// , DoubleSphereOptimizationCost, KannalaBrandtOptimizationCost, RadTanOptimizationCost};
use clap::Parser;
use flexi_logger::{colored_detailed_format, detailed_format, Duplicate, FileSpec, Logger};
use log::{error, info};
use nalgebra::{Matrix2xX, Matrix3xX};
use optimization::{DoubleSphereOptimizationCost, RadTanOptimizationCost};
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
    points_2d: Matrix2xX<f64>,
    points_3d: Matrix3xX<f64>,
) -> Result<Box<dyn Optimizer>, Box<dyn std::error::Error>> {
    let output_model: Box<dyn Optimizer> = match output_model_type {
        "rad_tan" => {
            info!("Estimated init params: RadTan");
            let model = RadTanModel {
                intrinsics: input_intrinsic.clone(),
                resolution: input_resolution.clone(),
                distortion: [0.0; 5], // Initialize with zero
            };
            let mut cost_model = RadTanOptimizationCost::new(model, points_3d, points_2d);
            cost_model.linear_estimation()?;
            Box::new(cost_model)
        }
        "double_sphere" => {
            info!("Estimated init params: DoubleSphere");
            let model = DoubleSphereModel {
                intrinsics: input_intrinsic.clone(),
                resolution: input_resolution.clone(),
                alpha: 0.0,
                xi: 0.0,
            };
            let mut cost_model = DoubleSphereOptimizationCost::new(model, points_3d, points_2d);
            cost_model.linear_estimation()?;
            Box::new(cost_model)
        }
        _ => {
            error!("Unsupported output model type: {}", output_model_type);
            return Err("Unsupported output model type".into());
        }
    };

    Ok(output_model)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger with info level filter
    Logger::try_with_str("info")?
        // Configure file logging with detailed format
        .log_to_file(
            FileSpec::default()
                .directory("logs") // Store logs in the "logs" directory
                .suppress_timestamp() // Avoid timestamp in filename
                .suffix("log"), // Use .log extension
        )
        // Ensure all log messages are duplicated to stdout
        .duplicate_to_stdout(Duplicate::All)
        // Use detailed format for log files
        .format_for_files(detailed_format)
        // Use colored format for terminal output
        .format_for_stdout(colored_detailed_format)
        // Set custom color palette for different log levels
        // Format: "error;warn;info;debug;trace"
        // Using ANSI color codes:
        // 196=bright red, 208=orange, 76=green, 39=cyan, 178=gold
        .set_palette("196;208;76;39;178".to_string())
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
        points_2d,
        points_3d,
    )?;

    output_model.optimize(true).unwrap();

    info!("Output Model Parameters:");
    // Attempt to get parameters from the optimizer if it holds a camera model
    // This part might need adjustment based on how Optimizer trait is implemented
    // and whether it provides direct access to the underlying model's parameters.
    // For now, we assume the Optimizer might be a wrapper around a CameraModel
    // and we'd need a way to access that model. If the Optimizer itself
    // should have get_intrinsics, get_resolution, get_distortion, then the
    // Optimizer trait and its implementations would need to be updated.

    // Placeholder: How to get these details from `Box<dyn Optimizer>` depends on its design.
    // If the optimizer directly wraps a CameraModel and provides access:
    // if let Some(camera_model) = output_model.get_camera_model() { // Assuming such a method exists
    //     info!("Intrinsics: {:?}", camera_model.get_intrinsics());
    //     info!("Resolution: {:?}", camera_model.get_resolution());
    //     info!("Distortion: {:?}", camera_model.get_distortion());
    // } else {
    //     info!("Could not retrieve detailed model parameters from the optimizer.");
    // }

    // The following lines are commented out as they were in the original code
    // and their direct application to `output_model` of type `Box<dyn Optimizer>`
    // is not straightforward without knowing more about the `Optimizer` trait's
    // capabilities to expose underlying model details.
    // info!("Intrinsics: {:?}", output_model.get_intrinsics());
    // info!("Resolution: {:?}", output_model.get_resolution());
    // info!("Distortion: {:?}", output_model.get_distortion()); // Corrected from Resolution to Distortion

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
