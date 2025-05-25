//! Command-line application for camera model conversion and optimization.
//!
//! This program allows users to load an existing camera model from a YAML file,
//! specify a target output camera model type, and then optimize the parameters
//! of this new model. The optimization is based on minimizing reprojection errors
//! using 2D-3D point correspondences generated from the input model.
//!
//! # Main Functionalities:
//!
//! *   **Model Loading**: Loads an input camera model (e.g., RadTan, DoubleSphere)
//!     from a YAML configuration file.
//! *   **Point Generation**: Generates a grid of 2D points across the input model's
//!     image plane and unprojects them to corresponding 3D rays, creating a set of
//!     2D-3D point correspondences.
//! *   **Output Model Initialization**: Creates an instance of the specified output camera
//!     model type, initializing its intrinsics and resolution from the input model.
//!     Distortion parameters are typically initialized to zero or estimated linearly.
//! *   **Linear Estimation**: Performs an initial linear estimation of the output model's
//!     distortion parameters using the generated 2D-3D correspondences.
//! *   **Non-linear Optimization**: Refines all parameters (intrinsics and distortion)
//!     of the output model by minimizing reprojection errors, using the `argmin`
//!     optimization library with a Gauss-Newton solver.
//! *   **Logging**: Uses `flexi_logger` for configurable logging to both console and file.
//!
//! # Modules Used:
//!
//! *   [`camera`]: Contains definitions for different camera models (e.g., [`RadTanModel`],
//!     [`DoubleSphereModel`]) and the core [`CameraModel`] trait.
//! *   [`geometry`]: Provides utilities for geometric operations, primarily the
//!     [`sample_points`] function used here to generate 2D-3D point correspondences.
//! *   [`optimization`]: Defines the [`Optimizer`] trait and provides specific
//!     optimization cost function implementations for each camera model (e.g.,
//!     [`RadTanOptimizationCost`], [`DoubleSphereOptimizationCost`]).

/// Module for camera models and operations.
pub mod camera;
/// Module for geometric utilities, like point sampling.
pub mod geometry;
/// Module for camera model parameter optimization.
pub mod optimization;

use crate::camera::{CameraModel, DoubleSphereModel, Intrinsics, RadTanModel, Resolution};
use crate::optimization::Optimizer;
use clap::Parser;
use flexi_logger::{colored_detailed_format, detailed_format, Duplicate, FileSpec, Logger};
use log::{error, info};
use nalgebra::{Matrix2xX, Matrix3xX};
use optimization::{DoubleSphereOptimizationCost, RadTanOptimizationCost};
use std::path::PathBuf; // Use PathBuf for paths

/// Simple program to demonstrate reading input/output model paths from args.
///
/// This struct defines the command-line arguments for the program, using `clap`
/// for parsing. It allows the user to specify the types and input paths for the
/// source and target camera models. The program will then attempt to load the
/// input model, generate sample points, initialize the output model, and optimize
/// its parameters.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)] // Fetches author, version, about from Cargo.toml
struct Cli {
    /// Type of the input camera model (e.g., "rad_tan", "double_sphere").
    /// This string identifier is used to select the correct loading mechanism.
    #[arg(short = 'i', long)]
    input_model: String,

    /// Type of the output camera model to be initialized and optimized
    /// (e.g., "rad_tan", "double_sphere").
    #[arg(short = 'o', long)]
    output_model: String,

    /// Path to the YAML file containing the parameters of the input camera model.
    #[arg(short = 'p', long)]
    input_path: PathBuf,
}

/// Creates a boxed [`CameraModel`] trait object from the specified type and YAML path.
///
/// This factory function attempts to load a camera model of the given `input_model_type`
/// from the YAML file specified by `input_path`. It supports different camera model
/// types through string identifiers.
///
/// # Arguments
///
/// *   `input_model_type`: `&str` - An identifier for the camera model type.
///     Supported values include "rad_tan" and "double_sphere".
/// *   `input_path`: `&str` - The file system path to the YAML file containing
///     the camera model's parameters.
///
/// # Return Value
///
/// Returns a `Result` containing:
/// *   `Ok(Box<dyn CameraModel>)`: On success, a dynamically dispatched camera model object
///     that implements the [`CameraModel`] trait.
/// *   `Err(Box<dyn std::error::Error>)`: On failure, an error indicating either an
///     unsupported model type, or an issue with file I/O or YAML parsing (propagated
///     from the specific model's `load_from_yaml` method).
fn create_input_model(
    input_model_type: &str,
    input_path: &str,
) -> Result<Box<dyn CameraModel>, Box<dyn std::error::Error>> {
    let input_model: Box<dyn CameraModel> = match input_model_type {
        "rad_tan" => {
            info!("Successfully loaded input model: RadTan from {}", input_path);
            Box::new(RadTanModel::load_from_yaml(input_path)?)
        }
        "double_sphere" => {
            info!("Successfully loaded input model: DoubleSphere from {}", input_path);
            Box::new(DoubleSphereModel::load_from_yaml(input_path)?)
        }
        _ => {
            let err_msg = format!("Unsupported input model type: {}", input_model_type);
            error!("{}", err_msg);
            return Err(err_msg.into());
        }
    };
    Ok(input_model)
}

/// Creates and initializes a boxed [`Optimizer`] trait object for the specified output model type.
///
/// This factory function serves to:
/// 1.  Instantiate a new camera model of the type specified by `output_model_type`.
///     The new model's intrinsics and resolution are cloned from the `input_intrinsic`
///     and `input_resolution` (typically derived from an input camera model).
///     Initial distortion parameters for the new model are set to zero or default values.
/// 2.  Wrap this newly created camera model within its corresponding optimization cost
///     structure (e.g., [`RadTanOptimizationCost`], [`DoubleSphereOptimizationCost`]).
/// 3.  Perform an initial `linear_estimation` of the distortion parameters for the
///     new model, using the provided `points_2d` and `points_3d` correspondences.
/// 4.  Return the configured optimizer setup, ready for non-linear optimization.
///
/// # Arguments
///
/// *   `output_model_type`: `&str` - An identifier for the target camera model type
///     (e.g., "rad_tan", "double_sphere").
/// *   `input_intrinsic`: `&Intrinsics` - A reference to the intrinsic parameters
///     (fx, fy, cx, cy) that will be used to initialize the output camera model.
/// *   `input_resolution`: `&Resolution` - A reference to the image resolution
///     (width, height) for the output camera model.
/// *   `points_2d`: [`Matrix2xX<f64>`] - A matrix of 2D image points (pixel coordinates)
///     used for the initial linear estimation of distortion parameters.
/// *   `points_3d`: [`Matrix3xX<f64>`] - A matrix of corresponding 3D world points
///     used for the initial linear estimation.
///
/// # Return Value
///
/// Returns a `Result` containing:
/// *   `Ok(Box<dyn Optimizer>)`: On success, a dynamically dispatched optimizer object
///     that implements the [`Optimizer`] trait. This object encapsulates the newly
///     initialized output camera model and is ready for the `optimize` method to be called.
/// *   `Err(Box<dyn std::error::Error>)`: On failure, an error indicating either an
///     unsupported output model type or an issue during the `linear_estimation` process.
fn create_output_model(
    output_model_type: &str,
    input_intrinsic: &Intrinsics,
    input_resolution: &Resolution,
    points_2d: Matrix2xX<f64>,
    points_3d: Matrix3xX<f64>,
) -> Result<Box<dyn Optimizer>, Box<dyn std::error::Error>> {
    let output_model: Box<dyn Optimizer> = match output_model_type {
        "rad_tan" => {
            info!("Initializing output model (RadTan) for optimization.");
            let model = RadTanModel {
                intrinsics: input_intrinsic.clone(),
                resolution: input_resolution.clone(),
                distortion: [0.0; 5], // Initialize distortion with zeros
            };
            let mut cost_model = RadTanOptimizationCost::new(model, points_3d, points_2d);
            cost_model.linear_estimation()?; // Perform initial linear estimation
            info!("RadTan model initialized with linear estimation.");
            Box::new(cost_model)
        }
        "double_sphere" => {
            info!("Initializing output model (DoubleSphere) for optimization.");
            let model = DoubleSphereModel {
                intrinsics: input_intrinsic.clone(),
                resolution: input_resolution.clone(),
                alpha: 0.5, // Default initial alpha (often near 0.5 for wide FoV)
                xi: 0.0,    // Default initial xi
            };
            let mut cost_model = DoubleSphereOptimizationCost::new(model, points_3d, points_2d);
            cost_model.linear_estimation()?; // Perform initial linear estimation
            info!("DoubleSphere model initialized with linear estimation.");
            Box::new(cost_model)
        }
        _ => {
            let err_msg = format!("Unsupported output model type: {}", output_model_type);
            error!("{}", err_msg);
            return Err(err_msg.into());
        }
    };

    Ok(output_model)
}

/// Main entry point for the camera model conversion and optimization application.
///
/// This function orchestrates the overall workflow of the program:
/// 1.  **Logging Initialization**: Sets up `flexi_logger` for console and file logging.
///     Log files are stored in a "logs" directory.
/// 2.  **Argument Parsing**: Parses command-line arguments using `clap` via the [`Cli`] struct.
///     This includes the input model type, output model type, and path to the input model's YAML file.
/// 3.  **Input Model Loading**: Calls [`create_input_model`] to load the specified source camera model
///     from its YAML file.
/// 4.  **Point Generation**: Invokes [`geometry::sample_points`] to generate a set of 2D image points
///     and their corresponding 3D unprojected rays based on the input model. This creates
///     the 2D-3D correspondences needed for optimizing the output model.
/// 5.  **Output Model Creation & Initialization**: Calls [`create_output_model`] to instantiate the target
///     camera model type. This function also performs an initial linear estimation of the
///     output model's distortion parameters using the generated 2D-3D points.
/// 6.  **Parameter Optimization**: Calls the `optimize()` method on the created output model
///     (which is a boxed [`Optimizer`] trait object). This performs non-linear optimization
///     to refine all parameters of the output model.
/// 7.  **Information Logging**: Logs various pieces of information throughout the process,
///     such as loaded model types, paths, parameters of the input model, number of generated points,
///     and parameters of the final optimized output model.
///
/// # Return Value
///
/// Returns `Result<(), Box<dyn std::error::Error>>`. `Ok(())` indicates successful execution
/// of the entire process. An `Err` is returned if any step in the workflow fails,
/// encapsulating the underlying error.
///
/// # Errors
///
/// This function can return various errors, including but not limited to:
/// *   Errors from logger initialization.
/// *   Errors during command-line argument parsing (though `clap` often handles these by exiting).
/// *   Errors from converting `PathBuf` to `&str` if the path is not valid UTF-8.
/// *   Errors from [`create_input_model`] (e.g., unsupported model type, file I/O, YAML parsing).
/// *   Errors from [`geometry::sample_points`] (e.g., unprojection issues). The current code `unwrap()`s this result.
/// *   Errors from [`create_output_model`] (e.g., unsupported model type, linear estimation failure).
/// *   Errors from the `optimize()` method of the [`Optimizer`] trait. The current code `unwrap()`s this result.
///
/// # Panics
///
/// The program may panic if `unwrap()` is called on a `Result` or `Option` that is `Err` or `None`.
/// Specifically:
/// *   `cli.input_path.to_str().ok_or("Invalid input path string")?` could panic if `ok_or` was not used
///     and `to_str()` returned `None` (though `ok_or` converts it to an error).
/// *   `geometry::sample_points(...).unwrap()` will panic if `sample_points` returns an `Err`.
/// *   `output_model.optimize(true).unwrap()` will panic if the `optimize` method returns an `Err`.
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
    let n = 100 as usize; // Approximate number of points for sampling
    let input_path_str = cli.input_path.to_str().ok_or("Invalid input path string (not valid UTF-8)")?;
    let input_model_type = cli.input_model.as_str();

    // Create and load the input camera model
    let input_model = create_input_model(input_model_type, input_path_str)?;
    info!("Input model loaded successfully.");

    // Generate 2D-3D point correspondences using the input model
    // Panics on error from sample_points.
    let (points_2d, points_3d) = geometry::sample_points(Some(&*input_model), n).unwrap();
    info!("Generated {} 2D-3D point correspondences.", points_2d.ncols());
    info!("First 2D point: {:?}", if points_2d.ncols() > 0 { Some(points_2d.column(0)) } else { None });
    info!("First 3D point: {:?}", if points_3d.ncols() > 0 { Some(points_3d.column(0)) } else { None });


    // Get parameters from the input model to initialize the output model
    let input_model_intrinsics = input_model.get_intrinsics();
    info!("Input model intrinsics: {:?}", input_model_intrinsics);
    let input_model_resolution = input_model.get_resolution();
    info!("Input model resolution: {:?}", input_model_resolution);
    let input_model_distortion = input_model.get_distortion();
    info!("Input model distortion parameters: {:?}", input_model_distortion);

    // Create and initialize the output camera model (optimizer context)
    let output_model_type = cli.output_model.as_str();
    let mut output_model = create_output_model(
        output_model_type,
        &input_model_intrinsics,
        &input_model_resolution,
        points_2d,
        points_3d,
    )?;
    info!("Output model initialized and linear estimation performed.");

    // Optimize the output model's parameters
    // Panics on error from optimize.
    output_model.optimize(true).unwrap();
    info!("Optimization of output model completed.");

    info!("Output Model Parameters (after optimization):");
    // The following lines were commented out in the original code.
    // To display the optimized parameters, the `Optimizer` trait would need methods
    // to access the underlying camera model, or the `Optimizer` itself would need
    // to store and provide access to these parameters post-optimization.
    // Example (assuming Optimizer has a way to get its internal model):
    // if let Some(final_model_params) = output_model.get_optimized_model_parameters_somehow() {
    //     info!("Intrinsics: {:?}", final_model_params.get_intrinsics());
    //     info!("Resolution: {:?}", final_model_params.get_resolution());
    //     info!("Distortion: {:?}", final_model_params.get_distortion());
    // } else {
    //     info!("Could not retrieve detailed model parameters from the optimizer directly.");
    //     info!("(This requires the Optimizer trait or its implementor to expose them)");
    // }

    Ok(())
}

/// Basic integration tests for camera model loading and projection.
#[cfg(test)]
mod tests {
    use crate::camera::{CameraModel, DoubleSphereModel, RadTanModel};
    use nalgebra::Vector3;

    /// Tests loading a RadTan model from YAML and projecting a sample 3D point.
    #[test]
    fn test_radtan_camera() {
        let model = RadTanModel::load_from_yaml("samples/rad_tan.yaml").unwrap();
        let point_3d = Vector3::new(1.0, 1.0, 3.0);
        let (point_2d, _) = model.project(&point_3d, false).unwrap();
        // Basic assertion: check that projected points are positive (assuming point is in front and FoV is typical)
        assert!(point_2d.x > 0.0);
        assert!(point_2d.y > 0.0);
    }

    /// Tests loading a DoubleSphere model from YAML and projecting a sample 3D point.
    #[test]
    fn test_double_sphere_camera() {
        let model = DoubleSphereModel::load_from_yaml("samples/double_sphere.yaml").unwrap();
        let point_3d = Vector3::new(1.0, 1.0, 3.0);
        let (point_2d, _) = model.project(&point_3d, false).unwrap();
        // Basic assertion: check that projected points are positive
        assert!(point_2d.x > 0.0);
        assert!(point_2d.y > 0.0);
    }
}
