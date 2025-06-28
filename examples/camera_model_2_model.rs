//! Camera Model Conversion Example
//!
//! This example demonstrates how to convert between different camera models
//! using the fisheye-tools library. It supports conversion between:
//! - Radial-Tangential distortion model
//! - Double Sphere camera model
//! - Kannala-Brandt camera model
//!
//! Usage:
//! ```bash
//! cargo run --example camera_model_conversion -- \
//!   --input_model kannala_brandt \
//!   --output_model double_sphere \
//!   --input_path samples/kannala_brandt.yaml
//! ```

use clap::Parser;
use fisheye_tools::camera::{
    CameraModel, DoubleSphereModel, Intrinsics, KannalaBrandtModel, RadTanModel, Resolution,
};
use fisheye_tools::optimization::Optimizer;
use fisheye_tools::{optimization, util};
use flexi_logger::{colored_detailed_format, detailed_format, Duplicate, FileSpec, Logger};
use log::{error, info};
use nalgebra::{Matrix2xX, Matrix3xX};
use optimization::{
    DoubleSphereOptimizationCost, KannalaBrandtOptimizationCost, RadTanOptimizationCost,
};
use std::path::PathBuf;

/// Camera model conversion tool
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Type of the input camera model
    #[arg(short = 'i', long)]
    input_model: String,

    /// Type of the output camera model
    #[arg(short = 'o', long)]
    output_model: String,

    /// Path to the input model file
    #[arg(short = 'p', long)]
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
        "kannala_brandt" => {
            info!("Successfully loaded input model: KannalaBrandt");
            Box::new(KannalaBrandtModel::load_from_yaml(input_path)?)
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
                distortions: [0.0; 5], // Initialize with zero
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
                alpha: 0.5, // Valid initial value in range (0, 1]
                xi: 0.1,    // Small positive initial value
            };
            let mut cost_model = DoubleSphereOptimizationCost::new(model, points_3d, points_2d);
            cost_model.linear_estimation()?;
            Box::new(cost_model)
        }
        "kannala_brandt" => {
            info!("Estimated init params: KannalaBrandt");
            let model = KannalaBrandtModel {
                intrinsics: input_intrinsic.clone(),
                resolution: input_resolution.clone(),
                distortions: [0.0; 4], // Initialize with zero distortion coefficients
            };
            let mut cost_model = KannalaBrandtOptimizationCost::new(model, points_3d, points_2d);
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

    // Also print to stdout for debugging
    println!("Input Model Type: {}", cli.input_model);
    println!("Output Model Type: {}", cli.output_model);
    println!("Input Path: {:?}", cli.input_path);

    // Convert PathBuf to &str for loading functions
    let n = 500_usize;
    let input_path_str = cli.input_path.to_str().ok_or("Invalid input path string")?;
    let input_model_type = cli.input_model.as_str();
    let input_model = create_input_model(input_model_type, input_path_str)?;
    let (points_2d, points_3d) = util::sample_points(Some(&*input_model), n).unwrap();
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

    // Try to optimize and handle errors gracefully
    match output_model.optimize(true) {
        Ok(()) => {
            info!("Optimization completed successfully!");
        }
        Err(e) => {
            error!("Optimization failed: {:?}", e);
            info!("Continuing with linear estimation results...");
        }
    }

    info!("Output Model Parameters:");
    info!("Intrinsics: {:?}", output_model.get_intrinsics());
    info!("Resolution: {:?}", output_model.get_resolution());
    info!("Distortion: {:?}", output_model.get_distortion());

    // Also print to stdout for debugging
    println!("\n=== CONVERSION RESULTS ===");
    println!("Input Model ({}): ", input_model_type);
    println!("  Intrinsics: {:?}", input_model_intrinsics);
    println!("  Resolution: {:?}", input_model_resolution);
    println!("  Distortion: {:?}", input_model_distortion);
    println!("\nOutput Model ({}):", output_model_type);
    println!("  Intrinsics: {:?}", output_model.get_intrinsics());
    println!("  Resolution: {:?}", output_model.get_resolution());
    println!("  Distortion: {:?}", output_model.get_distortion());
    println!("========================");

    Ok(())
}
