//! Comprehensive Camera Model Converter
//!
//! This example demonstrates parameter estimation for all supported camera models
//! by converting from any input camera model to all other supported models.
//!
//! Supported input models:
//! - Kannala-Brandt (KB)
//! - Double Sphere (DS)
//! - Radial-Tangential (RadTan)
//! - Unified Camera Model (UCM)
//! - Extended Unified Camera Model (EUCM)
//! - Pinhole
//!
//! Supported output models:
//! - Double Sphere (DS) model
//! - Radial-Tangential (RadTan) model
//! - Unified Camera Model (UCM)
//! - Extended Unified Camera Model (EUCM)
//!
//! Usage:
//! ```bash
//! cargo run --example camera_model_converter -- \
//!   --input-model kannala_brandt \
//!   --input-path samples/kannala_brandt.yaml
//! ```

use clap::Parser;
use fisheye_tools::camera::{
    CameraModel, CameraModelEnum, DoubleSphereModel, EucmModel, KannalaBrandtModel, PinholeModel,
    RadTanModel, UcmModel,
};
use fisheye_tools::geometry::{self, ConversionMetrics, ValidationResults};
use fisheye_tools::optimization::{
    DoubleSphereOptimizationCost, EucmOptimizationCost, KannalaBrandtOptimizationCost, Optimizer,
    RadTanOptimizationCost, UcmOptimizationCost,
};
use log::info;

// Structs now imported from geometry module
use std::path::PathBuf;
use std::time::Instant;

/// Camera model conversion tool
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Type of the input camera model (kb, ds, radtan, ucm, eucm, pinhole)
    #[arg(short = 'i', long)]
    input_model: String,

    /// Path to the input model YAML file
    #[arg(short = 'p', long)]
    input_path: PathBuf,

    /// Number of sample points to generate for optimization (default: 500)
    #[arg(short = 'n', long, default_value = "500")]
    num_points: usize,
}

// Structs moved to geometry module

/// Load input camera model from file based on type
fn load_input_model(
    model_type: &str,
    path: &str,
) -> Result<Box<dyn CameraModel>, Box<dyn std::error::Error>> {
    let model: Box<dyn CameraModel> = match model_type.to_lowercase().as_str() {
        "kb" | "kannala_brandt" => {
            info!("Loading Kannala-Brandt model from: {}", path);
            Box::new(KannalaBrandtModel::load_from_yaml(path)?)
        }
        "ds" | "double_sphere" => {
            info!("Loading Double Sphere model from: {}", path);
            Box::new(DoubleSphereModel::load_from_yaml(path)?)
        }
        "radtan" | "rad_tan" => {
            info!("Loading Radial-Tangential model from: {}", path);
            Box::new(RadTanModel::load_from_yaml(path)?)
        }
        "ucm" | "unified" => {
            info!("Loading Unified Camera Model from: {}", path);
            Box::new(UcmModel::load_from_yaml(path)?)
        }
        "eucm" | "extended_unified" => {
            info!("Loading Extended Unified Camera Model from: {}", path);
            Box::new(EucmModel::load_from_yaml(path)?)
        }
        "pinhole" => {
            info!("Loading Pinhole model from: {}", path);
            Box::new(PinholeModel::load_from_yaml(path)?)
        }
        _ => {
            return Err(format!("Unsupported input model type: {}. Supported types: kb, ds, radtan, ucm, eucm, pinhole", model_type).into());
        }
    };
    Ok(model)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();

    let cli = Cli::parse();

    // Display tool header
    // geometry::display_tool_header(&cli.input_model, &cli.input_path, cli.num_points);

    println!("🎯 COMPREHENSIVE CAMERA MODEL CONVERSION TOOL");
    println!("==============================================");
    println!(
        "Converting from {} model to all supported target models",
        cli.input_model.to_uppercase()
    );
    println!("Input file: {:?}", cli.input_path);
    println!("Sample points: {}\n", cli.num_points);

    info!("🎯 COMPREHENSIVE CAMERA MODEL CONVERSION TOOL");
    info!(
        "Converting from {} model to all supported target models",
        cli.input_model.to_uppercase()
    );

    // Step 1: Load input model
    let input_path_str = cli.input_path.to_str().ok_or("Invalid input path string")?;
    let input_model = load_input_model(&cli.input_model, input_path_str)?;

    // Display input model parameters
    geometry::display_input_model_parameters(&cli.input_model, &*input_model);

    info!(
        "✅ Successfully loaded {} model from YAML",
        cli.input_model.to_uppercase()
    );

    // Step 2: Generate sample points
    let (points_2d, points_3d) = geometry::sample_points(Some(&*input_model), cli.num_points)?;

    // Display sample points information
    // geometry::display_sample_points_info(cli.num_points, points_3d.ncols(), points_2d.ncols());

    println!("\n🎲 Generating Sample Points:");
    println!("Requested points: {}", cli.num_points);
    println!("Generated {} sample points", points_3d.ncols());

    println!("\n🔄 Creating 3D-2D Point Correspondences:");
    println!(
        "Valid 3D-2D correspondences: {} / {}",
        points_2d.ncols(),
        cli.num_points
    );

    // Export point correspondences using geometry module
    geometry::export_point_correspondences(&points_3d, &points_2d, "point_correspondences")?;

    info!(
        "✅ Generated {} 3D-2D point correspondences",
        points_2d.ncols()
    );

    // Step 3: Run all conversions with detailed results after each
    println!("\n🔄 Step 3: Converting to All Target Models");
    println!("==========================================");
    let mut all_metrics = Vec::new();

    // Convert to Double Sphere (if not the input model)
    if !matches!(
        cli.input_model.to_lowercase().as_str(),
        "ds" | "double_sphere"
    ) {
        println!(
            "\n📐 Converting {} → {}",
            cli.input_model.to_uppercase(),
            "Double Sphere"
        );
        println!("{}", "-".repeat(32 + cli.input_model.len()));
        if let Ok(metrics) = convert_to_double_sphere(&*input_model, &points_3d, &points_2d) {
            all_metrics.push(metrics);
        }
    }

    // Convert to Kannala-Brandt (if not the input model)
    if !matches!(
        cli.input_model.to_lowercase().as_str(),
        "kb" | "kannala_brandt"
    ) {
        println!(
            "\n📐 Converting {} → {}",
            cli.input_model.to_uppercase(),
            "Kannala-Brandt"
        );
        println!("{}", "-".repeat(32 + cli.input_model.len()));
        if let Ok(metrics) = convert_to_kannala_brandt(&*input_model, &points_3d, &points_2d) {
            all_metrics.push(metrics);
        }
    }

    // Convert to Radial-Tangential (if not the input model)
    if !matches!(
        cli.input_model.to_lowercase().as_str(),
        "radtan" | "rad_tan"
    ) {
        println!(
            "\n📐 Converting {} → {}",
            cli.input_model.to_uppercase(),
            "Radial-Tangential"
        );
        println!("{}", "-".repeat(37 + cli.input_model.len()));
        if let Ok(metrics) = convert_to_rad_tan(&*input_model, &points_3d, &points_2d) {
            all_metrics.push(metrics);
        }
    }

    // Convert to UCM (if not the input model)
    if !matches!(cli.input_model.to_lowercase().as_str(), "ucm" | "unified") {
        println!(
            "\n📐 Converting {} → {}",
            cli.input_model.to_uppercase(),
            "Unified Camera Model"
        );
        println!("{}", "-".repeat(40 + cli.input_model.len()));
        if let Ok(metrics) = convert_to_ucm(&*input_model, &points_3d, &points_2d) {
            all_metrics.push(metrics);
        }
    }

    // Convert to EUCM (if not the input model)
    if !matches!(
        cli.input_model.to_lowercase().as_str(),
        "eucm" | "extended_unified"
    ) {
        println!(
            "\n📐 Converting {} → {}",
            cli.input_model.to_uppercase(),
            "Extended Unified Camera Model"
        );
        println!("{}", "-".repeat(49 + cli.input_model.len()));
        if let Ok(metrics) = convert_to_eucm(&*input_model, &points_3d, &points_2d) {
            all_metrics.push(metrics);
        }
    }

    // Display results summary using geometry module
    geometry::display_results_summary(&all_metrics, &cli.input_model);

    if all_metrics.is_empty() {
        return Ok(());
    }

    Ok(())
}

// geometry::display_detailed_results function moved to geometry module as display_detailed_results

// Conversion function implementations
fn convert_to_double_sphere(
    input_model: &dyn CameraModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // Initialize Double Sphere model
    let initial_model = DoubleSphereModel {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        alpha: 0.5,
        xi: 0.1,
    };

    // Compute initial reprojection error
    let initial_error =
        geometry::compute_reprojection_error(Some(&initial_model), points_3d, points_2d)?;

    let mut optimizer =
        DoubleSphereOptimizationCost::new(initial_model, points_3d.clone(), points_2d.clone());

    // Linear estimation
    optimizer.linear_estimation()?;

    // Non-linear optimization with verbose output
    let optimization_result = optimizer.optimize(true);

    let optimization_time = start_time.elapsed().as_millis() as f64;

    // Get final parameters
    let final_intrinsics = optimizer.get_intrinsics();
    let final_distortion = optimizer.get_distortion();

    // Compute reprojection error
    let final_model = DoubleSphereModel {
        intrinsics: final_intrinsics.clone(),
        resolution: optimizer.get_resolution(),
        alpha: final_distortion[0],
        xi: final_distortion[1],
    };

    let reprojection_result =
        geometry::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;

    let convergence_status = match optimization_result {
        Ok(()) => "Success".to_string(),
        Err(_) => "Linear Only".to_string(),
    };

    let validation_results = geometry::validate_conversion_accuracy(&final_model, input_model)
        .unwrap_or_else(|_| ValidationResults {
            center_error: f64::NAN,
            near_center_error: f64::NAN,
            mid_region_error: f64::NAN,
            edge_region_error: f64::NAN,
            far_edge_error: f64::NAN,
            average_error: f64::NAN,
            max_error: f64::NAN,
            status: "NEEDS IMPROVEMENT".to_string(),
            region_data: vec![],
        });

    let metrics = ConversionMetrics {
        model: CameraModelEnum::DoubleSphere(final_model),
        model_name: "Double Sphere".to_string(),
        final_reprojection_error: reprojection_result,
        initial_reprojection_error: initial_error,
        optimization_time_ms: optimization_time,
        convergence_status,
        validation_results,
    };

    geometry::display_detailed_results(&metrics);
    Ok(metrics)
}

/// Convert input model to Kannala-Brandt camera model with optimization
fn convert_to_kannala_brandt(
    input_model: &dyn CameraModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // Initialize Kannala-Brandt model
    let initial_model = KannalaBrandtModel {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        distortions: [0.0; 4], // k1, k2, k3, k4
    };

    // Compute initial reprojection error
    let initial_error =
        geometry::compute_reprojection_error(Some(&initial_model), points_3d, points_2d)?;

    // Create optimizer and perform linear estimation
    let mut optimizer =
        KannalaBrandtOptimizationCost::new(initial_model, points_3d.clone(), points_2d.clone());

    // Perform linear estimation first
    optimizer.linear_estimation()?;

    // Run optimization
    let optimization_result = optimizer.optimize(true); // verbose=true for detailed tiny-solver output

    let optimization_time = start_time.elapsed().as_millis() as f64;

    // Get final parameters
    let final_intrinsics = optimizer.get_intrinsics();
    let final_distortion = optimizer.get_distortion();

    // Compute reprojection error
    let final_model = KannalaBrandtModel {
        intrinsics: final_intrinsics.clone(),
        resolution: optimizer.get_resolution(),
        distortions: [
            final_distortion[0],
            final_distortion[1],
            final_distortion[2],
            final_distortion[3],
        ],
    };

    let reprojection_result =
        geometry::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;

    // Validation testing using C++ reference test points
    let validation_results = geometry::validate_conversion_accuracy(&final_model, input_model)
        .unwrap_or_else(|_| ValidationResults {
            center_error: f64::NAN,
            near_center_error: f64::NAN,
            mid_region_error: f64::NAN,
            edge_region_error: f64::NAN,
            far_edge_error: f64::NAN,
            average_error: f64::NAN,
            max_error: f64::NAN,
            status: "NEEDS IMPROVEMENT".to_string(),
            region_data: vec![],
        });

    let convergence_status = match optimization_result {
        Ok(()) => "Success".to_string(),
        Err(_) => "Linear Only".to_string(),
    };

    let metrics = ConversionMetrics {
        model: CameraModelEnum::KannalaBrandt(final_model),
        model_name: "Kannala-Brandt".to_string(),
        final_reprojection_error: reprojection_result,
        initial_reprojection_error: initial_error,
        optimization_time_ms: optimization_time,
        convergence_status,
        validation_results,
    };

    geometry::display_detailed_results(&metrics);
    Ok(metrics)
}

fn convert_to_rad_tan(
    input_model: &dyn CameraModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // Initialize RadTan model
    let initial_model = RadTanModel {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        distortions: [0.0; 5], // k1, k2, p1, p2, k3
    };

    // Compute initial reprojection error
    let initial_error =
        geometry::compute_reprojection_error(Some(&initial_model), points_3d, points_2d)?;

    let mut optimizer =
        RadTanOptimizationCost::new(initial_model, points_3d.clone(), points_2d.clone());

    // Linear estimation
    optimizer.linear_estimation()?;

    // Non-linear optimization with verbose output
    let optimization_result = optimizer.optimize(true);

    let optimization_time = start_time.elapsed().as_millis() as f64;

    // Get final parameters
    let final_intrinsics = optimizer.get_intrinsics();
    let final_distortion = optimizer.get_distortion();

    // Compute reprojection error
    let final_model = RadTanModel {
        intrinsics: final_intrinsics.clone(),
        resolution: optimizer.get_resolution(),
        distortions: [
            final_distortion[0],
            final_distortion[1],
            final_distortion[2],
            final_distortion[3],
            final_distortion[4],
        ],
    };

    let reprojection_result =
        geometry::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;

    let convergence_status = match optimization_result {
        Ok(()) => "Success".to_string(),
        Err(_) => "Linear Only".to_string(),
    };

    let validation_results = geometry::validate_conversion_accuracy(&final_model, input_model)
        .unwrap_or_else(|_| ValidationResults {
            center_error: f64::NAN,
            near_center_error: f64::NAN,
            mid_region_error: f64::NAN,
            edge_region_error: f64::NAN,
            far_edge_error: f64::NAN,
            average_error: f64::NAN,
            max_error: f64::NAN,
            status: "NEEDS IMPROVEMENT".to_string(),
            region_data: vec![],
        });

    let metrics = ConversionMetrics {
        model: CameraModelEnum::RadTan(final_model),
        model_name: "Radial-Tangential".to_string(),
        final_reprojection_error: reprojection_result,
        initial_reprojection_error: initial_error,
        optimization_time_ms: optimization_time,
        convergence_status,
        validation_results,
    };

    geometry::display_detailed_results(&metrics);
    Ok(metrics)
}

fn convert_to_ucm(
    input_model: &dyn CameraModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // Initialize UCM model
    let initial_model = UcmModel {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        alpha: 0.5, // Initial guess
    };

    // Compute initial reprojection error
    let initial_error =
        geometry::compute_reprojection_error(Some(&initial_model), points_3d, points_2d)?;

    let mut optimizer =
        UcmOptimizationCost::new(initial_model, points_3d.clone(), points_2d.clone());

    // Linear estimation
    optimizer.linear_estimation()?;

    // Non-linear optimization with verbose output
    let optimization_result = optimizer.optimize(true);

    let optimization_time = start_time.elapsed().as_millis() as f64;

    // Get final parameters
    let final_intrinsics = optimizer.get_intrinsics();
    let final_distortion = optimizer.get_distortion();

    // Compute reprojection error
    let final_model = UcmModel {
        intrinsics: final_intrinsics.clone(),
        resolution: optimizer.get_resolution(),
        alpha: final_distortion[0],
    };

    let reprojection_result =
        geometry::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;

    let convergence_status = match optimization_result {
        Ok(()) => "Success".to_string(),
        Err(_) => "Linear Only".to_string(),
    };

    let validation_results = geometry::validate_conversion_accuracy(&final_model, input_model)
        .unwrap_or_else(|_| ValidationResults {
            center_error: f64::NAN,
            near_center_error: f64::NAN,
            mid_region_error: f64::NAN,
            edge_region_error: f64::NAN,
            far_edge_error: f64::NAN,
            average_error: f64::NAN,
            max_error: f64::NAN,
            status: "NEEDS IMPROVEMENT".to_string(),
            region_data: vec![],
        });

    let metrics = ConversionMetrics {
        model: CameraModelEnum::Ucm(final_model),
        model_name: "Unified Camera Model".to_string(),
        final_reprojection_error: reprojection_result,
        initial_reprojection_error: initial_error,
        optimization_time_ms: optimization_time,
        convergence_status,
        validation_results,
    };

    geometry::display_detailed_results(&metrics);
    Ok(metrics)
}

fn convert_to_eucm(
    input_model: &dyn CameraModel,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // Initialize EUCM model
    let initial_model = EucmModel {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        alpha: 0.5, // Initial guess
        beta: 1.0,  // Initial guess
    };

    // Compute initial reprojection error
    let initial_error =
        geometry::compute_reprojection_error(Some(&initial_model), points_3d, points_2d)?;

    let mut optimizer =
        EucmOptimizationCost::new(initial_model, points_3d.clone(), points_2d.clone());

    // Linear estimation
    optimizer.linear_estimation()?;

    // Non-linear optimization with verbose output
    let optimization_result = optimizer.optimize(true);

    let optimization_time = start_time.elapsed().as_millis() as f64;

    // Get final parameters
    let final_intrinsics = optimizer.get_intrinsics();
    let final_distortion = optimizer.get_distortion();

    // Compute reprojection error
    let final_model = EucmModel {
        intrinsics: final_intrinsics.clone(),
        resolution: optimizer.get_resolution(),
        alpha: final_distortion[0],
        beta: final_distortion[1],
    };

    let reprojection_result =
        geometry::compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;

    let convergence_status = match optimization_result {
        Ok(()) => "Success".to_string(),
        Err(_) => "Linear Only".to_string(),
    };

    let validation_results = geometry::validate_conversion_accuracy(&final_model, input_model)
        .unwrap_or_else(|_| ValidationResults {
            center_error: f64::NAN,
            near_center_error: f64::NAN,
            mid_region_error: f64::NAN,
            edge_region_error: f64::NAN,
            far_edge_error: f64::NAN,
            average_error: f64::NAN,
            max_error: f64::NAN,
            status: "NEEDS IMPROVEMENT".to_string(),
            region_data: vec![],
        });

    let metrics = ConversionMetrics {
        model: CameraModelEnum::Eucm(final_model),
        model_name: "Extended Unified Camera Model".to_string(),
        final_reprojection_error: reprojection_result,
        initial_reprojection_error: initial_error,
        optimization_time_ms: optimization_time,
        convergence_status,
        validation_results,
    };

    geometry::display_detailed_results(&metrics);
    Ok(metrics)
}

// Removed export_conversion_results function - now using geometry module functions
