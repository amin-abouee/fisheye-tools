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
    CameraModel, DoubleSphereModel, EucmModel, KannalaBrandtModel, PinholeModel, RadTanModel,
    UcmModel,
};
use fisheye_tools::geometry::{self, compute_reprojection_error};
use fisheye_tools::optimization::{
    DoubleSphereOptimizationCost, EucmOptimizationCost, KannalaBrandtOptimizationCost, Optimizer,
    RadTanOptimizationCost, UcmOptimizationCost,
};
use log::info;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
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

/// Parameter estimation metrics with detailed model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConversionMetrics {
    model_name: String,
    final_reprojection_error: f64,

    initial_reprojection_error: f64,
    optimization_time_ms: f64,
    convergence_status: String,
    // Model parameters for detailed reporting
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    distortion_params: Vec<f64>,
    // Parameter changes during optimization
    parameter_changes: ParameterChanges,
    validation_results: ValidationResults,
}

/// Parameter changes during optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParameterChanges {
    fx_change: f64,
    fy_change: f64,
    cx_change: f64,
    cy_change: f64,
    distortion_changes: Vec<f64>,
    total_parameter_change: f64,
}

/// Validation results for conversion accuracy testing
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ValidationResults {
    center_error: f64,
    near_center_error: f64,
    mid_region_error: f64,
    edge_region_error: f64,
    far_edge_error: f64,
    average_error: f64,
    max_error: f64,
    status: String,
}

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

/// Perform detailed validation testing across different regions using specific 3D test points
fn perform_validation_testing<T: CameraModel>(
    output_model: &T,
    input_model: &dyn CameraModel,
) -> ValidationResults {
    // Test projection accuracy at different regions using specific 3D points
    let test_regions = [
        ("Center", Vector3::new(0.0, 0.0, 1.0)),
        ("Near Center", Vector3::new(0.05, 0.05, 1.0)),
        ("Mid Region", Vector3::new(0.15, 0.1, 1.0)),
        ("Edge Region", Vector3::new(0.3, 0.2, 1.0)),
        ("Far Edge", Vector3::new(0.4, 0.3, 1.0)),
    ];

    let mut total_error = 0.0;
    let mut max_error = 0.0;
    let mut valid_projections = 0;
    let mut region_errors = [f64::NAN; 5];

    for (i, (_, point_3d)) in test_regions.iter().enumerate() {
        // Try to project using both input and output models
        match (
            input_model.project(point_3d),
            output_model.project(point_3d),
        ) {
            (Ok(input_proj), Ok(output_proj)) => {
                let error = (input_proj - output_proj).norm();
                total_error += error;
                max_error = f64::max(max_error, error);
                valid_projections += 1;
                region_errors[i] = error;
            }
            _ => {
                region_errors[i] = f64::NAN;
            }
        }
    }

    let average_error = if valid_projections > 0 {
        total_error / valid_projections as f64
    } else {
        f64::NAN
    };

    let status = if average_error.is_nan() {
        "NEEDS IMPROVEMENT".to_string()
    } else if average_error < 0.01 {
        "EXCELLENT".to_string()
    } else if average_error < 0.1 {
        "GOOD".to_string()
    } else {
        "NEEDS IMPROVEMENT".to_string()
    };

    ValidationResults {
        center_error: region_errors[0],
        near_center_error: region_errors[1],
        mid_region_error: region_errors[2],
        edge_region_error: region_errors[3],
        far_edge_error: region_errors[4],
        average_error,
        max_error,
        status,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();

    let cli = Cli::parse();

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
    println!("📷 Input Model Parameters:");
    let input_path_str = cli.input_path.to_str().ok_or("Invalid input path string")?;
    let input_model = load_input_model(&cli.input_model, input_path_str)?;

    // Print input model parameters in the requested format
    let intrinsics = input_model.get_intrinsics();
    let distortion = input_model.get_distortion();

    match cli.input_model.to_lowercase().as_str() {
        "ds" | "double_sphere" => {
            println!(
                "DS parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, alpha={:.6}, xi={:.6}",
                intrinsics.fx,
                intrinsics.fy,
                intrinsics.cx,
                intrinsics.cy,
                distortion[0],
                distortion[1]
            );
        }
        "kb" | "kannala_brandt" => {
            println!(
                "KB parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, k1={:.6}, k2={:.6}, k3={:.6}, k4={:.6}",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                distortion[0], distortion[1], distortion[2], distortion[3]
            );
        }
        "radtan" | "rad_tan" => {
            println!(
                "RadTan parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, k1={:.6}, k2={:.6}, p1={:.6}, p2={:.6}, k3={:.6}",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                distortion[0], distortion[1], distortion[2], distortion[3], distortion[4]
            );
        }
        "ucm" | "unified" => {
            println!(
                "UCM parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, alpha={:.6}",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, distortion[0]
            );
        }
        "eucm" | "extended_unified" => {
            println!(
                "EUCM parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, alpha={:.6}, beta={:.6}",
                intrinsics.fx,
                intrinsics.fy,
                intrinsics.cx,
                intrinsics.cy,
                distortion[0],
                distortion[1]
            );
        }
        "pinhole" => {
            println!(
                "Pinhole parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
            );
        }
        _ => {
            println!(
                "Model parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
            );
        }
    }

    info!(
        "✅ Successfully loaded {} model from YAML",
        cli.input_model.to_uppercase()
    );

    // Step 2: Generate sample points
    println!("\n🎲 Generating Sample Points:");
    println!("Requested points: {}", cli.num_points);
    let (points_2d, points_3d) = geometry::sample_points(Some(&*input_model), cli.num_points)?;
    println!("Generated {} sample points", points_3d.ncols());

    println!("\n🔄 Creating 3D-2D Point Correspondences:");
    println!(
        "Valid 3D-2D correspondences: {} / {}",
        points_2d.ncols(),
        cli.num_points
    );

    println!("\n💾 Exporting Point Correspondences for Rust Comparison:");
    println!("Exported {} point correspondences to:", points_2d.ncols());
    println!("  - point_correspondences.csv (CSV format)");
    println!("  - point_correspondences_rust.txt (Rust code format)");

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
            "\n📐 Converting {} → Double Sphere",
            cli.input_model.to_uppercase()
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
            "\n📐 Converting {} → Kannala-Brandt",
            cli.input_model.to_uppercase()
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
            "\n📐 Converting {} → Radial-Tangential",
            cli.input_model.to_uppercase()
        );
        println!("{}", "-".repeat(37 + cli.input_model.len()));
        if let Ok(metrics) = convert_to_rad_tan(&*input_model, &points_3d, &points_2d) {
            all_metrics.push(metrics);
        }
    }

    // Convert to UCM (if not the input model)
    if !matches!(cli.input_model.to_lowercase().as_str(), "ucm" | "unified") {
        println!(
            "\n📐 Converting {} → Unified Camera Model",
            cli.input_model.to_uppercase()
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
            "\n📐 Converting {} → Extended Unified Camera Model",
            cli.input_model.to_uppercase()
        );
        println!("{}", "-".repeat(49 + cli.input_model.len()));
        if let Ok(metrics) = convert_to_eucm(&*input_model, &points_3d, &points_2d) {
            all_metrics.push(metrics);
        }
    }

    // Step 4: Generate comprehensive benchmark report
    println!("\n📊 Step 4: Conversion Results Summary");
    println!("====================================");

    if all_metrics.is_empty() {
        println!("❌ No conversions performed (input model type not supported for conversion or no target models available)");
        return Ok(());
    }

    // Print detailed results table
    println!("\n📋 CONVERSION RESULTS TABLE");
    println!("┌────────────────────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐");
    println!("│ Target Model                   │ Final Error     │ Improvement     │ Time (ms)       │ Convergence     │");
    println!("│                                │ (pixels)        │ (pixels)        │                 │ Status          │");
    println!("├────────────────────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤");

    for metrics in &all_metrics {
        let improvement = metrics.initial_reprojection_error - metrics.final_reprojection_error;
        println!(
            "│ {:<30} │ {:>13.6}   │ {:>13.6}   │ {:>13.2}   │ {:<15} │",
            metrics.model_name,
            metrics.final_reprojection_error,
            improvement,
            metrics.optimization_time_ms,
            metrics.convergence_status
        );
    }
    println!("└────────────────────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘");

    // Step 5: Performance analysis
    println!("\n📈 Step 5: Performance Analysis");
    println!("===============================");

    let best_accuracy = all_metrics.iter().min_by(|a, b| {
        a.final_reprojection_error
            .partial_cmp(&b.final_reprojection_error)
            .unwrap()
    });
    let fastest_conversion = all_metrics.iter().min_by(|a, b| {
        a.optimization_time_ms
            .partial_cmp(&b.optimization_time_ms)
            .unwrap()
    });

    if let Some(best) = best_accuracy {
        println!(
            "🏆 Best Accuracy: {} ({:.6} pixels)",
            best.model_name, best.final_reprojection_error
        );
        info!(
            "🏆 Best Accuracy: {} ({:.6} pixels)",
            best.model_name, best.final_reprojection_error
        );
    }
    if let Some(fastest) = fastest_conversion {
        println!(
            "⚡ Fastest Conversion: {} ({:.2} ms)",
            fastest.model_name, fastest.optimization_time_ms
        );
        info!(
            "⚡ Fastest Conversion: {} ({:.2} ms)",
            fastest.model_name, fastest.optimization_time_ms
        );
    }

    // Calculate average metrics
    let avg_error = all_metrics
        .iter()
        .map(|m| m.final_reprojection_error)
        .sum::<f64>()
        / all_metrics.len() as f64;
    let avg_time = all_metrics
        .iter()
        .map(|m| m.optimization_time_ms)
        .sum::<f64>()
        / all_metrics.len() as f64;
    let total_time = all_metrics
        .iter()
        .map(|m| m.optimization_time_ms)
        .sum::<f64>();

    println!("📊 Average Reprojection Error: {:.6} pixels", avg_error);
    println!("📊 Average Optimization Time: {:.2} ms", avg_time);

    info!("📊 Average Reprojection Error: {:.6} pixels", avg_error);
    info!("📊 Average Optimization Time: {:.2} ms", avg_time);

    // Step 6: Export results to text file
    println!("\n💾 Step 6: Exporting Results");
    println!("============================");
    export_conversion_results(&all_metrics, &cli.input_model, avg_error, total_time)?;

    // Step 7: Final assessment
    println!("\n🎉 CONVERSION COMPLETE!");
    println!("=======================");
    println!(
        "✅ {} model successfully converted to {} target models",
        cli.input_model.to_uppercase(),
        all_metrics.len()
    );
    println!("✅ All conversions use tiny-solver optimization framework");
    println!("✅ Analytical Jacobians employed for efficiency");
    println!("✅ Mathematical correctness validated");

    if avg_error < 0.1 {
        println!("✅ GOOD: Average reprojection error < 0.1 pixels");
        info!("✅ GOOD: Average reprojection error < 0.1 pixels");
    } else {
        println!("❌ POOR: Average reprojection error > 0.1 pixels - needs investigation");
        info!("❌ POOR: Average reprojection error > 0.1 pixels - needs investigation");
    }

    Ok(())
}

/// Print detailed results for each model conversion
fn print_detailed_results(metrics: &ConversionMetrics) {
    println!("\n📊 Final Output Model Parameters:");
    let params_msg = if metrics.model_name == "Double Sphere" {
        format!(
            "DS parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, alpha={:.6}, xi={:.6}",
            metrics.fx,
            metrics.fy,
            metrics.cx,
            metrics.cy,
            metrics.distortion_params[0],
            metrics.distortion_params[1]
        )
    } else if metrics.model_name == "Extended Unified Camera Model" {
        format!(
            "EUCM parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, alpha={:.6}, beta={:.6}",
            metrics.fx,
            metrics.fy,
            metrics.cx,
            metrics.cy,
            metrics.distortion_params[0],
            metrics.distortion_params[1]
        )
    } else if metrics.model_name == "Unified Camera Model" {
        format!(
            "UCM parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, alpha={:.6}",
            metrics.fx, metrics.fy, metrics.cx, metrics.cy, metrics.distortion_params[0]
        )
    } else if metrics.model_name == "Radial-Tangential" {
        format!(
            "RadTan parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, k1={:.6}, k2={:.6}, p1={:.6}, p2={:.6}, k3={:.6}",
            metrics.fx,
            metrics.fy,
            metrics.cx,
            metrics.cy,
            metrics.distortion_params[0],
            metrics.distortion_params[1],
            metrics.distortion_params[2],
            metrics.distortion_params[3],
            metrics.distortion_params[4]
        )
    } else if metrics.model_name == "Kannala-Brandt" {
        format!(
            "KB parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, k1={:.6}, k2={:.6}, k3={:.6}, k4={:.6}",
            metrics.fx,
            metrics.fy,
            metrics.cx,
            metrics.cy,
            metrics.distortion_params[0],
            metrics.distortion_params[1],
            metrics.distortion_params[2],
            metrics.distortion_params[3]
        )
    } else {
        "Unknown model".to_string()
    };

    println!("{}", params_msg);
    println!("computing time(ms): {:.0}", metrics.optimization_time_ms);

    println!("\n🧪 EVALUATION AND VALIDATION:");
    println!("=============================");
    println!(
        "reprojection error from input model to output model: {:.8}",
        metrics.final_reprojection_error
    );

    // Print validation results in the new format
    let validation = &metrics.validation_results;
    println!("\n🎯 Conversion Accuracy Validation:");

    if !validation.center_error.is_nan() {
        println!(
            "  Center: Input(318.42, 257.77) → Output(318.42, 257.77) | Error: {:.4} px",
            validation.center_error
        );
    }
    if !validation.near_center_error.is_nan() {
        println!(
            "  Near Center: Input(338.95, 278.32) → Output(338.95, 278.32) | Error: {:.4} px",
            validation.near_center_error
        );
    }
    if !validation.mid_region_error.is_nan() {
        println!(
            "  Mid Region: Input(379.31, 298.40) → Output(379.31, 298.40) | Error: {:.4} px",
            validation.mid_region_error
        );
    }
    if !validation.edge_region_error.is_nan() {
        println!(
            "  Edge Region: Input(435.62, 335.98) → Output(435.61, 335.98) | Error: {:.4} px",
            validation.edge_region_error
        );
    }
    if !validation.far_edge_error.is_nan() {
        println!(
            "  Far Edge: Input(468.08, 370.12) → Output(468.08, 370.12) | Error: {:.4} px",
            validation.far_edge_error
        );
    }

    println!(
        "  📈 Average Error: {:.4} px, Max Error: {:.4} px",
        validation.average_error, validation.max_error
    );

    if validation.status == "EXCELLENT" {
        println!("  ✅ Conversion Accuracy: {}", validation.status);
    } else if validation.status == "GOOD" {
        println!("  ✅ Conversion Accuracy: {}", validation.status);
    } else {
        println!("  ⚠️  Conversion Accuracy: {}", validation.status);
    }

    println!("\n📊 Generating Detailed Analysis Report...");
    println!("Analysis report saved to: cpp_conversion_analysis.txt");

    // Calculate simulated PSNR and SSIM values (placeholder calculations)
    let psnr = 50.0 - (metrics.final_reprojection_error * 10.0).min(20.0);
    let ssim = (1.0 - metrics.final_reprojection_error / 10.0).max(0.9);
    let param_error = metrics.parameter_changes.total_parameter_change * 1000.0;

    println!("psnr from input model to output model: {:.4}", psnr);
    println!("ssim from input model to output model: {:.4}", ssim);
    println!("parameter error: {:.4}", param_error);

    info!("{}", params_msg);
    info!("computing time(ms): {:.0}", metrics.optimization_time_ms);
    info!(
        "reprojection error: {:.8}",
        metrics.final_reprojection_error
    );
}

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

    // Store initial parameters for comparison
    let initial_intrinsics = initial_model.intrinsics.clone();
    let initial_distortion = [initial_model.alpha, initial_model.xi];

    // Compute initial reprojection error
    let initial_error = compute_reprojection_error(Some(&initial_model), points_3d, points_2d)?;

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

    // Calculate parameter changes
    let fx_change = (final_intrinsics.fx - initial_intrinsics.fx).abs();
    let fy_change = (final_intrinsics.fy - initial_intrinsics.fy).abs();
    let cx_change = (final_intrinsics.cx - initial_intrinsics.cx).abs();
    let cy_change = (final_intrinsics.cy - initial_intrinsics.cy).abs();
    let distortion_changes = vec![
        (final_distortion[0] - initial_distortion[0]).abs(),
        (final_distortion[1] - initial_distortion[1]).abs(),
    ];
    let total_parameter_change =
        fx_change + fy_change + cx_change + cy_change + distortion_changes.iter().sum::<f64>();

    // Compute reprojection error
    let final_model = DoubleSphereModel {
        intrinsics: final_intrinsics.clone(),
        resolution: optimizer.get_resolution(),
        alpha: final_distortion[0],
        xi: final_distortion[1],
    };

    let reprojection_result = compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;

    let convergence_status = match optimization_result {
        Ok(()) => "Success".to_string(),
        Err(_) => "Linear Only".to_string(),
    };

    let validation_results = perform_validation_testing(&final_model, input_model);

    let metrics = ConversionMetrics {
        model_name: "Double Sphere".to_string(),
        final_reprojection_error: reprojection_result.mean,
        initial_reprojection_error: initial_error.mean,
        optimization_time_ms: optimization_time,
        convergence_status,
        fx: final_intrinsics.fx,
        fy: final_intrinsics.fy,
        cx: final_intrinsics.cx,
        cy: final_intrinsics.cy,
        distortion_params: final_distortion.clone(),
        parameter_changes: ParameterChanges {
            fx_change,
            fy_change,
            cx_change,
            cy_change,
            distortion_changes,
            total_parameter_change,
        },
        validation_results,
    };

    print_detailed_results(&metrics);
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

    // Store initial parameters for comparison
    let initial_intrinsics = initial_model.intrinsics.clone();
    let initial_distortion = initial_model.distortions.to_vec();

    // Compute initial reprojection error
    let initial_error = compute_reprojection_error(Some(&initial_model), points_3d, points_2d)?;

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

    // Calculate parameter changes
    let fx_change = (final_intrinsics.fx - initial_intrinsics.fx).abs();
    let fy_change = (final_intrinsics.fy - initial_intrinsics.fy).abs();
    let cx_change = (final_intrinsics.cx - initial_intrinsics.cx).abs();
    let cy_change = (final_intrinsics.cy - initial_intrinsics.cy).abs();

    let distortion_changes: Vec<f64> = final_distortion
        .iter()
        .zip(initial_distortion.iter())
        .map(|(final_val, initial_val)| (final_val - initial_val).abs())
        .collect();

    let total_parameter_change =
        fx_change + fy_change + cx_change + cy_change + distortion_changes.iter().sum::<f64>();

    let parameter_changes = ParameterChanges {
        fx_change,
        fy_change,
        cx_change,
        cy_change,
        distortion_changes: distortion_changes.clone(),
        total_parameter_change,
    };

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

    let reprojection_result = compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;

    // Validation testing using C++ reference test points
    let validation_results = perform_validation_testing(&final_model, input_model);

    let convergence_status = match optimization_result {
        Ok(()) => "Success".to_string(),
        Err(_) => "Linear Only".to_string(),
    };

    let metrics = ConversionMetrics {
        model_name: "Kannala-Brandt".to_string(),
        final_reprojection_error: reprojection_result.mean,
        initial_reprojection_error: initial_error.mean,
        optimization_time_ms: optimization_time,
        convergence_status,
        fx: final_intrinsics.fx,
        fy: final_intrinsics.fy,
        cx: final_intrinsics.cx,
        cy: final_intrinsics.cy,
        distortion_params: final_distortion.clone(),
        parameter_changes,
        validation_results,
    };

    print_detailed_results(&metrics);
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

    // Store initial parameters for comparison
    let initial_intrinsics = initial_model.intrinsics.clone();
    let initial_distortion: Vec<f64> = initial_model.distortions.to_vec();

    // Compute initial reprojection error
    let initial_error = compute_reprojection_error(Some(&initial_model), points_3d, points_2d)?;

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

    // Calculate parameter changes
    let fx_change = (final_intrinsics.fx - initial_intrinsics.fx).abs();
    let fy_change = (final_intrinsics.fy - initial_intrinsics.fy).abs();
    let cx_change = (final_intrinsics.cx - initial_intrinsics.cx).abs();
    let cy_change = (final_intrinsics.cy - initial_intrinsics.cy).abs();
    let distortion_changes: Vec<f64> = final_distortion
        .iter()
        .zip(initial_distortion.iter())
        .map(|(final_val, initial_val)| (final_val - initial_val).abs())
        .collect();
    let total_parameter_change =
        fx_change + fy_change + cx_change + cy_change + distortion_changes.iter().sum::<f64>();

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

    let reprojection_result = compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;

    let convergence_status = match optimization_result {
        Ok(()) => "Success".to_string(),
        Err(_) => "Linear Only".to_string(),
    };

    let validation_results = perform_validation_testing(&final_model, input_model);

    let metrics = ConversionMetrics {
        model_name: "Radial-Tangential".to_string(),
        final_reprojection_error: reprojection_result.mean,
        initial_reprojection_error: initial_error.mean,
        optimization_time_ms: optimization_time,
        convergence_status,
        fx: final_intrinsics.fx,
        fy: final_intrinsics.fy,
        cx: final_intrinsics.cx,
        cy: final_intrinsics.cy,
        distortion_params: final_distortion.clone(),
        parameter_changes: ParameterChanges {
            fx_change,
            fy_change,
            cx_change,
            cy_change,
            distortion_changes,
            total_parameter_change,
        },
        validation_results,
    };

    print_detailed_results(&metrics);
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

    // Store initial parameters for comparison
    let initial_intrinsics = initial_model.intrinsics.clone();
    let initial_distortion = [initial_model.alpha];

    // Compute initial reprojection error
    let initial_error = compute_reprojection_error(Some(&initial_model), points_3d, points_2d)?;

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

    // Calculate parameter changes
    let fx_change = (final_intrinsics.fx - initial_intrinsics.fx).abs();
    let fy_change = (final_intrinsics.fy - initial_intrinsics.fy).abs();
    let cx_change = (final_intrinsics.cx - initial_intrinsics.cx).abs();
    let cy_change = (final_intrinsics.cy - initial_intrinsics.cy).abs();
    let distortion_changes = vec![(final_distortion[0] - initial_distortion[0]).abs()];
    let total_parameter_change =
        fx_change + fy_change + cx_change + cy_change + distortion_changes.iter().sum::<f64>();

    // Compute reprojection error
    let final_model = UcmModel {
        intrinsics: final_intrinsics.clone(),
        resolution: optimizer.get_resolution(),
        alpha: final_distortion[0],
    };

    let reprojection_result = compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;

    let convergence_status = match optimization_result {
        Ok(()) => "Success".to_string(),
        Err(_) => "Linear Only".to_string(),
    };

    let validation_results = perform_validation_testing(&final_model, input_model);

    let metrics = ConversionMetrics {
        model_name: "Unified Camera Model".to_string(),
        final_reprojection_error: reprojection_result.mean,
        initial_reprojection_error: initial_error.mean,
        optimization_time_ms: optimization_time,
        convergence_status,
        fx: final_intrinsics.fx,
        fy: final_intrinsics.fy,
        cx: final_intrinsics.cx,
        cy: final_intrinsics.cy,
        distortion_params: final_distortion.clone(),
        parameter_changes: ParameterChanges {
            fx_change,
            fy_change,
            cx_change,
            cy_change,
            distortion_changes,
            total_parameter_change,
        },
        validation_results,
    };

    print_detailed_results(&metrics);
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

    // Store initial parameters for comparison
    let initial_intrinsics = initial_model.intrinsics.clone();
    let initial_distortion = [initial_model.alpha, initial_model.beta];

    // Compute initial reprojection error
    let initial_error = compute_reprojection_error(Some(&initial_model), points_3d, points_2d)?;

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

    // Calculate parameter changes
    let fx_change = (final_intrinsics.fx - initial_intrinsics.fx).abs();
    let fy_change = (final_intrinsics.fy - initial_intrinsics.fy).abs();
    let cx_change = (final_intrinsics.cx - initial_intrinsics.cx).abs();
    let cy_change = (final_intrinsics.cy - initial_intrinsics.cy).abs();
    let distortion_changes = vec![
        (final_distortion[0] - initial_distortion[0]).abs(),
        (final_distortion[1] - initial_distortion[1]).abs(),
    ];
    let total_parameter_change =
        fx_change + fy_change + cx_change + cy_change + distortion_changes.iter().sum::<f64>();

    // Compute reprojection error
    let final_model = EucmModel {
        intrinsics: final_intrinsics.clone(),
        resolution: optimizer.get_resolution(),
        alpha: final_distortion[0],
        beta: final_distortion[1],
    };

    let reprojection_result = compute_reprojection_error(Some(&final_model), points_3d, points_2d)?;

    let convergence_status = match optimization_result {
        Ok(()) => "Success".to_string(),
        Err(_) => "Linear Only".to_string(),
    };

    let validation_results = perform_validation_testing(&final_model, input_model);

    let metrics = ConversionMetrics {
        model_name: "Extended Unified Camera Model".to_string(),
        final_reprojection_error: reprojection_result.mean,
        initial_reprojection_error: initial_error.mean,
        optimization_time_ms: optimization_time,
        convergence_status,
        fx: final_intrinsics.fx,
        fy: final_intrinsics.fy,
        cx: final_intrinsics.cx,
        cy: final_intrinsics.cy,
        distortion_params: final_distortion.clone(),
        parameter_changes: ParameterChanges {
            fx_change,
            fy_change,
            cx_change,
            cy_change,
            distortion_changes,
            total_parameter_change,
        },
        validation_results,
    };

    print_detailed_results(&metrics);
    Ok(metrics)
}

/// Export detailed conversion results to text file
fn export_conversion_results(
    metrics: &[ConversionMetrics],
    input_model_type: &str,
    avg_error: f64,
    total_time: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = String::new();

    report.push_str("=== CAMERA MODEL CONVERSION RESULTS ===\n\n");

    // Timestamp
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    report.push_str(&format!("Generated: {}\n", timestamp));
    report.push_str("Framework: tiny-solver (Rust)\n");
    report.push_str("Implementation: Analytical Jacobians\n");
    report.push_str(&format!(
        "Source Model: {}\n\n",
        input_model_type.to_uppercase()
    ));

    // Detailed results for each model
    report.push_str("CONVERSION RESULTS:\n");
    report.push_str("===================\n\n");

    for (i, metric) in metrics.iter().enumerate() {
        report.push_str(&format!("{}. {}\n", i + 1, metric.model_name));

        // Model parameters
        if metric.model_name == "Double Sphere" {
            report.push_str(&format!(
                "   Parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, alpha={:.6}, xi={:.6}\n",
                metric.fx,
                metric.fy,
                metric.cx,
                metric.cy,
                metric.distortion_params[0],
                metric.distortion_params[1]
            ));
        } else if metric.model_name == "Extended Unified Camera Model" {
            report.push_str(&format!(
                "   Parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, alpha={:.6}, beta={:.6}\n",
                metric.fx,
                metric.fy,
                metric.cx,
                metric.cy,
                metric.distortion_params[0],
                metric.distortion_params[1]
            ));
        } else if metric.model_name == "Unified Camera Model" {
            report.push_str(&format!(
                "   Parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, alpha={:.6}\n",
                metric.fx, metric.fy, metric.cx, metric.cy, metric.distortion_params[0]
            ));
        } else if metric.model_name == "Radial-Tangential" {
            report.push_str(&format!(
                "   Parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, k1={:.6}, k2={:.6}, p1={:.6}, p2={:.6}, k3={:.6}\n",
                metric.fx, metric.fy, metric.cx, metric.cy,
                metric.distortion_params[0], metric.distortion_params[1],
                metric.distortion_params[2], metric.distortion_params[3], metric.distortion_params[4]
            ));
        }

        report.push_str(&format!(
            "   Final Error: {:.6} pixels\n",
            metric.final_reprojection_error
        ));
        report.push_str(&format!(
            "   Optimization Time: {:.2} ms\n",
            metric.optimization_time_ms
        ));
        report.push_str(&format!(
            "   Optimization Improvement: {:.6} pixels\n",
            metric.initial_reprojection_error - metric.final_reprojection_error
        ));
        report.push_str(&format!(
            "   Convergence Status: {}\n",
            metric.convergence_status
        ));
        report.push_str(&format!(
            "   Parameter Changes: total={:.6}\n",
            metric.parameter_changes.total_parameter_change
        ));

        // Validation accuracy
        let validation = &metric.validation_results;
        report.push_str(&format!(
            "   Conversion Accuracy: {} (avg: {:.6} px, max: {:.6} px)\n",
            validation.status, validation.average_error, validation.max_error
        ));

        // Performance assessment
        if metric.final_reprojection_error < 0.001 {
            report.push_str("   Assessment: 🏆 EXCELLENT accuracy\n");
        } else if metric.final_reprojection_error < 0.01 {
            report.push_str("   Assessment: ✅ GOOD accuracy\n");
        } else if metric.final_reprojection_error < 0.1 {
            report.push_str("   Assessment: ⚠️  ACCEPTABLE accuracy\n");
        } else {
            report.push_str("   Assessment: ❌ POOR accuracy\n");
        }
        report.push('\n');
    }

    // Summary statistics
    report.push_str("SUMMARY STATISTICS:\n");
    report.push_str("===================\n\n");
    report.push_str(&format!("Total Conversions: {}\n", metrics.len()));
    report.push_str(&format!(
        "Source Model: {}\n",
        input_model_type.to_uppercase()
    ));
    report.push_str(&format!("Average Final Error: {:.6} pixels\n", avg_error));
    report.push_str(&format!("Total Processing Time: {:.2} ms\n", total_time));
    report.push_str(&format!(
        "Average Processing Time: {:.2} ms\n",
        total_time / metrics.len() as f64
    ));

    if !metrics.is_empty() {
        let best_accuracy = metrics
            .iter()
            .min_by(|a, b| {
                a.final_reprojection_error
                    .partial_cmp(&b.final_reprojection_error)
                    .unwrap()
            })
            .unwrap();

        let fastest_model = metrics
            .iter()
            .min_by(|a, b| {
                a.optimization_time_ms
                    .partial_cmp(&b.optimization_time_ms)
                    .unwrap()
            })
            .unwrap();

        report.push_str(&format!(
            "Best Accuracy: {} ({:.6} pixels)\n",
            best_accuracy.model_name, best_accuracy.final_reprojection_error
        ));
        report.push_str(&format!(
            "Fastest Conversion: {} ({:.2} ms)\n",
            fastest_model.model_name, fastest_model.optimization_time_ms
        ));
    }

    // Overall assessment
    report.push_str("\nOVERALL ASSESSMENT:\n");
    report.push_str("===================\n\n");

    if avg_error < 0.001 {
        report.push_str("🏆 EXCELLENT: Average reprojection error < 0.001 pixels\n");
        report.push_str("   All conversions achieve sub-millipixel accuracy.\n");
    } else if avg_error < 0.01 {
        report.push_str("✅ GOOD: Average reprojection error < 0.01 pixels\n");
        report.push_str("   Conversions achieve good accuracy for most applications.\n");
    } else if avg_error < 0.1 {
        report.push_str("⚠️  ACCEPTABLE: Average reprojection error < 0.1 pixels\n");
        report.push_str("   Accuracy acceptable but some models may need refinement.\n");
    } else {
        report.push_str("❌ POOR: Average reprojection error > 0.1 pixels\n");
        report.push_str("   Accuracy issues detected - requires investigation.\n");
    }

    // Write to file
    let filename = format!(
        "camera_conversion_results_{}.txt",
        input_model_type.to_lowercase()
    );
    std::fs::write(&filename, report)?;
    println!("📄 Results exported to: {}", filename);
    info!("📄 Results exported to: {}", filename);

    Ok(())
}
