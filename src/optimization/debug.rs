//! Debugging and analysis tools for camera calibration optimization.
//!
//! This module provides comprehensive tools to investigate performance discrepancies
//! between C++ and Rust implementations, analyze convergence behavior, and validate
//! mathematical correctness of the optimization process.

use crate::camera::{CameraModel, CameraModelError};
use nalgebra::{Matrix2xX, Matrix3xX, Vector2, Vector3};
use serde::{Deserialize, Serialize};

/// Detailed optimization statistics for debugging and analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStats {
    /// Number of iterations performed
    pub iterations: usize,
    /// Initial cost value
    pub initial_cost: f64,
    /// Final cost value
    pub final_cost: f64,
    /// Cost values per iteration
    pub cost_history: Vec<f64>,
    /// Parameter values per iteration
    pub parameter_history: Vec<Vec<f64>>,
    /// Convergence reason
    pub convergence_reason: String,
    /// Total optimization time in milliseconds
    pub optimization_time_ms: f64,
    /// Final reprojection error (RMS)
    pub final_reprojection_error: f64,
    /// Initial parameters
    pub initial_parameters: Vec<f64>,
    /// Final parameters
    pub final_parameters: Vec<f64>,
}

/// Comprehensive debugging data for optimization analysis.
#[derive(Debug, Clone)]
pub struct OptimizationDebugData {
    /// Input 3D points
    pub points_3d: Vec<Vector3<f64>>,
    /// Input 2D points
    pub points_2d: Vec<Vector2<f64>>,
    /// Projected 2D points using initial parameters
    pub initial_projections: Vec<Vector2<f64>>,
    /// Projected 2D points using final parameters
    pub final_projections: Vec<Vector2<f64>>,
    /// Per-point residuals (initial)
    pub initial_residuals: Vec<Vector2<f64>>,
    /// Per-point residuals (final)
    pub final_residuals: Vec<Vector2<f64>>,
    /// Jacobian matrices for each point (final parameters)
    pub jacobians: Vec<Vec<Vec<f64>>>, // [point_idx][row][col]
    /// Optimization statistics
    pub stats: OptimizationStats,
}

/// Comparison data between C++ and Rust implementations.
#[derive(Debug, Clone)]
pub struct ImplementationComparison {
    /// C++ optimization results
    pub cpp_data: OptimizationDebugData,
    /// Rust optimization results
    pub rust_data: OptimizationDebugData,
    /// Parameter differences
    pub parameter_differences: Vec<f64>,
    /// Projection differences (per point)
    pub projection_differences: Vec<f64>,
    /// Analysis summary
    pub analysis_summary: String,
}

/// Debugging utilities for optimization analysis.
pub struct OptimizationDebugger;

impl OptimizationDebugger {
    /// Compute reprojection error for a set of points and camera model.
    pub fn compute_reprojection_error<T: CameraModel>(
        model: &T,
        points_3d: &Matrix3xX<f64>,
        points_2d: &Matrix2xX<f64>,
    ) -> Result<f64, CameraModelError> {
        let mut total_error = 0.0;
        let mut valid_points = 0;

        for i in 0..points_3d.ncols() {
            let p3d = Vector3::new(points_3d[(0, i)], points_3d[(1, i)], points_3d[(2, i)]);
            let p2d_observed = Vector2::new(points_2d[(0, i)], points_2d[(1, i)]);

            match model.project(&p3d) {
                Ok(p2d_projected) => {
                    let error = (p2d_projected - p2d_observed).norm();
                    total_error += error * error;
                    valid_points += 1;
                }
                Err(_) => {
                    // Skip invalid projections
                    continue;
                }
            }
        }

        if valid_points == 0 {
            return Err(CameraModelError::NumericalError(
                "No valid projections found".to_string(),
            ));
        }

        Ok((total_error / valid_points as f64).sqrt())
    }

    /// Analyze projection differences between two camera models.
    pub fn analyze_projection_differences<T: CameraModel>(
        model1: &T,
        model2: &T,
        points_3d: &Matrix3xX<f64>,
        label1: &str,
        label2: &str,
    ) -> Result<String, CameraModelError> {
        let mut differences = Vec::new();
        let mut max_diff = 0.0_f64;
        let mut total_diff = 0.0_f64;
        let mut valid_comparisons = 0;

        for i in 0..points_3d.ncols() {
            let p3d = Vector3::new(points_3d[(0, i)], points_3d[(1, i)], points_3d[(2, i)]);

            match (model1.project(&p3d), model2.project(&p3d)) {
                (Ok(proj1), Ok(proj2)) => {
                    let diff = (proj1 - proj2).norm();
                    differences.push(diff);
                    max_diff = max_diff.max(diff);
                    total_diff += diff;
                    valid_comparisons += 1;
                }
                _ => continue,
            }
        }

        if valid_comparisons == 0 {
            return Ok("No valid projection comparisons possible".to_string());
        }

        let mean_diff = total_diff / valid_comparisons as f64;
        differences.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_diff = differences[valid_comparisons / 2];

        Ok(format!(
            "Projection Differences ({} vs {}):\n\
             - Valid comparisons: {}\n\
             - Mean difference: {:.6} pixels\n\
             - Median difference: {:.6} pixels\n\
             - Max difference: {:.6} pixels\n\
             - Min difference: {:.6} pixels",
            label1, label2, valid_comparisons, mean_diff, median_diff, max_diff, differences[0]
        ))
    }

    /// Export optimization data to a simple text format for external analysis.
    pub fn export_debug_data(
        data: &OptimizationDebugData,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut output = String::new();
        output.push_str("Optimization Statistics:\n");
        output.push_str(&format!("  Iterations: {}\n", data.stats.iterations));
        output.push_str(&format!(
            "  Initial Cost: {:.6e}\n",
            data.stats.initial_cost
        ));
        output.push_str(&format!("  Final Cost: {:.6e}\n", data.stats.final_cost));
        output.push_str(&format!(
            "  Final Reprojection Error: {:.6}\n",
            data.stats.final_reprojection_error
        ));
        output.push_str(&format!(
            "  Optimization Time: {:.2} ms\n",
            data.stats.optimization_time_ms
        ));
        output.push_str(&format!(
            "  Convergence: {}\n",
            data.stats.convergence_reason
        ));

        std::fs::write(filename, output)?;
        println!("Debug data exported to: {}", filename);
        Ok(())
    }

    /// Generate a comprehensive analysis report.
    pub fn generate_analysis_report(comparison: &ImplementationComparison) -> String {
        let cpp_stats = &comparison.cpp_data.stats;
        let rust_stats = &comparison.rust_data.stats;

        format!(
            "=== OPTIMIZATION ANALYSIS REPORT ===\n\n\
             C++ Implementation:\n\
             - Iterations: {}\n\
             - Initial Cost: {:.6e}\n\
             - Final Cost: {:.6e}\n\
             - Final Reprojection Error: {:.6} pixels\n\
             - Optimization Time: {:.2} ms\n\
             - Convergence: {}\n\n\
             Rust Implementation:\n\
             - Iterations: {}\n\
             - Initial Cost: {:.6e}\n\
             - Final Cost: {:.6e}\n\
             - Final Reprojection Error: {:.6} pixels\n\
             - Optimization Time: {:.2} ms\n\
             - Convergence: {}\n\n\
             Parameter Differences:\n\
             {}\n\n\
             Analysis Summary:\n\
             {}",
            cpp_stats.iterations,
            cpp_stats.initial_cost,
            cpp_stats.final_cost,
            cpp_stats.final_reprojection_error,
            cpp_stats.optimization_time_ms,
            cpp_stats.convergence_reason,
            rust_stats.iterations,
            rust_stats.initial_cost,
            rust_stats.final_cost,
            rust_stats.final_reprojection_error,
            rust_stats.optimization_time_ms,
            rust_stats.convergence_reason,
            format_parameter_differences(&comparison.parameter_differences),
            comparison.analysis_summary
        )
    }
}

/// Format parameter differences for display.
fn format_parameter_differences(differences: &[f64]) -> String {
    differences
        .iter()
        .enumerate()
        .map(|(i, diff)| format!("  Param {}: {:.6}", i, diff))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Validate Jacobian accuracy by comparing analytical vs numerical derivatives.
pub fn validate_jacobian_accuracy(
    analytical_jacobian: &[Vec<f64>],
    numerical_jacobian: &[Vec<f64>],
    tolerance: f64,
) -> Result<bool, String> {
    if analytical_jacobian.len() != numerical_jacobian.len() {
        return Err("Jacobian dimension mismatch (rows)".to_string());
    }

    let mut max_error = 0.0_f64;
    let mut total_error = 0.0_f64;
    let mut num_elements = 0;

    for (i, (anal_row, num_row)) in analytical_jacobian
        .iter()
        .zip(numerical_jacobian.iter())
        .enumerate()
    {
        if anal_row.len() != num_row.len() {
            return Err(format!("Jacobian dimension mismatch at row {} (cols)", i));
        }

        for (j, (anal_val, num_val)) in anal_row.iter().zip(num_row.iter()).enumerate() {
            let error = (anal_val - num_val).abs();
            max_error = max_error.max(error);
            total_error += error;
            num_elements += 1;

            if error > tolerance {
                return Err(format!(
                    "Jacobian error exceeds tolerance at ({}, {}): analytical={:.6}, numerical={:.6}, error={:.6}",
                    i, j, anal_val, num_val, error
                ));
            }
        }
    }

    let mean_error = total_error / num_elements as f64;
    println!(
        "Jacobian validation passed: mean_error={:.6}, max_error={:.6}",
        mean_error, max_error
    );
    Ok(true)
}
