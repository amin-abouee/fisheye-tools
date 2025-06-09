//! This module provides the cost function and optimization routines
//! for calibrating an Extended Unified Camera Model (EUCM).
//!
//! It uses the `tiny_solver` crate for non-linear optimization and defines
//! the necessary structures and traits to integrate the EUCM
//! camera model with the optimization framework.

use crate::camera::{CameraModel, CameraModelError, EucmModel};
use crate::optimization::Optimizer;

use log::info;
use nalgebra::{DVector, Matrix2xX, Matrix3xX, Vector2, Vector3};
use std::collections::HashMap;
use std::fmt;
use tiny_solver::factors::Factor;
use tiny_solver::{LevenbergMarquardtOptimizer, Optimizer as TinySolverOptimizer};

/// Cost function for EUCM camera model optimization.
///
/// This structure holds the 3D-2D point correspondences and the camera model
/// instance used during camera calibration optimization. It implements the
/// [`Optimizer`] trait for use with the optimization framework.
#[derive(Clone)]
pub struct EucmOptimizationCost {
    /// The EUCM camera model to be optimized.
    model: EucmModel,
    /// 3D points in the camera's coordinate system (3×N matrix).
    /// Each column represents a 3D point.
    points3d: Matrix3xX<f64>,
    /// Corresponding 2D points in image coordinates (2×N matrix).
    /// Each column represents a 2D point observed in the image.
    points2d: Matrix2xX<f64>,
}

/// Cost function for `tiny_solver` optimization of the [`EucmModel`].
///
/// This struct defines the residual error between the observed 2D point and
/// the projected 2D point from a 3D point using the current camera model parameters.
/// It is used by the `tiny_solver` optimization framework.
#[derive(Debug, Clone)]
struct EUCMCost {
    /// 3D points in the camera's coordinate system.
    points3d: Vec<Vector3<f64>>,
    /// Corresponding observed 2D points in image coordinates.
    points2d: Vec<Vector2<f64>>,
}

impl EUCMCost {
    /// Creates a new residual for a set of 3D-2D point correspondences.
    pub fn new(points3d: &Matrix3xX<f64>, points2d: &Matrix2xX<f64>) -> Self {
        let points3d = (0..points3d.ncols())
            .map(|i| points3d.column(i).into_owned())
            .collect();
        let points2d = (0..points2d.ncols())
            .map(|i| points2d.column(i).into_owned())
            .collect();
        Self { points3d, points2d }
    }
}

impl<T: nalgebra::RealField> Factor<T> for EUCMCost {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        let cam_params = &params[0];
        let fx = cam_params[0].clone();
        let fy = cam_params[1].clone();
        let cx = cam_params[2].clone();
        let cy = cam_params[3].clone();
        let alpha = cam_params[4].clone();
        let beta = cam_params[5].clone();

        let mut residuals = DVector::zeros(self.points2d.len() * 2);

        for i in 0..self.points2d.len() {
            let p3d = &self.points3d[i];
            let p2d = &self.points2d[i];

            let gt_u = T::from_f64(p2d.x).unwrap();
            let gt_v = T::from_f64(p2d.y).unwrap();
            let obs_x = T::from_f64(p3d.x).unwrap();
            let obs_y = T::from_f64(p3d.y).unwrap();
            let obs_z = T::from_f64(p3d.z).unwrap();

            // EUCM projection model implementation (matching C++ exactly)
            let r_squared = obs_x.clone() * obs_x.clone() + obs_y.clone() * obs_y.clone();
            let d = (beta.clone() * r_squared + obs_z.clone() * obs_z.clone()).sqrt();
            let denom =
                alpha.clone() * d + (T::from_f64(1.0).unwrap() - alpha.clone()) * obs_z.clone();

            let precision = T::from_f64(1e-3).unwrap();

            // Check projection validity (matching C++ implementation)
            let z_f64 = p3d.z;
            let denom_f64 = denom.clone().to_subset().unwrap_or(0.0);
            let alpha_f64 = alpha.clone().to_subset().unwrap_or(0.0);
            let beta_f64 = beta.clone().to_subset().unwrap_or(0.0);

            if denom < precision
                || !EucmModel::check_proj_condition(z_f64, denom_f64, alpha_f64, beta_f64)
            {
                residuals[i * 2] = T::from_f64(1e6).unwrap();
                residuals[i * 2 + 1] = T::from_f64(1e6).unwrap();
            } else {
                // Use the exact C++ residual formula
                // residuals[0] = fx * obs_x_ - u_cx * denom;
                // residuals[1] = fy * obs_y_ - v_cy * denom;
                // where u_cx = gt_u_ - cx and v_cy = gt_v_ - cy
                let u_cx = gt_u.clone() - cx.clone();
                let v_cy = gt_v.clone() - cy.clone();

                residuals[i * 2] = fx.clone() * obs_x.clone() - u_cx * denom.clone();
                residuals[i * 2 + 1] = fy.clone() * obs_y.clone() - v_cy * denom;
            }
        }
        residuals
    }
}

impl EucmOptimizationCost {
    /// Creates a new optimization cost function for the EUCM camera model.
    ///
    /// # Arguments
    ///
    /// * `model` - The initial EUCM camera model to be optimized.
    /// * `points3d` - A 3×N matrix where each column represents a 3D point in camera coordinates.
    /// * `points2d` - A 2×N matrix where each column represents the corresponding 2D observation.
    ///
    /// # Returns
    ///
    /// A new `EucmOptimizationCost` instance ready for optimization.
    pub fn new(model: EucmModel, points3d: Matrix3xX<f64>, points2d: Matrix2xX<f64>) -> Self {
        Self {
            model,
            points3d,
            points2d,
        }
    }

    /// Returns a reference to the optimized camera model.
    ///
    /// This method should be called after optimization to retrieve the updated model parameters.
    pub fn get_model(&self) -> &EucmModel {
        &self.model
    }
}

impl fmt::Debug for EucmOptimizationCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EucmOptimizationCost Summary:\n model: {:?}\n points3d size: {}, points2d size: {} ",
            self.model,
            self.points3d.ncols(),
            self.points2d.ncols(),
        )
    }
}

impl Optimizer for EucmOptimizationCost {
    /// Performs non-linear optimization to refine the EUCM camera model parameters.
    ///
    /// This method uses the `tiny_solver` crate with Levenberg-Marquardt optimization
    /// to minimize the reprojection error between observed 2D points and projected
    /// 3D points using the current camera model.
    ///
    /// # Arguments
    ///
    /// * `verbose` - If `true`, prints optimization progress and results.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If optimization was successful and model parameters were updated.
    /// * `Err(CameraModelError)` - If optimization failed or parameters are invalid.
    fn optimize(&mut self, verbose: bool) -> Result<(), CameraModelError> {
        if self.points3d.ncols() != self.points2d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }

        if self.points3d.ncols() == 0 {
            return Err(CameraModelError::InvalidParams(
                "Points arrays cannot be empty".to_string(),
            ));
        }

        if verbose {
            info!("Starting EUCM camera model optimization...");
            info!("Initial model: {:?}", self.model);
            info!("Number of point correspondences: {}", self.points3d.ncols());
        }

        // Create a tiny_solver Problem
        let mut problem = tiny_solver::Problem::new();

        // Create the cost function
        let cost_function = EUCMCost::new(&self.points3d, &self.points2d);
        let num_residuals = self.points2d.ncols() * 2;
        problem.add_residual_block(num_residuals, &["params"], Box::new(cost_function), None);

        // Set parameter bounds
        problem.set_variable_bounds("params", 4, 1e-6, 1.0); // alpha bounds
        problem.set_variable_bounds("params", 5, 1e-6, 1.0); // beta bounds

        // Initial parameters
        let initial_params = DVector::from_vec(vec![
            self.model.intrinsics.fx,
            self.model.intrinsics.fy,
            self.model.intrinsics.cx,
            self.model.intrinsics.cy,
            self.model.alpha.clamp(1e-6, 1.0), // Clamp alpha to valid range
            self.model.beta.clamp(1e-6, 1.0),  // Clamp beta to valid range
        ]);
        let mut initial_values = HashMap::new();
        initial_values.insert("params".to_string(), initial_params);

        if verbose {
            info!("Starting EUCM optimization with tiny_solver Levenberg-Marquardt...");
        }

        // Create a Levenberg-Marquardt optimizer
        let optimizer = LevenbergMarquardtOptimizer::default();

        // Run the optimization
        let result = optimizer
            .optimize(&problem, &initial_values, None)
            .ok_or_else(|| CameraModelError::NumericalError("Optimization failed".to_string()))?;

        if verbose {
            info!("EUCM optimization finished");
        }

        // Extract the optimized parameters
        let optimized_params = result.get("params").unwrap();

        // Update the model parameters
        self.model.intrinsics.fx = optimized_params[0];
        self.model.intrinsics.fy = optimized_params[1];
        self.model.intrinsics.cx = optimized_params[2];
        self.model.intrinsics.cy = optimized_params[3];
        self.model.alpha = optimized_params[4].clamp(1e-6, 1.0); // Ensure bounds
        self.model.beta = optimized_params[5].clamp(1e-6, 1.0); // Ensure bounds

        // Validate the optimized parameters
        self.model.validate_params()?;

        if verbose {
            info!("Optimized EUCM model: {:?}", self.model);
        }

        Ok(())
    }

    /// Performs linear estimation of the alpha parameter for the EUCM model.
    ///
    /// This method follows the C++ reference implementation exactly:
    /// - Sets beta = 1.0 initially (fixed)
    /// - Solves only for alpha using a linear system
    /// - The intrinsic parameters `fx, fy, cx, cy` are assumed to be known and fixed.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the linear estimation was successful and alpha has been updated.
    /// * `Err(CameraModelError)` - If an error occurred, such as mismatched point
    ///   counts or numerical issues in solving the linear system.
    fn linear_estimation(&mut self) -> Result<(), CameraModelError> {
        if self.points2d.ncols() != self.points3d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }

        let num_points = self.points2d.ncols();
        if num_points < 1 {
            return Err(CameraModelError::InvalidParams(
                "Need at least 1 point for EUCM linear estimation".to_string(),
            ));
        }

        // Set beta to 1.0 (following C++ reference implementation)
        self.model.beta = 1.0;

        // Set up the linear system to solve for alpha only
        // Following the C++ reference: A(i*2, 0) = u_cx * (d - Z), b[i*2] = fx*X - u_cx*Z
        let mut a = nalgebra::DMatrix::zeros(num_points * 2, 1);
        let mut b = nalgebra::DVector::zeros(num_points * 2);

        for i in 0..num_points {
            let x = self.points3d[(0, i)];
            let y = self.points3d[(1, i)];
            let z = self.points3d[(2, i)];
            let u = self.points2d[(0, i)];
            let v = self.points2d[(1, i)];

            // Following C++ reference: d = sqrt(X*X + Y*Y + Z*Z)
            let d = (x * x + y * y + z * z).sqrt();
            let u_cx = u - self.model.intrinsics.cx;
            let v_cy = v - self.model.intrinsics.cy;

            // C++ reference linear system for alpha:
            // A(i * 2, 0) = u_cx * (d - Z);
            // A(i * 2 + 1, 0) = v_cy * (d - Z);
            // b[i * 2] = (fx * X) - (u_cx * Z);
            // b[i * 2 + 1] = (fy * Y) - (v_cy * Z);
            a[(i * 2, 0)] = u_cx * (d - z);
            a[(i * 2 + 1, 0)] = v_cy * (d - z);

            b[i * 2] = self.model.intrinsics.fx * x - u_cx * z;
            b[i * 2 + 1] = self.model.intrinsics.fy * y - v_cy * z;
        }

        // Solve the linear system using SVD
        let svd = a.svd(true, true);
        let solution = match svd.solve(&b, 1e-10) {
            Ok(sol) => sol,
            Err(err_msg) => {
                return Err(CameraModelError::NumericalError(err_msg.to_string()));
            }
        };

        self.model.alpha = solution[0];

        info!(
            "EUCM linear estimation results: alpha = {}, beta = {} (fixed)",
            self.model.alpha, self.model.beta
        );

        // Clamp alpha to valid range if needed
        if self.model.alpha <= 0.0 {
            info!("Alpha {} is too small, clamping to 0.01", self.model.alpha);
            self.model.alpha = 0.01;
        } else if self.model.alpha > 2.0 {
            info!("Alpha {} is too large, clamping to 2.0", self.model.alpha);
            self.model.alpha = 2.0;
        }

        // Validate parameters
        self.model.validate_params()?;

        Ok(())
    }

    /// Returns a clone of the current intrinsic parameters of the camera model.
    fn get_intrinsics(&self) -> crate::camera::Intrinsics {
        self.model.intrinsics.clone()
    }

    /// Returns a clone of the current resolution of the camera model.
    fn get_resolution(&self) -> crate::camera::Resolution {
        self.model.resolution.clone()
    }

    /// Returns a vector containing the distortion parameters of the model.
    /// For the EUCM model, these are `[alpha, beta]`.
    fn get_distortion(&self) -> Vec<f64> {
        self.model.get_distortion()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{Matrix2xX, Matrix3xX};

    /// Helper function to create a sample EUCM model for testing.
    fn get_sample_eucm_model() -> EucmModel {
        EucmModel {
            intrinsics: crate::camera::Intrinsics {
                fx: 500.0,
                fy: 500.0,
                cx: 320.0,
                cy: 240.0,
            },
            resolution: crate::camera::Resolution {
                width: 640,
                height: 480,
            },
            alpha: 0.6,
            beta: 0.8,
        }
    }

    /// Helper function to generate sample 3D-2D point correspondences.
    fn generate_sample_points(
        model: &EucmModel,
        num_points: usize,
    ) -> (Matrix3xX<f64>, Matrix2xX<f64>) {
        let (points_2d, points_3d) =
            crate::geometry::sample_points(Some(model), num_points).unwrap();
        (points_3d, points_2d)
    }

    #[test]
    fn test_eucm_optimization_cost_creation() {
        let model = get_sample_eucm_model();
        let (points_3d, points_2d) = generate_sample_points(&model, 10);

        let optimization_cost = EucmOptimizationCost::new(model.clone(), points_3d, points_2d);

        assert_eq!(optimization_cost.model.intrinsics.fx, model.intrinsics.fx);
        assert_eq!(optimization_cost.model.intrinsics.fy, model.intrinsics.fy);
        assert_eq!(optimization_cost.model.intrinsics.cx, model.intrinsics.cx);
        assert_eq!(optimization_cost.model.intrinsics.cy, model.intrinsics.cy);
        assert_eq!(optimization_cost.model.alpha, model.alpha);
        assert_eq!(optimization_cost.model.beta, model.beta);

        // Test the getter
        assert_eq!(optimization_cost.get_model().alpha, model.alpha);
    }

    #[test]
    fn test_eucm_optimization_getters() {
        let model = get_sample_eucm_model();
        let (points_3d, points_2d) = generate_sample_points(&model, 10);

        let optimization_cost = EucmOptimizationCost::new(model.clone(), points_3d, points_2d);

        let intrinsics = optimization_cost.get_intrinsics();
        let resolution = optimization_cost.get_resolution();
        let distortion = optimization_cost.get_distortion();

        assert_eq!(intrinsics.fx, model.intrinsics.fx);
        assert_eq!(intrinsics.fy, model.intrinsics.fy);
        assert_eq!(intrinsics.cx, model.intrinsics.cx);
        assert_eq!(intrinsics.cy, model.intrinsics.cy);
        assert_eq!(resolution.width, model.resolution.width);
        assert_eq!(resolution.height, model.resolution.height);
        assert_eq!(distortion, vec![model.alpha, model.beta]);
    }

    #[test]
    fn test_eucm_optimization_linear_estimation() {
        let reference_model = get_sample_eucm_model();
        let (points_3d, points_2d) = generate_sample_points(&reference_model, 20);

        let mut optimization_cost =
            EucmOptimizationCost::new(reference_model.clone(), points_3d, points_2d);

        let result = optimization_cost.linear_estimation();
        assert!(result.is_ok());

        // Parameters should be within reasonable range
        assert!(optimization_cost.model.alpha > 0.0);
        assert!(optimization_cost.model.alpha <= 1.0);
        assert!(optimization_cost.model.beta > 0.0);
        assert!(optimization_cost.model.beta <= 1.0);

        // Alpha and beta should be within a reasonable range of the reference
        assert_relative_eq!(
            optimization_cost.model.alpha,
            reference_model.alpha,
            epsilon = 1.0
        );
        assert_relative_eq!(
            optimization_cost.model.beta,
            reference_model.beta,
            epsilon = 1.0
        );
    }
}
