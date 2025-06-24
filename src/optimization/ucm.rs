//! This module provides the cost function and optimization routines
//! for calibrating a Unified Camera Model (UCM).
//!
//! It uses the `tiny_solver` crate for non-linear optimization and defines
//! the necessary structures and traits to integrate the UCM
//! camera model with the optimization framework.

use crate::camera::{CameraModel, CameraModelError, UcmModel};
use crate::optimization::Optimizer;

use log::info;
use nalgebra::{DVector, Matrix2xX, Matrix3xX, Vector2, Vector3};
use std::collections::HashMap;
use std::fmt;
use tiny_solver::factors::Factor;
use tiny_solver::{LevenbergMarquardtOptimizer, Optimizer as TinySolverOptimizer};

/// Cost function for UCM camera model optimization.
///
/// This structure holds the 3D-2D point correspondences and the camera model
/// instance used during camera calibration optimization. It implements the
/// [`Optimizer`] trait for use with the optimization framework.
#[derive(Clone)]
pub struct UcmOptimizationCost {
    /// The UCM camera model to be optimized.
    model: UcmModel,
    /// 3D points in the camera's coordinate system (3×N matrix).
    /// Each column represents a 3D point.
    points3d: Matrix3xX<f64>,
    /// Corresponding 2D points in image coordinates (2×N matrix).
    /// Each column represents a 2D point observed in the image.
    points2d: Matrix2xX<f64>,
}

/// Cost function for `tiny_solver` optimization of the [`UcmModel`].
///
/// This struct defines the residual error between the observed 2D point and
/// the projected 2D point from a 3D point using the current camera model parameters.
/// It is used by the `tiny_solver` optimization framework.
#[derive(Debug, Clone)]
struct UCMCost {
    /// 3D points in the camera's coordinate system.
    points3d: Vec<Vector3<f64>>,
    /// Corresponding observed 2D points in image coordinates.
    points2d: Vec<Vector2<f64>>,
}

impl UCMCost {
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

impl<T: nalgebra::RealField> Factor<T> for UCMCost {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        let cam_params = &params[0];
        let fx = cam_params[0].clone();
        let fy = cam_params[1].clone();
        let cx = cam_params[2].clone();
        let cy = cam_params[3].clone();
        let alpha = cam_params[4].clone();

        let mut residuals = DVector::zeros(self.points2d.len() * 2);

        for i in 0..self.points2d.len() {
            let p3d = &self.points3d[i];
            let p2d = &self.points2d[i];

            let gt_u = T::from_f64(p2d.x).unwrap();
            let gt_v = T::from_f64(p2d.y).unwrap();
            let obs_x = T::from_f64(p3d.x).unwrap();
            let obs_y = T::from_f64(p3d.y).unwrap();
            let obs_z = T::from_f64(p3d.z).unwrap();

            let u_cx = gt_u - cx.clone();
            let v_cy = gt_v - cy.clone();
            let d = (obs_x.clone() * obs_x.clone()
                + obs_y.clone() * obs_y.clone()
                + obs_z.clone() * obs_z.clone())
            .sqrt();
            let denom = alpha.clone() * d.clone()
                + (T::from_f64(1.0).unwrap() - alpha.clone()) * obs_z.clone();

            let precision = T::from_f64(1e-3).unwrap();
            if denom < precision {
                residuals[i * 2] = T::from_f64(1e6).unwrap();
                residuals[i * 2 + 1] = T::from_f64(1e6).unwrap();
            } else {
                residuals[i * 2] = fx.clone() * obs_x - u_cx * denom.clone();
                residuals[i * 2 + 1] = fy.clone() * obs_y - v_cy * denom;
            }
        }
        residuals
    }
}

impl UcmOptimizationCost {
    /// Creates a new optimization cost function for the UCM camera model.
    ///
    /// # Arguments
    ///
    /// * `model` - The initial UCM camera model to be optimized.
    /// * `points3d` - 3D points in the camera's coordinate system (3×N matrix).
    /// * `points2d` - Corresponding 2D points in image coordinates (2×N matrix).
    ///
    /// # Returns
    ///
    /// A new `UcmOptimizationCost` instance.
    pub fn new(model: UcmModel, points3d: Matrix3xX<f64>, points2d: Matrix2xX<f64>) -> Self {
        Self {
            model,
            points3d,
            points2d,
        }
    }
}

impl fmt::Debug for UcmOptimizationCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "UcmOptimizationCost Summary:\n model: {:?}\n points3d size: {}, points2d size: {} ",
            self.model,
            self.points3d.ncols(),
            self.points2d.ncols(),
        )
    }
}

impl Optimizer for UcmOptimizationCost {
    /// Optimizes the UCM camera model parameters.
    ///
    /// This method uses the Levenberg-Marquardt algorithm provided by the `tiny_solver`
    /// crate to minimize the reprojection error between the observed 2D points
    /// and the 2D points projected from the 3D points using the current
    /// camera model parameters.
    ///
    /// The parameters being optimized are `[fx, fy, cx, cy, alpha]`.
    /// Alpha parameter is constrained to be between 0 and 1.
    ///
    /// # Arguments
    ///
    /// * `verbose` - If `true`, prints optimization progress and results to the console.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the optimization was successful and the model parameters
    ///   have been updated.
    /// * `Err(CameraModelError)` - If an error occurred during optimization,
    ///   such as invalid input parameters or numerical issues.
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

        // Create a tiny_solver Problem
        let mut problem = tiny_solver::Problem::new();

        // Create the cost function
        let cost_function = UCMCost::new(&self.points3d, &self.points2d);
        let num_residuals = self.points2d.ncols() * 2;
        problem.add_residual_block(num_residuals, &["params"], Box::new(cost_function), None);

        // Set parameter bounds for alpha (critical constraint: 0 < alpha <= 1)
        problem.set_variable_bounds("params", 4, 1e-6, 1.0); // alpha

        // Initial parameters
        let initial_params = DVector::from_vec(vec![
            self.model.intrinsics.fx,
            self.model.intrinsics.fy,
            self.model.intrinsics.cx,
            self.model.intrinsics.cy,
            self.model.alpha.clamp(1e-6, 1.0), // Clamp alpha to valid range
        ]);
        let mut initial_values = HashMap::new();
        initial_values.insert("params".to_string(), initial_params);

        if verbose {
            info!("Starting UCM optimization with tiny_solver Levenberg-Marquardt...");
        }

        // Create a Levenberg-Marquardt optimizer
        let optimizer = LevenbergMarquardtOptimizer::default();

        // Run the optimization
        let result = optimizer
            .optimize(&problem, &initial_values, None)
            .ok_or_else(|| CameraModelError::NumericalError("Optimization failed".to_string()))?;

        if verbose {
            info!("UCM optimization finished");
        }

        // Extract the optimized parameters
        let optimized_params = result.get("params").unwrap();

        // Update the model parameters
        self.model.intrinsics.fx = optimized_params[0];
        self.model.intrinsics.fy = optimized_params[1];
        self.model.intrinsics.cx = optimized_params[2];
        self.model.intrinsics.cy = optimized_params[3];
        self.model.alpha = optimized_params[4].clamp(1e-6, 1.0); // Ensure bounds

        // Validate the optimized parameters
        self.model.validate_params()?;

        Ok(())
    }

    /// Performs a linear estimation of the `alpha` parameter for the UCM model.
    ///
    /// This method provides an initial estimate for the `alpha` parameter by
    /// reformulating the projection equations into a linear system.
    /// The intrinsic parameters `fx, fy, cx, cy` are assumed to be known and fixed.
    ///
    /// After estimation, `alpha` is clamped to the range `(0.0, 1.0]`.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the linear estimation was successful and `self.model.alpha`
    ///   has been updated.
    /// * `Err(CameraModelError)` - If an error occurred, such as mismatched point
    ///   counts or numerical issues in solving the linear system.
    fn linear_estimation(&mut self) -> Result<(), CameraModelError> {
        // Check if the number of 2D and 3D points match
        if self.points2d.ncols() != self.points3d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }

        // Set up the linear system to solve for alpha
        let num_points = self.points2d.ncols();
        let mut a = nalgebra::DMatrix::zeros(num_points * 2, 1);
        let mut b = nalgebra::DVector::zeros(num_points * 2);

        for i in 0..num_points {
            let x = self.points3d[(0, i)];
            let y = self.points3d[(1, i)];
            let z = self.points3d[(2, i)];
            let u = self.points2d[(0, i)];
            let v = self.points2d[(1, i)];

            let d = (x * x + y * y + z * z).sqrt();
            let u_cx = u - self.model.intrinsics.cx;
            let v_cy = v - self.model.intrinsics.cy;

            a[(i * 2, 0)] = u_cx * (d - z);
            a[(i * 2 + 1, 0)] = v_cy * (d - z);

            b[i * 2] = (self.model.intrinsics.fx * x) - (u_cx * z);
            b[i * 2 + 1] = (self.model.intrinsics.fy * y) - (v_cy * z);
        }

        // Solve the linear system using SVD
        let svd = a.svd(true, true);
        let alpha = match svd.solve(&b, 1e-10) {
            Ok(sol) => sol[0],
            Err(err_msg) => {
                return Err(CameraModelError::NumericalError(err_msg.to_string()));
            }
        };

        self.model.alpha = alpha;

        info!(
            "UCM linear estimation results: alpha = {}",
            self.model.alpha
        );

        // Clamp alpha to valid range if needed
        if self.model.alpha <= 0.0 {
            info!("Alpha {} is too small, clamping to 0.01", self.model.alpha);
            self.model.alpha = 0.01;
        } else if self.model.alpha > 1.0 {
            info!("Alpha {} is too large, clamping to 1.0", self.model.alpha);
            self.model.alpha = 1.0;
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
    /// For the UCM model, this is `[alpha]`.
    fn get_distortion(&self) -> Vec<f64> {
        self.model.get_distortion()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraModel, UcmModel as UCMCameraModel};
    use nalgebra::{Matrix2xX, Matrix3xX};

    fn get_sample_ucm_model() -> UCMCameraModel {
        let path = "samples/ucm.yaml";
        UcmModel::load_from_yaml(path).unwrap()
    }

    fn generate_sample_points(
        model: &UCMCameraModel,
        num_points: usize,
    ) -> (Matrix3xX<f64>, Matrix2xX<f64>) {
        let (points_2d, points_3d) =
            crate::geometry::sample_points(Some(model), num_points).unwrap();
        (points_3d, points_2d)
    }

    #[test]
    fn test_ucm_optimization_cost_creation() {
        let model = get_sample_ucm_model();
        let (points_3d, points_2d) = generate_sample_points(&model, 10);

        let optimization_cost = UcmOptimizationCost::new(model.clone(), points_3d, points_2d);

        assert_eq!(optimization_cost.model.intrinsics.fx, model.intrinsics.fx);
        assert_eq!(optimization_cost.model.intrinsics.fy, model.intrinsics.fy);
        assert_eq!(optimization_cost.model.intrinsics.cx, model.intrinsics.cx);
        assert_eq!(optimization_cost.model.intrinsics.cy, model.intrinsics.cy);
        assert_eq!(optimization_cost.model.alpha, model.alpha);
    }

    #[test]
    fn test_ucm_optimization_linear_estimation() {
        let model = get_sample_ucm_model();
        let (points_3d, points_2d) = generate_sample_points(&model, 100);

        let mut optimizer = UcmOptimizationCost::new(model, points_3d, points_2d);

        // This should not panic
        let result = optimizer.linear_estimation();
        assert!(result.is_ok());
    }

    #[test]
    fn test_ucm_optimization_getters() {
        let model = get_sample_ucm_model();
        let (points_3d, points_2d) = generate_sample_points(&model, 10);

        let optimization_cost = UcmOptimizationCost::new(model.clone(), points_3d, points_2d);

        let intrinsics = optimization_cost.get_intrinsics();
        let resolution = optimization_cost.get_resolution();
        let distortion = optimization_cost.get_distortion();

        assert_eq!(intrinsics.fx, model.intrinsics.fx);
        assert_eq!(intrinsics.fy, model.intrinsics.fy);
        assert_eq!(intrinsics.cx, model.intrinsics.cx);
        assert_eq!(intrinsics.cy, model.intrinsics.cy);
        assert_eq!(resolution.width, model.resolution.width);
        assert_eq!(resolution.height, model.resolution.height);
        assert_eq!(distortion, vec![model.alpha]);
    }
}
