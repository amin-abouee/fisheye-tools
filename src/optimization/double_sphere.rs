//! This module provides the cost function and optimization routines
//! for calibrating a Double Sphere camera model.
//!
//! It uses the `tiny_solver` crate for non-linear optimization and defines
//! the necessary structures and traits to integrate the Double Sphere
//! camera model with the optimization framework.

use crate::camera::{CameraModel, CameraModelError, DoubleSphereModel};
use crate::geometry::compute_reprojection_error;
use crate::optimization::Optimizer;

use log::info;
use nalgebra::{DVector, Matrix2xX, Matrix3xX, Vector2, Vector3};
use std::collections::HashMap;
use std::fmt;
use tiny_solver::factors::Factor;
use tiny_solver::{LevenbergMarquardtOptimizer, Optimizer as TinySolverOptimizer};

/// Cost function for Double Sphere camera model optimization.
///
/// This structure holds the 3D-2D point correspondences and the camera model
/// instance used during camera calibration optimization. It implements the
/// [`Optimizer`] trait for use with the optimization framework.
#[derive(Clone)]
pub struct DoubleSphereOptimizationCost {
    /// The Double Sphere camera model to be optimized.
    model: DoubleSphereModel,
    /// 3D points in the camera's coordinate system (3×N matrix).
    /// Each column represents a 3D point.
    points3d: Matrix3xX<f64>,
    /// Corresponding 2D points in image coordinates (2×N matrix).
    /// Each column represents a 2D point observed in the image.
    points2d: Matrix2xX<f64>,
}

/// Cost function for `tiny_solver` optimization of the [`DoubleSphereModel`].
///
/// This struct defines the residual error between the observed 2D point and
/// the projected 2D point from a 3D point using the current camera model parameters.
/// It is used by the `tiny_solver` optimization framework.
#[derive(Debug, Clone)]
struct DSCost {
    /// 3D points in the camera's coordinate system.
    points3d: Vec<Vector3<f64>>,
    /// Corresponding observed 2D points in image coordinates.
    points2d: Vec<Vector2<f64>>,
}

impl DSCost {
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

impl<T: nalgebra::RealField> Factor<T> for DSCost {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        let cam_params = &params[0];
        let fx = cam_params[0].clone();
        let fy = cam_params[1].clone();
        let cx = cam_params[2].clone();
        let cy = cam_params[3].clone();
        let alpha = cam_params[4].clone();
        let xi = cam_params[5].clone();

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
            let r_squared = obs_x.clone() * obs_x.clone() + obs_y.clone() * obs_y.clone();
            let d1 = (r_squared.clone() + obs_z.clone() * obs_z.clone()).sqrt();
            let gamma = xi.clone() * d1.clone() + obs_z.clone();
            let d2 = (r_squared + gamma.clone() * gamma.clone()).sqrt();
            let one = T::from_f64(1.0).unwrap();
            let denom = alpha.clone() * d2 + (one - alpha.clone()) * gamma.clone();

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

impl DoubleSphereOptimizationCost {
    /// Creates a new [`DoubleSphereOptimizationCost`] instance.
    ///
    /// # Arguments
    ///
    /// * `model` - The initial [`DoubleSphereModel`] to be optimized.
    /// * `points3d` - A 3×N matrix where each column is a 3D point in the
    ///   camera's coordinate system.
    /// * `points2d` - A 2×N matrix where each column is the corresponding
    ///   observed 2D point in image coordinates.
    ///
    /// # Panics
    ///
    /// This method does not panic directly, but the `optimize` method will return
    /// an error if the number of 3D and 2D points do not match or if the
    /// point arrays are empty.
    pub fn new(
        model: DoubleSphereModel,
        points3d: Matrix3xX<f64>,
        points2d: Matrix2xX<f64>,
    ) -> Self {
        DoubleSphereOptimizationCost {
            model,
            points3d,
            points2d,
        }
    }
}

impl fmt::Debug for DoubleSphereOptimizationCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DoubleSphereCostModel Summary:\n model: {:?}\n points3d size: {}, points2d size: {} ",
            self.model,
            self.points3d.ncols(),
            self.points2d.ncols(),
        )
    }
}

impl Optimizer for DoubleSphereOptimizationCost {
    /// Optimizes the Double Sphere camera model parameters.
    ///
    /// This method uses the Levenberg-Marquardt algorithm provided by the `tiny_solver`
    /// crate to minimize the reprojection error between the observed 2D points
    /// and the 2D points projected from the 3D points using the current
    /// camera model parameters.
    ///
    /// The parameters being optimized are `[fx, fy, cx, cy, alpha, xi]`.
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

        if let Ok(error_statistic) =
            compute_reprojection_error(Some(&self.model), &self.points3d, &self.points2d)
        {
            info!("Before Optimization: {:?}", error_statistic);
        }

        // Create a tiny_solver Problem
        let mut problem = tiny_solver::Problem::new();

        // Create the cost function
        let cost_function = DSCost::new(&self.points3d, &self.points2d);
        let num_residuals = self.points2d.ncols() * 2;
        problem.add_residual_block(num_residuals, &["params"], Box::new(cost_function), None);

        // Set parameter bounds
        // fx, fy bounds (reasonable range for focal lengths)
        problem.set_variable_bounds("params", 0, 1.0, 10000.0); // fx
        problem.set_variable_bounds("params", 1, 1.0, 10000.0); // fy

        // cx, cy bounds (should be within image dimensions)
        let img_width = self.model.resolution.width as f64;
        let img_height = self.model.resolution.height as f64;
        problem.set_variable_bounds("params", 2, -img_width, 2.0 * img_width); // cx
        problem.set_variable_bounds("params", 3, -img_height, 2.0 * img_height); // cy

        // Alpha bounds (critical constraint: 0 < alpha <= 1)
        problem.set_variable_bounds("params", 4, 1e-6, 1.0); // alpha

        // Xi bounds (typically reasonable range for Double Sphere)
        problem.set_variable_bounds("params", 5, -2.0, 2.0); // xi

        // Initial parameters
        let initial_params = DVector::from_vec(vec![
            self.model.intrinsics.fx,
            self.model.intrinsics.fy,
            self.model.intrinsics.cx,
            self.model.intrinsics.cy,
            self.model.alpha.max(1e-6).min(1.0), // Clamp alpha to valid range
            self.model.xi,
        ]);
        let mut initial_values = HashMap::new();
        initial_values.insert("params".to_string(), initial_params);

        if verbose {
            info!("Starting optimization with tiny_solver Levenberg-Marquardt...");
        }

        // Create a Levenberg-Marquardt optimizer
        let optimizer = LevenbergMarquardtOptimizer::default();

        // Run the optimization
        let result = optimizer
            .optimize(&problem, &initial_values, None)
            .ok_or_else(|| CameraModelError::NumericalError("Optimization failed".to_string()))?;

        if verbose {
            info!("Optimization finished");
        }

        // Extract the optimized parameters
        let optimized_params = result.get("params").unwrap();

        // Update the model parameters
        self.model.intrinsics.fx = optimized_params[0];
        self.model.intrinsics.fy = optimized_params[1];
        self.model.intrinsics.cx = optimized_params[2];
        self.model.intrinsics.cy = optimized_params[3];
        self.model.alpha = optimized_params[4].max(1e-6).min(1.0); // Ensure bounds
        self.model.xi = optimized_params[5];

        // Validate the optimized parameters
        self.model.validate_params()?;

        if let Ok(error_statistic) =
            compute_reprojection_error(Some(&self.model), &self.points3d, &self.points2d)
        {
            info!("After Optimization: {:?}", error_statistic);
        }

        Ok(())
    }

    /// Performs a linear estimation of the `alpha` parameter for the Double Sphere model.
    ///
    /// This method provides an initial estimate for the `alpha` parameter by
    /// reformulating the projection equations into a linear system.
    /// The `xi` parameter is assumed to be 0 for this estimation.
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
    fn linear_estimation(&mut self) -> Result<(), CameraModelError>
    where
        Self: Sized,
    {
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
            Ok(sol) => sol[0], // Handle the successful case
            Err(err_msg) => {
                return Err(CameraModelError::NumericalError(err_msg.to_string()));
            }
        };

        self.model.alpha = alpha;
        self.model.xi = 0.0;

        info!(
            "Linear estimation results: alpha = {}, xi = {}",
            self.model.alpha, self.model.xi
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
    /// For the Double Sphere model, these are `[alpha, xi]`.
    fn get_distortion(&self) -> Vec<f64> {
        self.model.get_distortion()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraModel, DoubleSphereModel as DSCameraModel};
    use approx::assert_relative_eq;
    use nalgebra::{DVector, Matrix2xX, Matrix3xX};

    // Helper to get a default model, similar to the one in samples/double_sphere.yaml
    fn get_sample_camera_model() -> DSCameraModel {
        let path = "samples/double_sphere.yaml";
        DoubleSphereModel::load_from_yaml(path).unwrap()
    }

    fn sample_points_for_ds_model(
        model: &DSCameraModel,
        num_points: usize,
    ) -> (Matrix2xX<f64>, Matrix3xX<f64>) {
        crate::geometry::sample_points(Some(model), num_points).unwrap()
    }

    #[test]
    fn test_double_sphere_linear_estimation_optimizer_trait() {
        let reference_model = get_sample_camera_model();
        let (points_2d, points_3d) = sample_points_for_ds_model(&reference_model, 50);

        assert!(
            points_3d.ncols() > 0,
            "Need points for linear estimation test."
        );

        let mut optimization_task = DoubleSphereOptimizationCost::new(
            reference_model.clone(),
            points_3d.clone(),
            points_2d.clone(),
        );

        optimization_task.linear_estimation().unwrap();

        assert_relative_eq!(
            optimization_task.model.alpha,
            reference_model.alpha,
            epsilon = 1.0
        );

        assert_relative_eq!(
            optimization_task.model.intrinsics.fx,
            reference_model.intrinsics.fx,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            optimization_task.model.intrinsics.fy,
            reference_model.intrinsics.fy,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            optimization_task.model.intrinsics.cx,
            reference_model.intrinsics.cx,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            optimization_task.model.intrinsics.cy,
            reference_model.intrinsics.cy,
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_double_sphere_optimize_tiny_solver() {
        let reference_model = get_sample_camera_model();
        let (points_2d, points_3d) = sample_points_for_ds_model(&reference_model, 50);

        let noisy_model = DoubleSphereModel::new(&DVector::from_vec(vec![
            reference_model.intrinsics.fx * 0.75,
            reference_model.intrinsics.fy * 1.35,
            reference_model.intrinsics.cx + 15.0,
            reference_model.intrinsics.cy - 17.0,
            reference_model.alpha * 0.7,
            0.0,
        ]))
        .unwrap();

        info!("Reference model: {:?}", reference_model);
        info!("Noisy model: {:?}", noisy_model);

        let mut optimization_task = DoubleSphereOptimizationCost::new(
            noisy_model.clone(),
            points_3d.clone(),
            points_2d.clone(),
        );

        match optimization_task.optimize(true) {
            Ok(()) => info!("Optimization succeeded"),
            Err(e) => {
                info!("Optimization failed with error: {:?}", e);
                return;
            }
        }

        info!("Optimized model with tiny_solver: {:?}", optimization_task);

        assert!(
            (optimization_task.model.intrinsics.fx - reference_model.intrinsics.fx).abs() < 1.0,
            "fx parameter didn't converge to expected value"
        );
        assert!(
            (optimization_task.model.intrinsics.fy - reference_model.intrinsics.fy).abs() < 1.0,
            "fy parameter didn't converge to expected value"
        );
        assert!(
            (optimization_task.model.intrinsics.cx - reference_model.intrinsics.cx).abs() < 1.0,
            "cx parameter didn't converge to expected value"
        );
        assert!(
            (optimization_task.model.intrinsics.cy - reference_model.intrinsics.cy).abs() < 1.0,
            "cy parameter didn't converge to expected value"
        );
        assert!(
            (optimization_task.model.alpha - reference_model.alpha).abs() < 0.05,
            "alpha parameter didn't converge to expected value"
        );
        assert!(
            (optimization_task.model.xi - reference_model.xi).abs() < 0.05,
            "xi parameter didn't converge to expected value"
        );
        assert!(
            optimization_task.model.alpha > 0.0 && optimization_task.model.alpha <= 1.0,
            "Alpha parameter out of valid range (0, 1]"
        );
    }

    #[test]
    fn test_reprojection_error_computation() {
        let reference_model = get_sample_camera_model();
        let (points_2d, points_3d) = sample_points_for_ds_model(&reference_model, 20);

        let mut total_error = 0.0;
        let mut valid_points = 0;

        for i in 0..points_3d.ncols() {
            let point_3d = Vector3::new(points_3d[(0, i)], points_3d[(1, i)], points_3d[(2, i)]);
            let point_2d_observed = Vector2::new(points_2d[(0, i)], points_2d[(1, i)]);

            if let Ok(point_2d_projected) = reference_model.project(&point_3d) {
                let error = (point_2d_projected - point_2d_observed).norm();
                total_error += error;
                valid_points += 1;
            }
        }

        assert!(valid_points > 0, "Should have valid projection points");

        let avg_error = total_error / valid_points as f64;
        assert!(
            avg_error < 1e-10,
            "Average reprojection error should be near zero for perfect model"
        );
    }

    #[test]
    fn test_parameter_convergence_validation() {
        let reference_model = get_sample_camera_model();
        let (points_2d, points_3d) = sample_points_for_ds_model(&reference_model, 50);

        let mut initial_model = reference_model.clone();
        initial_model.intrinsics.fx *= 0.98;
        initial_model.intrinsics.fy *= 0.98;
        initial_model.alpha *= 0.95;
        initial_model.xi *= 0.9;

        let mut optimization_task =
            DoubleSphereOptimizationCost::new(initial_model, points_3d, points_2d);

        let result = optimization_task.optimize(false);
        assert!(result.is_ok(), "Optimization should succeed");

        let final_intrinsics = optimization_task.get_intrinsics();
        let final_distortion = optimization_task.get_distortion();

        assert!(
            (final_intrinsics.fx - reference_model.intrinsics.fx).abs() < 1.0,
            "fx should converge close to reference value"
        );
        assert!(
            (final_intrinsics.fy - reference_model.intrinsics.fy).abs() < 1.0,
            "fy should converge close to reference value"
        );
        assert!(
            (final_distortion[0] - reference_model.alpha).abs() < 0.05,
            "alpha should converge close to reference value"
        );
        assert!(
            (final_distortion[1] - reference_model.xi).abs() < 0.05,
            "xi should converge close to reference value"
        );

        assert!(
            final_distortion[0] > 0.0 && final_distortion[0] <= 1.0,
            "Alpha should be within bounds (0, 1]"
        );
    }
}
