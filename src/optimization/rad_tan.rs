//! This module provides the cost function and optimization routines
//! for calibrating a Radial-Tangential (RadTan) camera model.
//!
//! It uses the `tiny_solver` crate for non-linear optimization and defines
//! the necessary structures and traits to integrate the RadTan
//! camera model with the optimization framework.

use crate::camera::{CameraModel, CameraModelError, RadTanModel};
use crate::optimization::Optimizer;

use log::info;
use nalgebra::{DVector, Matrix2xX, Matrix3xX, Vector2, Vector3};
use std::collections::HashMap;
use std::fmt;
use tiny_solver::factors::Factor;
use tiny_solver::{LevenbergMarquardtOptimizer, Optimizer as TinySolverOptimizer};

/// Cost function for Radial-Tangential (RadTan) camera model optimization.
///
/// This structure holds the 3D-2D point correspondences and the camera model
/// instance used during camera calibration optimization. It implements the
/// [`Optimizer`] trait for use with the optimization framework.
#[derive(Clone)]
pub struct RadTanOptimizationCost {
    /// The RadTan camera model to be optimized.
    model: RadTanModel,
    /// 3D points in the camera's coordinate system (3×N matrix).
    /// Each column represents a 3D point.
    points3d: Matrix3xX<f64>,
    /// Corresponding 2D points in image coordinates (2×N matrix).
    /// Each column represents a 2D point observed in the image.
    points2d: Matrix2xX<f64>,
}

/// Cost function for `tiny_solver` optimization of the [`RadTanModel`].
///
/// This struct defines the residual error between the observed 2D point and
/// the projected 2D point from a 3D point using the current camera model parameters.
/// It is used by the `tiny_solver` optimization framework.
#[derive(Debug, Clone)]
struct RadTanCost {
    /// 3D points in the camera's coordinate system.
    points3d: Vec<Vector3<f64>>,
    /// Corresponding observed 2D points in image coordinates.
    points2d: Vec<Vector2<f64>>,
}

impl RadTanCost {
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

impl<T: nalgebra::RealField> Factor<T> for RadTanCost {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        let cam_params = &params[0];
        let fx = cam_params[0].clone();
        let fy = cam_params[1].clone();
        let cx = cam_params[2].clone();
        let cy = cam_params[3].clone();
        let k1 = cam_params[4].clone();
        let k2 = cam_params[5].clone();
        let p1 = cam_params[6].clone();
        let p2 = cam_params[7].clone();
        let k3 = cam_params[8].clone();

        let mut residuals = DVector::zeros(self.points2d.len() * 2);

        for i in 0..self.points2d.len() {
            let p3d = &self.points3d[i];
            let p2d = &self.points2d[i];

            // Convert to autodiff-compatible types
            let point3d_x = T::from_f64(p3d.x).unwrap();
            let point3d_y = T::from_f64(p3d.y).unwrap();
            let point3d_z = T::from_f64(p3d.z).unwrap();
            let gt_u = T::from_f64(p2d.x).unwrap();
            let gt_v = T::from_f64(p2d.y).unwrap();

            // Create RadTanModel-like computation in autodiff format
            // Project 3D point to normalized image coordinates
            let x_norm = point3d_x.clone() / point3d_z.clone();
            let y_norm = point3d_y.clone() / point3d_z.clone();

            // Apply radial and tangential distortion
            let r2 = x_norm.clone() * x_norm.clone() + y_norm.clone() * y_norm.clone();
            let r4 = r2.clone() * r2.clone();
            let r6 = r4.clone() * r2.clone();

            let radial_distortion = T::from_f64(1.0).unwrap()
                + k1.clone() * r2.clone()
                + k2.clone() * r4.clone()
                + k3.clone() * r6;

            let two = T::from_f64(2.0).unwrap();
            let tangential_x = two.clone() * p1.clone() * x_norm.clone() * y_norm.clone()
                + p2.clone() * (r2.clone() + two.clone() * x_norm.clone() * x_norm.clone());
            let tangential_y = p1.clone()
                * (r2.clone() + two.clone() * y_norm.clone() * y_norm.clone())
                + two * p2.clone() * x_norm.clone() * y_norm.clone();

            let x_distorted = x_norm.clone() * radial_distortion.clone() + tangential_x;
            let y_distorted = y_norm.clone() * radial_distortion + tangential_y;

            // Project to pixel coordinates
            let u_projected = fx.clone() * x_distorted + cx.clone();
            let v_projected = fy.clone() * y_distorted + cy.clone();

            // Compute residuals (projected - observed)
            residuals[i * 2] = u_projected - gt_u;
            residuals[i * 2 + 1] = v_projected - gt_v;
        }
        residuals
    }
}

impl RadTanOptimizationCost {
    /// Creates a new [`RadTanOptimizationCost`] instance.
    ///
    /// # Arguments
    ///
    /// * `model` - The initial [`RadTanModel`] to be optimized.
    /// * `points3d` - A 3×N matrix where each column is a 3D point in the
    ///   camera's coordinate system.
    /// * `points2d` - A 2×N matrix where each column is the corresponding
    ///   observed 2D point in image coordinates.
    ///
    /// # Panics
    ///
    /// Asserts that the number of 3D points matches the number of 2D points.
    /// The `optimize` method will also return an error if the point arrays are empty.
    pub fn new(model: RadTanModel, points3d: Matrix3xX<f64>, points2d: Matrix2xX<f64>) -> Self {
        assert_eq!(points3d.ncols(), points2d.ncols());
        RadTanOptimizationCost {
            model,
            points3d,
            points2d,
        }
    }
}

impl fmt::Debug for RadTanOptimizationCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RadTanOptimizationCost Summary:\n model: {:?}\n points3d size: {}, points2d size: {} ",
            self.model,
            self.points3d.ncols(),
            self.points2d.ncols(),
        )
    }
}

impl Optimizer for RadTanOptimizationCost {
    /// Optimizes the RadTan camera model parameters.
    ///
    /// This method uses the Levenberg-Marquardt algorithm provided by the `tiny_solver`
    /// crate to minimize the reprojection error between the observed 2D points
    /// and the 2D points projected from the 3D points using the current
    /// camera model parameters.
    ///
    /// The parameters being optimized are `[fx, fy, cx, cy, k1, k2, p1, p2, k3]`.
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
        let cost_function = RadTanCost::new(&self.points3d, &self.points2d);
        let num_residuals = self.points2d.ncols() * 2;
        problem.add_residual_block(num_residuals, &["params"], Box::new(cost_function), None);

        // Initial parameters
        let initial_params = DVector::from_vec(vec![
            self.model.intrinsics.fx,
            self.model.intrinsics.fy,
            self.model.intrinsics.cx,
            self.model.intrinsics.cy,
            self.model.distortions[0], // k1
            self.model.distortions[1], // k2
            self.model.distortions[2], // p1
            self.model.distortions[3], // p2
            self.model.distortions[4], // k3
        ]);
        let mut initial_values = HashMap::new();
        initial_values.insert("params".to_string(), initial_params);

        if verbose {
            info!("Starting RadTan optimization with tiny_solver Levenberg-Marquardt...");
        }

        // Create a Levenberg-Marquardt optimizer
        let optimizer = LevenbergMarquardtOptimizer::default();

        // Run the optimization
        let result = optimizer
            .optimize(&problem, &initial_values, None)
            .ok_or_else(|| CameraModelError::NumericalError("Optimization failed".to_string()))?;

        if verbose {
            info!("RadTan optimization finished");
        }

        // Extract the optimized parameters
        let optimized_params = result.get("params").unwrap();

        // Update the model parameters
        self.model.intrinsics.fx = optimized_params[0];
        self.model.intrinsics.fy = optimized_params[1];
        self.model.intrinsics.cx = optimized_params[2];
        self.model.intrinsics.cy = optimized_params[3];
        self.model.distortions[0] = optimized_params[4]; // k1
        self.model.distortions[1] = optimized_params[5]; // k2
        self.model.distortions[2] = optimized_params[6]; // p1
        self.model.distortions[3] = optimized_params[7]; // p2
        self.model.distortions[4] = optimized_params[8]; // k3

        // Validate the optimized parameters
        self.model.validate_params()?;

        Ok(())
    }

    /// Performs a linear estimation for initial parameter values.
    ///
    /// For RadTan model, this method estimates the coefficients using linear least squares.
    /// The intrinsic parameters (fx, fy, cx, cy) are assumed to be known and fixed.
    /// This method estimates the distortion coefficients [k1, k2, p1, p2, k3].
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the linear estimation was successful and the distortion coefficients
    ///   have been updated.
    /// * `Err(CameraModelError)` - If an error occurred, such as mismatched point
    ///   counts or numerical issues in solving the linear system.
    fn linear_estimation(&mut self) -> Result<(), CameraModelError>
    where
        Self: Sized,
    {
        if self.points2d.ncols() != self.points3d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }

        let num_points = self.points2d.ncols();
        if num_points < 3 {
            return Err(CameraModelError::InvalidParams(
                "Need at least 3 points for RadTan linear estimation".to_string(),
            ));
        }

        // Set up the linear system to estimate distortion coefficients
        let mut a = nalgebra::DMatrix::zeros(num_points * 2, 3); // Only estimate k1, k2, k3
        let mut b = nalgebra::DVector::zeros(num_points * 2);

        // Extract intrinsics
        let fx = self.model.intrinsics.fx;
        let fy = self.model.intrinsics.fy;
        let cx = self.model.intrinsics.cx;
        let cy = self.model.intrinsics.cy;

        for i in 0..num_points {
            let x = self.points3d[(0, i)];
            let y = self.points3d[(1, i)];
            let z = self.points3d[(2, i)];
            let u = self.points2d[(0, i)];
            let v = self.points2d[(1, i)];

            // Project to normalized coordinates
            let x_norm = x / z;
            let y_norm = y / z;
            let r2 = x_norm * x_norm + y_norm * y_norm;
            let r4 = r2 * r2;
            let r6 = r4 * r2;

            // Predicted undistorted pixel coordinates
            let u_undist = fx * x_norm + cx;
            let v_undist = fy * y_norm + cy;

            // Set up linear system for distortion coefficients
            a[(i * 2, 0)] = fx * x_norm * r2; // k1 term
            a[(i * 2, 1)] = fx * x_norm * r4; // k2 term
            a[(i * 2, 2)] = fx * x_norm * r6; // k3 term

            a[(i * 2 + 1, 0)] = fy * y_norm * r2; // k1 term
            a[(i * 2 + 1, 1)] = fy * y_norm * r4; // k2 term
            a[(i * 2 + 1, 2)] = fy * y_norm * r6; // k3 term

            b[i * 2] = u - u_undist;
            b[i * 2 + 1] = v - v_undist;
        }

        // Solve the linear system using SVD
        let svd = a.svd(true, true);
        let distortion_coeffs = match svd.solve(&b, 1e-10) {
            Ok(sol) => sol,
            Err(err_msg) => {
                return Err(CameraModelError::NumericalError(err_msg.to_string()));
            }
        };

        // Update distortion coefficients (keep p1, p2 as zero for linear estimation)
        self.model.distortions[0] = distortion_coeffs[0]; // k1
        self.model.distortions[1] = distortion_coeffs[1]; // k2
        self.model.distortions[2] = 0.0; // p1 (tangential)
        self.model.distortions[3] = 0.0; // p2 (tangential)
        self.model.distortions[4] = distortion_coeffs[2]; // k3

        info!(
            "RadTan linear estimation results: k1={}, k2={}, k3={}",
            self.model.distortions[0], self.model.distortions[1], self.model.distortions[4]
        );

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
    /// For the RadTan model, these are `[k1, k2, p1, p2, k3]`.
    fn get_distortion(&self) -> Vec<f64> {
        self.model.get_distortion()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::RadTanModel;

    /// Helper function to create a sample RadTan model for testing.
    fn get_sample_rt_camera_model() -> RadTanModel {
        RadTanModel {
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
            distortions: [0.1, -0.02, 0.001, -0.0005, 0.0001], // k1, k2, p1, p2, k3
        }
    }

    /// Helper function to generate sample 3D-2D point correspondences for RadTan.
    fn sample_points_for_rt_model(
        model: &RadTanModel,
        num_points: usize,
    ) -> (Matrix2xX<f64>, Matrix3xX<f64>) {
        crate::geometry::sample_points(Some(model), num_points).unwrap()
    }

    #[test]
    fn test_radtan_optimize_trait_method_call() {
        let model = get_sample_rt_camera_model();
        let (points_2d, points_3d) = sample_points_for_rt_model(&model, 50);

        let mut optimization_cost = RadTanOptimizationCost::new(model, points_3d, points_2d);

        // Test that the optimize method exists and can be called
        let result = optimization_cost.optimize(false);

        // Should succeed for valid input
        assert!(result.is_ok(), "RadTan optimization should succeed");

        // Verify that the intrinsics are reasonable after optimization
        let intrinsics = optimization_cost.get_intrinsics();
        assert!(intrinsics.fx > 0.0);
        assert!(intrinsics.fy > 0.0);

        // Verify that the distortion parameters are updated
        let distortion = optimization_cost.get_distortion();
        assert_eq!(distortion.len(), 5);
    }
}
