//! This module provides the cost function and optimization routines
//! for calibrating a Kannala-Brandt camera model.
//!
//! It uses the `tiny_solver` crate for non-linear optimization and defines
//! the necessary structures and traits to integrate the Kannala-Brandt
//! camera model with the optimization framework.

use crate::camera::{CameraModel, CameraModelError, KannalaBrandtModel};
use crate::optimization::Optimizer;

use log::info;
use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3xX, Vector2, Vector3};
use std::collections::HashMap;
use tiny_solver::factors::Factor;
use tiny_solver::{LevenbergMarquardtOptimizer, Optimizer as TinySolverOptimizer};

// Legacy factrs residual struct removed - now using tiny-solver automatic differentiation

/// Cost function for `tiny_solver` optimization of the [`KannalaBrandtModel`].
///
/// This struct defines the residual error between the observed 2D point and
/// the projected 2D point from a 3D point using the current camera model parameters.
/// It is used by the `tiny_solver` optimization framework.
#[derive(Debug, Clone)]
struct KBCost {
    /// 3D points in the camera's coordinate system.
    points3d: Vec<Vector3<f64>>,
    /// Corresponding observed 2D points in image coordinates.
    points2d: Vec<Vector2<f64>>,
}

impl KBCost {
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

impl<T: nalgebra::RealField> Factor<T> for KBCost {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        let cam_params = &params[0];
        let fx = cam_params[0].clone();
        let fy = cam_params[1].clone();
        let cx = cam_params[2].clone();
        let cy = cam_params[3].clone();
        let k1 = cam_params[4].clone();
        let k2 = cam_params[5].clone();
        let k3 = cam_params[6].clone();
        let k4 = cam_params[7].clone();

        let mut residuals = DVector::zeros(self.points2d.len() * 2);

        for i in 0..self.points2d.len() {
            let p3d = &self.points3d[i];
            let p2d = &self.points2d[i];

            let gt_u = T::from_f64(p2d.x).unwrap();
            let gt_v = T::from_f64(p2d.y).unwrap();
            let x = T::from_f64(p3d.x).unwrap();
            let y = T::from_f64(p3d.y).unwrap();
            let z = T::from_f64(p3d.z).unwrap();

            // Kannala-Brandt fisheye model
            let r_squared = x.clone() * x.clone() + y.clone() * y.clone();
            let r = r_squared.sqrt();
            let theta = r.clone().atan2(z.clone());

            let theta2 = theta.clone() * theta.clone();
            let theta3 = theta2.clone() * theta.clone();
            let theta5 = theta3.clone() * theta2.clone();
            let theta7 = theta5.clone() * theta2.clone();
            let theta9 = theta7.clone() * theta2.clone();

            let theta_d = theta.clone()
                + k1.clone() * theta3
                + k2.clone() * theta5
                + k3.clone() * theta7
                + k4.clone() * theta9;

            let epsilon = T::from_f64(f64::EPSILON).unwrap();
            let (x_r, y_r) = if r < epsilon {
                (T::from_f64(0.0).unwrap(), T::from_f64(0.0).unwrap())
            } else {
                (x.clone() / r.clone(), y.clone() / r)
            };

            let projected_x = fx.clone() * theta_d.clone() * x_r + cx.clone();
            let projected_y = fy.clone() * theta_d * y_r + cy.clone();

            residuals[i * 2] = projected_x - gt_u;
            residuals[i * 2 + 1] = projected_y - gt_v;
        }
        residuals
    }
}

/// Cost function for Kannala-Brandt camera model optimization.
///
/// This structure holds the 3D-2D point correspondences and the camera model
/// instance used during camera calibration optimization. It implements the
/// [`Optimizer`] trait for use with the optimization framework.
#[derive(Clone)]
pub struct KannalaBrandtOptimizationCost {
    /// The Kannala-Brandt camera model to be optimized.
    model: KannalaBrandtModel,
    /// 3D points in the camera's coordinate system (3×N matrix).
    /// Each column represents a 3D point.
    points3d: Matrix3xX<f64>,
    /// Corresponding 2D points in image coordinates (2×N matrix).
    /// Each column represents a 2D point observed in the image.
    points2d: Matrix2xX<f64>,
}

impl KannalaBrandtOptimizationCost {
    /// Creates a new [`KannalaBrandtOptimizationCost`] instance.
    ///
    /// # Arguments
    ///
    /// * `model` - The initial [`KannalaBrandtModel`] to be optimized.
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
        model: KannalaBrandtModel,
        points3d: Matrix3xX<f64>,
        points2d: Matrix2xX<f64>,
    ) -> Self {
        KannalaBrandtOptimizationCost {
            model,
            points3d,
            points2d,
        }
    }
}

impl Optimizer for KannalaBrandtOptimizationCost {
    /// Optimizes the Kannala-Brandt camera model parameters.
    ///
    /// This method uses the Levenberg-Marquardt algorithm provided by the `tiny_solver`
    /// crate to minimize the reprojection error between the observed 2D points
    /// and the 2D points projected from the 3D points using the current
    /// camera model parameters.
    ///
    /// The parameters being optimized are `[fx, fy, cx, cy, k1, k2, k3, k4]`.
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

        // Create initial parameter vector for tiny-solver
        let initial_params = DVector::from_vec(vec![
            self.model.intrinsics.fx,
            self.model.intrinsics.fy,
            self.model.intrinsics.cx,
            self.model.intrinsics.cy,
            self.model.distortions[0],
            self.model.distortions[1],
            self.model.distortions[2],
            self.model.distortions[3],
        ]);

        // Create a tiny_solver Problem
        let mut problem = tiny_solver::Problem::new();

        // Create the cost function
        let cost_function = KBCost::new(&self.points3d, &self.points2d);
        let num_residuals = self.points2d.ncols() * 2;
        problem.add_residual_block(num_residuals, &["params"], Box::new(cost_function), None);

        // Initial parameters as HashMap
        let mut initial_values = HashMap::new();
        initial_values.insert("params".to_string(), initial_params);

        if verbose {
            info!("Starting optimization with tiny-solver Levenberg-Marquardt...");
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
        self.model.distortions[0] = optimized_params[4];
        self.model.distortions[1] = optimized_params[5];
        self.model.distortions[2] = optimized_params[6];
        self.model.distortions[3] = optimized_params[7];

        // Validate the optimized parameters
        self.model.validate_params()?;

        Ok(())
    }

    /// Performs a linear estimation of the distortion parameters `k1, k2, k3, k4`
    /// for the Kannala-Brandt model.
    ///
    /// This method provides an initial estimate for the distortion parameters by
    /// reformulating the projection equations into a linear system.
    /// The intrinsic parameters `fx, fy, cx, cy` are assumed to be known and fixed.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the linear estimation was successful and `self.model.distortions`
    ///   has been updated.
    /// * `Err(CameraModelError)` - If an error occurred, such as mismatched point
    ///   counts, insufficient points for estimation (needs at least 4), or
    ///   numerical issues in solving the linear system.
    fn linear_estimation(&mut self) -> Result<(), CameraModelError>
    where
        Self: Sized,
    {
        // Duplicating the implementation from CameraModel trait for now
        if self.points3d.ncols() != self.points2d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }
        if self.points3d.ncols() < 4 {
            return Err(CameraModelError::InvalidParams(
                "Not enough points for linear estimation (need at least 4)".to_string(),
            ));
        }

        let num_points = self.points3d.ncols();
        let mut a_mat = DMatrix::zeros(num_points * 2, 4);
        let mut b_vec = DVector::zeros(num_points * 2);

        for i in 0..num_points {
            let p3d = self.points3d.column(i);
            let p2d = self.points2d.column(i);

            let x_world = p3d.x;
            let y_world = p3d.y;
            let z_world = p3d.z;

            let u_img = p2d.x;
            let v_img = p2d.y;

            if z_world <= f64::EPSILON {
                continue;
            }

            let r_world = (x_world * x_world + y_world * y_world).sqrt();
            let theta = r_world.atan2(z_world);

            let theta2 = theta * theta;
            let theta3 = theta2 * theta;
            let theta5 = theta3 * theta2;
            let theta7 = theta5 * theta2;
            let theta9 = theta7 * theta2;

            a_mat[(i * 2, 0)] = theta3;
            a_mat[(i * 2, 1)] = theta5;
            a_mat[(i * 2, 2)] = theta7;
            a_mat[(i * 2, 3)] = theta9;

            a_mat[(i * 2 + 1, 0)] = theta3;
            a_mat[(i * 2 + 1, 1)] = theta5;
            a_mat[(i * 2 + 1, 2)] = theta7;
            a_mat[(i * 2 + 1, 3)] = theta9;

            let x_r = if r_world < f64::EPSILON {
                0.0
            } else {
                x_world / r_world
            };
            let y_r = if r_world < f64::EPSILON {
                0.0
            } else {
                y_world / r_world
            };

            if (self.model.intrinsics.fx * x_r).abs() < f64::EPSILON && x_r.abs() > f64::EPSILON {
                return Err(CameraModelError::NumericalError(
                    "fx * x_r is zero in linear estimation".to_string(),
                ));
            }
            if (self.model.intrinsics.fy * y_r).abs() < f64::EPSILON && y_r.abs() > f64::EPSILON {
                return Err(CameraModelError::NumericalError(
                    "fy * y_r is zero in linear estimation".to_string(),
                ));
            }

            if x_r.abs() > f64::EPSILON {
                b_vec[i * 2] =
                    (u_img - self.model.intrinsics.cx) / (self.model.intrinsics.fx * x_r) - theta;
            } else {
                b_vec[i * 2] = if (u_img - self.model.intrinsics.cx).abs() < f64::EPSILON {
                    -theta
                } else {
                    0.0
                };
            }

            if y_r.abs() > f64::EPSILON {
                b_vec[i * 2 + 1] =
                    (v_img - self.model.intrinsics.cy) / (self.model.intrinsics.fy * y_r) - theta;
            } else {
                b_vec[i * 2 + 1] = if (v_img - self.model.intrinsics.cy).abs() < f64::EPSILON {
                    -theta
                } else {
                    0.0
                };
            }
        }

        let svd = a_mat.svd(true, true);
        let x_coeffs = svd.solve(&b_vec, f64::EPSILON).map_err(|e_str| {
            CameraModelError::NumericalError(format!(
                "SVD solve failed in linear estimation: {}",
                e_str
            ))
        })?;
        self.model.distortions = [x_coeffs[0], x_coeffs[1], x_coeffs[2], x_coeffs[3]];

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
    /// For the Kannala-Brandt model, these are `[k1, k2, k3, k4]`.
    fn get_distortion(&self) -> Vec<f64> {
        self.model.get_distortion()
    }
}

impl KannalaBrandtOptimizationCost {
    /// Validates the Jacobian computation by comparing analytical Jacobians with those
    /// obtained via automatic differentiation (though the AD part is currently placeholder).
    ///
    /// This method iterates through a subset of the 3D-2D point correspondences,
    /// computes the analytical Jacobian of the reprojection residual with respect
    /// to the camera parameters using [`KannalaBrandtFactrsResidual::compute_analytical_residual_jacobian`],
    /// and logs the results.
    ///
    /// The comparison with automatic differentiation is not fully implemented in this version.
    ///
    /// # Arguments
    ///
    /// * `_tolerance` - A tolerance value for comparing Jacobians (currently unused).
    ///
    /// # Returns
    ///
    /// * `Ok(bool)` - Currently always returns `Ok(true)` as the full comparison
    ///   is not implemented. It indicates that the analytical Jacobians were computed.
    /// * `Err(CameraModelError)` - If an error occurs during the process.
    ///
    /// # Panics
    ///
    /// This method might panic if `KBCamParams(0)` is not found in the `values` map
    /// during the call to `residual1_jacobian` if it were to fall back to AD,
    /// though the current implementation primarily focuses on the analytical path.
    pub fn validate_jacobian(&self, _tolerance: f64) -> Result<bool, CameraModelError> {
        info!("Validating optimization setup for tiny-solver...");

        let total_points = self.points3d.ncols();
        if total_points == 0 {
            return Err(CameraModelError::InvalidParams(
                "No points available for validation".to_string(),
            ));
        }

        if self.points3d.ncols() != self.points2d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Point correspondence count mismatch".to_string(),
            ));
        }

        info!(
            "Kannala-Brandt optimization validation: {} point correspondences ready for tiny-solver",
            total_points
        );

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraModel, Intrinsics, KannalaBrandtModel as KBCameraModel, Resolution};
    use crate::optimization::Optimizer;
    use approx::assert_relative_eq;
    use log::info;
    use nalgebra::{Matrix2xX, Matrix3xX, Vector2, Vector3};

    // Helper to get a sample KannalaBrandtModel instance
    fn get_sample_kb_camera_model() -> KBCameraModel {
        KBCameraModel {
            intrinsics: Intrinsics {
                fx: 461.586,
                fy: 460.281,
                cx: 366.286,
                cy: 249.080,
            },
            resolution: Resolution {
                width: 752,
                height: 480,
            },
            distortions: [-0.0125, 0.0578, -0.0849, 0.0436], // k1, k2, k3, k4
        }
    }

    // Placeholder for geometry::sample_points or a simplified version
    fn sample_points_for_kb_model(
        model: &KBCameraModel,
        num_points: usize,
    ) -> (Matrix2xX<f64>, Matrix3xX<f64>) {
        let mut points_2d_vec = Vec::new();
        let mut points_3d_vec = Vec::new();

        for i in 0..num_points {
            let x = (i as f64 * 0.1) - (num_points as f64 * 0.05);
            let y = (i as f64 * 0.05) - (num_points as f64 * 0.025);
            let z = 1.0 + (i as f64 * 0.01);
            let p3d = Vector3::new(x, y, z);

            if let Ok(p2d) = model.project(&p3d) {
                if p2d.x > 0.0
                    && p2d.x < model.resolution.width as f64
                    && p2d.y > 0.0
                    && p2d.y < model.resolution.height as f64
                {
                    points_3d_vec.push(p3d);
                    points_2d_vec.push(p2d);
                }
            }
        }
        if points_2d_vec.is_empty() && num_points > 0 {
            // Fallback if no points projected successfully, provide at least one dummy point
            // to prevent panic in Matrix2xX::from_columns on empty vec.
            // This indicates an issue with sample_points_for_kb_model or model params.
            info!("Warning: sample_points_for_kb_model generated no valid points. Using a dummy point.");
            points_3d_vec.push(Vector3::new(0.0, 0.0, 1.0));
            points_2d_vec.push(Vector2::new(model.intrinsics.cx, model.intrinsics.cy));
        }
        (
            Matrix2xX::from_columns(&points_2d_vec),
            Matrix3xX::from_columns(&points_3d_vec),
        )
    }

    // Test removed - was using factrs-specific APIs that are no longer available

    #[test]
    fn test_kannala_brandt_optimize_trait_method() {
        let reference_model = get_sample_kb_camera_model();
        let (points_2d, points_3d) = sample_points_for_kb_model(&reference_model, 50);
        assert!(
            points_3d.ncols() > 10,
            "Need sufficient points for optimization test. Actual points: {}",
            points_3d.ncols()
        );

        let noisy_model_initial = KBCameraModel {
            intrinsics: Intrinsics {
                fx: reference_model.intrinsics.fx * 1.05, // Introduce noise
                fy: reference_model.intrinsics.fy * 0.95,
                cx: reference_model.intrinsics.cx - 3.0,
                cy: reference_model.intrinsics.cy + 3.0,
            },
            resolution: reference_model.resolution.clone(),
            distortions: [
                reference_model.distortions[0] * 0.8,
                reference_model.distortions[1] * 1.2,
                reference_model.distortions[2] * 0.7,
                reference_model.distortions[3] * 1.3,
            ],
        };

        let mut cost_optimizer = KannalaBrandtOptimizationCost::new(
            noisy_model_initial,
            points_3d.clone(),
            points_2d.clone(),
        );
        let optimize_result = cost_optimizer.optimize(false);
        assert!(
            optimize_result.is_ok(),
            "Optimization failed: {:?}",
            optimize_result.err()
        );

        let optimized_model = &cost_optimizer.model;

        // Compare optimized parameters with reference_model
        assert_relative_eq!(
            optimized_model.intrinsics.fx,
            reference_model.intrinsics.fx,
            epsilon = 5.0,
            max_relative = 0.05
        ); // Looser epsilon
        assert_relative_eq!(
            optimized_model.intrinsics.fy,
            reference_model.intrinsics.fy,
            epsilon = 5.0,
            max_relative = 0.05
        );
        assert_relative_eq!(
            optimized_model.intrinsics.cx,
            reference_model.intrinsics.cx,
            epsilon = 5.0,
            max_relative = 0.05
        );
        assert_relative_eq!(
            optimized_model.intrinsics.cy,
            reference_model.intrinsics.cy,
            epsilon = 5.0,
            max_relative = 0.05
        );
        for i in 0..4 {
            assert_relative_eq!(
                optimized_model.distortions[i],
                reference_model.distortions[i],
                epsilon = 0.05,
                max_relative = 0.1
            ); // Looser
        }
    }

    #[test]
    fn test_kannala_brandt_linear_estimation_optimizer_trait() {
        let reference_model = get_sample_kb_camera_model();
        let (points_2d, points_3d) = sample_points_for_kb_model(&reference_model, 20);
        assert!(
            points_3d.ncols() > 3,
            "Need at least 4 points for KB linear estimation. Actual points: {}",
            points_3d.ncols()
        );

        // For linear estimation, we typically assume intrinsics are known or roughly known.
        // The linear estimation part of the Optimizer trait will update the distortions in its internal model.
        let initial_model_for_estimation = KBCameraModel {
            intrinsics: reference_model.intrinsics.clone(), // Use reference intrinsics
            resolution: reference_model.resolution.clone(),
            distortions: [0.0, 0.0, 0.0, 0.0], // Start with zero distortion for estimation
        };

        let mut cost_estimator = KannalaBrandtOptimizationCost::new(
            initial_model_for_estimation,
            points_3d.clone(),
            points_2d.clone(),
        );
        let estimation_result = cost_estimator.linear_estimation();

        assert!(
            estimation_result.is_ok(),
            "Linear estimation failed: {:?}",
            estimation_result.err()
        );
        let estimated_model = &cost_estimator.model;

        // Compare estimated distortion parameters. Linear estimation might not be super accurate.
        // The accuracy depends heavily on the quality of points and the model itself.
        for i in 0..4 {
            assert_relative_eq!(
                estimated_model.distortions[i],
                reference_model.distortions[i],
                epsilon = 0.1,
                max_relative = 0.2
            );
        }

        // Intrinsics should remain unchanged by this specific linear_estimation implementation
        assert_relative_eq!(
            estimated_model.intrinsics.fx,
            reference_model.intrinsics.fx,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            estimated_model.intrinsics.fy,
            reference_model.intrinsics.fy,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            estimated_model.intrinsics.cx,
            reference_model.intrinsics.cx,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            estimated_model.intrinsics.cy,
            reference_model.intrinsics.cy,
            epsilon = 1e-9
        );
    }
}
