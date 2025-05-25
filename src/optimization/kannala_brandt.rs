//! Provides optimization capabilities for the Kannala-Brandt camera model.
//!
//! This module defines [`KannalaBrandtOptimizationCost`], a struct that encapsulates
//! the necessary data (camera model and 2D-3D point correspondences) and implements
//! various traits from the `argmin` optimization library. It is used to refine
//! the parameters of a [`KannalaBrandtModel`] by minimizing reprojection errors.
//!
//! This module implements the [`Optimizer`] trait from the parent `optimization`
//! module, offering both non-linear optimization (via `argmin` and Gauss-Newton)
//! and a linear estimation method for initializing parameters.
//!
//! # References
//!
//! The Kannala-Brandt model is based on the paper:
//! *   Kannala, J., & Brandt, S. S. (2006). A generic camera model and calibration
//!     method for conventional, wide-angle, and fish-eye lenses.
//!     *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *28*(8), 1335-1340.

use crate::camera::{CameraModel, CameraModelError, KannalaBrandtModel};
use crate::optimization::Optimizer;
use argmin::{
    core::{
        observers::ObserverMode, CostFunction, Error as ArgminError, Executor, Gradient, Hessian,
        Jacobian, Operator, State,
    },
    solver::gaussnewton::GaussNewton,
};
use argmin_observer_slog::SlogLogger;
use log::info; // For logging optimization progress and potential warnings
use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3xX};

/// Cost function for optimizing the parameters of a [`KannalaBrandtModel`].
///
/// This structure holds the 3D-2D point correspondences and the camera model
/// instance. It is designed to be used with the `argmin` optimization library,
/// implementing traits such as [`Operator`], [`Jacobian`], [`CostFunction`],
/// [`Gradient`], and [`Hessian`] to define the optimization problem.
/// The goal is to find camera parameters that minimize the reprojection error
/// between the observed 2D points and the 2D points projected from the 3D points
/// using the current camera model estimate.
///
/// # Fields
/// * `model`: [`KannalaBrandtModel`] - The camera model instance whose parameters are being optimized.
///   This model is updated during the optimization process.
/// * `points3d`: [`Matrix3xX<f64>`] - A matrix where each column represents a 3D point
///   in the camera's coordinate system. These are the reference points.
/// * `points2d`: [`Matrix2xX<f64>`] - A matrix where each column represents the observed 2D projection
///   (in pixel coordinates) of the corresponding 3D point in `points3d`.
///
/// # Examples
/// ```rust
/// use nalgebra::{Matrix2xX, Matrix3xX, Vector3, Vector2};
/// use vision_toolkit_rs::camera::{KannalaBrandtModel, Intrinsics, Resolution};
/// use vision_toolkit_rs::optimization::kannala_brandt::KannalaBrandtOptimizationCost;
///
/// // Assume 'initial_model' is a KannalaBrandtModel instance
/// # let initial_model = KannalaBrandtModel {
/// #     intrinsics: Intrinsics { fx: 460.0, fy: 460.0, cx: 320.0, cy: 240.0 },
/// #     resolution: Resolution { width: 640, height: 480 },
/// #     distortions: [-0.01, 0.05, -0.08, 0.04],
/// # };
/// // Assume 'world_points' and 'image_points' are matrices of corresponding points
/// # let world_points = Matrix3xX::from_columns(&[Vector3::new(0.0,0.0,1.0)]);
/// # let image_points = Matrix2xX::from_columns(&[Vector2::new(320.0,240.0)]);
///
/// let cost_function_context = KannalaBrandtOptimizationCost::new(
///     initial_model,
///     world_points,
///     image_points
/// );
/// // This 'cost_function_context' can then be used with an Optimizer.
/// // println!("{:?}", cost_function_context); // Debug print
/// ```
#[derive(Clone)]
pub struct KannalaBrandtOptimizationCost {
    /// The [`KannalaBrandtModel`] instance that is being optimized.
    /// Its parameters are updated by the optimization process.
    model: KannalaBrandtModel,
    /// A 3xN matrix where each column is a 3D point in the camera's coordinate system.
    points3d: Matrix3xX<f64>,
    /// A 2xN matrix where each column is the observed 2D projection (in pixels)
    /// of the corresponding 3D point in `points3d`.
    points2d: Matrix2xX<f64>,
}

impl KannalaBrandtOptimizationCost {
    /// Creates a new [`KannalaBrandtOptimizationCost`] instance for optimizing a [`KannalaBrandtModel`].
    ///
    /// This constructor initializes the cost function context with the camera model
    /// to be optimized and the sets of corresponding 3D world points and 2D image observations.
    ///
    /// # Arguments
    ///
    /// * `model`: [`KannalaBrandtModel`] - The initial state of the Kannala-Brandt camera model
    ///   whose parameters will be optimized.
    /// * `points3d`: [`Matrix3xX<f64>`] - A matrix of 3D points (3xN), where each column
    ///   is a point in the camera's coordinate system.
    /// * `points2d`: [`Matrix2xX<f64>`] - A matrix of 2D points (2xN), where each column
    ///   is the observed projection of the corresponding 3D point, in pixel coordinates.
    ///
    /// # Return Value
    ///
    /// Returns a new instance of `KannalaBrandtOptimizationCost`.
    ///
    /// # Panics
    ///
    /// Panics if the number of columns (points) in `points3d` and `points2d` do not match,
    /// as verified by `assert_eq!(points3d.ncols(), points2d.ncols());`.
    pub fn new(
        model: KannalaBrandtModel,
        points3d: Matrix3xX<f64>,
        points2d: Matrix2xX<f64>,
    ) -> Self {
        assert_eq!(points3d.ncols(), points2d.ncols());
        KannalaBrandtOptimizationCost {
            model,
            points3d,
            points2d,
        }
    }
}

/// Implements the [`Optimizer`] trait for [`KannalaBrandtOptimizationCost`].
impl Optimizer for KannalaBrandtOptimizationCost {
    /// Performs non-linear optimization of the [`KannalaBrandtModel`] parameters.
    ///
    /// This method uses the `argmin` library with a Gauss-Newton solver to refine
    /// the camera model's parameters (intrinsics `fx, fy, cx, cy` and the 4 distortion
    /// coefficients `k1, k2, k3, k4`). The optimization aims to minimize the sum of squared
    /// reprojection errors.
    ///
    /// # Algorithm
    ///
    /// 1.  Validates that the number of 3D and 2D points match and are non-zero.
    /// 2.  Clones the current `KannalaBrandtOptimizationCost` instance to serve as the
    ///     cost function for `argmin`.
    /// 3.  Extracts initial parameters (`fx, fy, cx, cy, k1, k2, k3, k4`) from `self.model`.
    /// 4.  Configures a `GaussNewton` solver.
    /// 5.  Sets up an `Executor` with the solver, cost function, initial parameters,
    ///     and a maximum number of iterations (100).
    /// 6.  Optionally enables detailed logging via `SlogLogger` if `verbose` is true.
    /// 7.  Runs the optimization.
    /// 8.  If successful, updates `self.model` with the optimized parameters.
    /// 9.  Validates the final optimized parameters using `self.model.validate_params()`.
    ///
    /// # Arguments
    ///
    /// * `verbose`: `bool` - If true, enables detailed iteration logging during optimization.
    ///   If false, logs only new best states.
    ///
    /// # Return Value
    ///
    /// Returns `Ok(())` if the optimization completes successfully and parameters are valid.
    ///
    /// # Errors
    ///
    /// *   [`CameraModelError::InvalidParams`]: If point counts mismatch or are zero.
    /// *   [`CameraModelError::NumericalError`]: If the `argmin` solver fails or doesn't find parameters.
    /// *   Errors from `self.model.validate_params()` if optimized parameters are invalid.
    ///
    /// # Side Effects
    ///
    /// Modifies the `model` field of `self` with optimized parameters.
    fn optimize(&mut self, verbose: bool) -> Result<(), CameraModelError> {
        // This is the same implementation as the original optimize method
        if self.points3d.ncols() != self.points2d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }
        if self.points3d.ncols() == 0 {
            return Err(CameraModelError::InvalidParams(
                "Points arrays cannot be empty for optimization".to_string(),
            ));
        }

        let cost_function = self.clone(); // Clone for use in argmin

        let initial_params_vec = vec![
            self.model.intrinsics.fx,
            self.model.intrinsics.fy,
            self.model.intrinsics.cx,
            self.model.intrinsics.cy,
            self.model.distortions[0], // k1
            self.model.distortions[1], // k2
            self.model.distortions[2], // k3
            self.model.distortions[3], // k4
        ];
        let init_param: DVector<f64> = DVector::from_vec(initial_params_vec);

        let solver = GaussNewton::new();

        let mut executor_builder = Executor::new(cost_function, solver)
            .configure(|state| state.param(init_param).max_iters(100));

        if verbose {
            executor_builder =
                executor_builder.add_observer(SlogLogger::term(), ObserverMode::Always);
            info!("Starting Kannala-Brandt optimization with Gauss-Newton...");
        } else {
            // Log only when a new best solution is found or on termination
            executor_builder =
                executor_builder.add_observer(SlogLogger::term(), ObserverMode::NewBest);
        }

        let res = executor_builder
            .run()
            .map_err(|e| CameraModelError::NumericalError(e.to_string()))?;

        if verbose { // Log final state if verbose
            info!("Optimization finished: \nResult state: {}", res);
            info!("Termination status: {:?}", res.state().termination_status);
        }

        if let Some(best_params_dvec) = res.state().get_best_param() {
            self.model.intrinsics.fx = best_params_dvec[0];
            self.model.intrinsics.fy = best_params_dvec[1];
            self.model.intrinsics.cx = best_params_dvec[2];
            self.model.intrinsics.cy = best_params_dvec[3];
            self.model.distortions[0] = best_params_dvec[4]; // k1
            self.model.distortions[1] = best_params_dvec[5]; // k2
            self.model.distortions[2] = best_params_dvec[6]; // k3
            self.model.distortions[3] = best_params_dvec[7]; // k4
            self.model.validate_params()?; // Validate the newly set parameters
        } else {
            return Err(CameraModelError::NumericalError(
                "Optimization failed to find best parameters".to_string(),
            ));
        }
        Ok(())
    }

    /// Provides an initial estimate for the 4 distortion parameters (`k1, k2, k3, k4`)
    /// of the [`KannalaBrandtModel`] using a linear least squares method.
    ///
    /// This method assumes that the intrinsic parameters (`fx, fy, cx, cy`) of the model
    /// are already reasonably known and fixed. It then sets up a linear system of equations
    /// based on the 3D-2D point correspondences to solve for the distortion coefficients.
    ///
    /// # Algorithm
    ///
    /// For each point correspondence:
    /// 1.  Calculate `theta = atan2(sqrt(X^2+Y^2), Z)`.
    /// 2.  The Kannala-Brandt projection for the radial distance is `theta_d = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9`.
    /// 3.  From observed pixel `(u,v)` and known intrinsics, calculate observed `theta_d_obs_x = ((u - cx)/fx) / (X/sqrt(X^2+Y^2))`
    ///     and `theta_d_obs_y = ((v - cy)/fy) / (Y/sqrt(X^2+Y^2))`.
    ///     A mean `theta_d_obs` can be used if X/r and Y/r are stable.
    /// 4.  Form equations: `theta_d_obs - theta = k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9`.
    /// 5.  These equations are stacked into a system `A * [k1,k2,k3,k4]^T = B` and solved.
    ///
    /// # Return Value
    ///
    /// Returns `Ok(())` if the linear estimation is successful and the model's
    /// distortion parameters are updated.
    ///
    /// # Errors
    ///
    /// *   [`CameraModelError::InvalidParams`]: If point counts mismatch or are insufficient (needs at least 4).
    /// *   [`CameraModelError::NumericalError`]: If SVD solving fails or other numerical issues occur
    ///     (e.g., division by zero if points are too close to the optical axis or camera center).
    ///
    /// # Side Effects
    ///
    /// Modifies the `distortions` field of the internal `self.model`.
    /// Calls `self.model.validate_params()` after updating distortion.
    fn linear_estimation(&mut self) -> Result<(), CameraModelError>
    where
        Self: Sized,
    {
        if self.points3d.ncols() != self.points2d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }
        // Each point gives 2 equations for 4 unknowns. Need at least 2 points for a determined system.
        // However, the original code checks for < 4 points. Let's keep that, assuming it might be
        // for stability or overdetermination.
        if self.points3d.ncols() < 4 {
            return Err(CameraModelError::InvalidParams(
                "Not enough points for linear estimation (need at least 4)".to_string(),
            ));
        }

        let num_points = self.points3d.ncols();
        let mut a_mat = DMatrix::zeros(num_points * 2, 4); // 4 unknowns: k1, k2, k3, k4
        let mut b_vec = DVector::zeros(num_points * 2);

        for i in 0..num_points {
            let p3d = self.points3d.column(i);
            let p2d = self.points2d.column(i);

            let x_world = p3d.x;
            let y_world = p3d.y;
            let z_world = p3d.z;

            let u_img = p2d.x;
            let v_img = p2d.y;

            if z_world <= f64::EPSILON { // Avoid points at or behind camera center
                // Skip point or return error, current impl skips
                continue;
            }

            let r_world = (x_world * x_world + y_world * y_world).sqrt();
            let theta = r_world.atan2(z_world); // Angle theta

            // Powers of theta for the polynomial
            let theta2 = theta * theta;
            let theta3 = theta2 * theta;
            let theta5 = theta3 * theta2;
            let theta7 = theta5 * theta2;
            let theta9 = theta7 * theta2;

            // Coefficients for matrix A
            // Row for u-coordinate
            a_mat[(i * 2, 0)] = theta3;
            a_mat[(i * 2, 1)] = theta5;
            a_mat[(i * 2, 2)] = theta7;
            a_mat[(i * 2, 3)] = theta9;
            // Row for v-coordinate (same coefficients for radial distortion)
            a_mat[(i * 2 + 1, 0)] = theta3;
            a_mat[(i * 2 + 1, 1)] = theta5;
            a_mat[(i * 2 + 1, 2)] = theta7;
            a_mat[(i * 2 + 1, 3)] = theta9;

            // x_r and y_r are x_world/r_world and y_world/r_world respectively
            let (x_r, y_r) = if r_world < f64::EPSILON { (0.0, 0.0) } else { (x_world / r_world, y_world / r_world) };

            // Target values for vector B: (theta_d_obs - theta)
            // theta_d_obs_x = ((u - cx)/fx) / x_r
            // theta_d_obs_y = ((v - cy)/fy) / y_r
            // These are approximations of theta_d
            if x_r.abs() > f64::EPSILON {
                 if (self.model.intrinsics.fx * x_r).abs() < f64::EPSILON {
                    return Err(CameraModelError::NumericalError("fx * x_r is near zero in linear estimation, division by zero.".to_string()));
                }
                b_vec[i * 2] = (u_img - self.model.intrinsics.cx) / (self.model.intrinsics.fx * x_r) - theta;
            } else {
                // If x_r is zero, point is on Y-axis (or optical axis).
                // If u_img is also cx, then this term is effectively 0. theta_d_obs_x * x_r = 0.
                // theta_d_obs_x - theta = 0 => 0 * k_params = -theta. This only works if theta is also 0.
                // If x_r is zero, the u-equation doesn't constrain k's well unless u=cx and theta=0.
                // Setting b_vec to -theta if u_img is close to cx, otherwise 0 to avoid NaN/Inf.
                b_vec[i * 2] = if (u_img - self.model.intrinsics.cx).abs() < f64::EPSILON { -theta } else { 0.0 };
            }

            if y_r.abs() > f64::EPSILON {
                if (self.model.intrinsics.fy * y_r).abs() < f64::EPSILON {
                     return Err(CameraModelError::NumericalError("fy * y_r is near zero in linear estimation, division by zero.".to_string()));
                }
                b_vec[i * 2 + 1] = (v_img - self.model.intrinsics.cy) / (self.model.intrinsics.fy * y_r) - theta;
            } else {
                b_vec[i * 2 + 1] = if (v_img - self.model.intrinsics.cy).abs() < f64::EPSILON { -theta } else { 0.0 };
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

        self.model.validate_params()?; // Validate intrinsics, as distortion validation is minimal.
        Ok(())
    }
}

/// Implements the [`Operator`] trait from `argmin` for [`KannalaBrandtOptimizationCost`].
///
/// The `Operator` trait defines how to compute the residuals (or cost vector) given
/// a set of parameters.
impl Operator for KannalaBrandtOptimizationCost {
    /// The parameter type is a `DVector<f64>` representing `[fx, fy, cx, cy, k1, k2, k3, k4]`.
    type Param = DVector<f64>;
    /// The output type is a `DVector<f64>` representing the reprojection error residuals
    /// `[err_u1, err_v1, err_u2, err_v2, ..., err_uN, err_vN]`.
    type Output = DVector<f64>;

    /// Computes the reprojection error residuals for the given camera parameters.
    ///
    /// For each 3D point, it projects the point using the provided camera parameters `p`
    /// and calculates the difference between the projected 2D point and the observed 2D point.
    /// These differences (residuals) for all points are concatenated into a single vector.
    /// Points that fail to project are currently logged and skipped, leading to a shorter residual vector.
    ///
    /// # Arguments
    ///
    /// * `p`: A `&DVector<f64>` containing the camera parameters `[fx, fy, cx, cy, k1, k2, k3, k4]`
    ///   for which to compute the residuals.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<DVector<f64>, argmin::core::Error>`.
    /// The `DVector<f64>` contains the concatenated reprojection residuals.
    ///
    /// # Errors
    ///
    /// *   Returns `argmin::core::Error` if [`KannalaBrandtModel::new(p)`] fails (e.g., if `p`
    ///     has incorrect length, causing `KannalaBrandtModel::new` to return an error).
    fn apply(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {
        let num_points = self.points3d.ncols();
        let mut residuals = DVector::zeros(num_points * 2);
        // Create a temporary model with parameters p for this evaluation
        let model = KannalaBrandtModel::new(&p)?; // Converts CameraModelError to ArgminError via From impl
        let mut valid_projections_count = 0;

        for i in 0..num_points {
            let p3d = &self.points3d.column(i).into_owned();
            let p2d_gt = &self.points2d.column(i).into_owned();

            match model.project(p3d, false) {
                Ok((p2d_projected, _)) => {
                    residuals[valid_projections_count * 2] = p2d_projected.x - p2d_gt.x;
                    residuals[valid_projections_count * 2 + 1] = p2d_projected.y - p2d_gt.y;
                    valid_projections_count += 1;
                }
                Err(_) => {
                    // Log a warning if a point fails to project.
                    // These points will be excluded from the residual vector.
                    info!("3d points {} are not projected during residual calculation", p3d);
                }
            }
        }
        // Resize residuals vector to only include valid projections
        residuals = residuals.rows(0, valid_projections_count * 2).into_owned();
        Ok(residuals)
    }
}

/// Implements the [`Jacobian`] trait from `argmin` for [`KannalaBrandtOptimizationCost`].
///
/// The `Jacobian` trait defines how to compute the Jacobian matrix of the residuals
/// with respect to the camera parameters.
impl Jacobian for KannalaBrandtOptimizationCost {
    /// The parameter type is a `DVector<f64>` representing `[fx, fy, cx, cy, k1, k2, k3, k4]`.
    type Param = DVector<f64>;
    /// The Jacobian type is a `DMatrix<f64>`.
    type Jacobian = DMatrix<f64>;

    /// Computes the Jacobian of the reprojection residuals with respect to camera parameters.
    ///
    /// The Jacobian matrix `J` has dimensions `(2*M) x 8`, where `M` is the number of
    /// successfully projected points and 8 is the number of parameters. Each pair of rows
    /// in `J` corresponds to a 2D point and contains the partial derivatives of its
    /// reprojection residuals `(err_u, err_v)` with respect to each of the 8 parameters.
    /// Points that fail to project are skipped.
    ///
    /// # Arguments
    ///
    /// * `p`: A `&DVector<f64>` containing the camera parameters at which to evaluate the Jacobian.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<DMatrix<f64>, argmin::core::Error>`.
    /// The `DMatrix<f64>` is the computed Jacobian matrix.
    ///
    /// # Errors
    ///
    /// *   Returns `argmin::core::Error` if [`KannalaBrandtModel::new(p)`] fails.
    fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, ArgminError> {
        let num_points = self.points3d.ncols();
        // Max possible rows: num_points * 2. Actual rows depend on successful projections.
        let mut jacobian_matrix = DMatrix::zeros(num_points * 2, 8);
        let model = KannalaBrandtModel::new(&p)?; // Create model from parameters
        let mut valid_projections_count = 0;

        for i in 0..num_points {
            let p3d = &self.points3d.column(i).into_owned();
            match model.project(p3d, true) { // Request Jacobian from project method
                Ok((_, Some(jac_point))) => { // jac_point is the 2x8 Jacobian for this point
                    jacobian_matrix
                        .view_mut((valid_projections_count * 2, 0), (2, 8))
                        .copy_from(&jac_point);
                    valid_projections_count += 1;
                }
                Ok((_, None)) => {
                    // Jacobian was requested but not returned by project
                    info!("3d points {} doesn't have jacobian, though requested.", p3d);
                }
                Err(_) => {
                    // Projection failed for this point
                    info!("3d points {} are not projected during Jacobian calculation.", p3d);
                }
            }
        }
        // Resize Jacobian matrix to only include rows from valid projections
        jacobian_matrix = jacobian_matrix.rows(0, valid_projections_count * 2).into_owned();
        Ok(jacobian_matrix)
    }
}

/// Implements the [`CostFunction`] trait from `argmin` for [`KannalaBrandtOptimizationCost`].
///
/// The `CostFunction` trait defines how to compute the scalar cost value from the parameters.
impl CostFunction for KannalaBrandtOptimizationCost {
    /// The parameter type is a `DVector<f64>` representing `[fx, fy, cx, cy, k1, k2, k3, k4]`.
    type Param = DVector<f64>;
    /// The output type is an `f64` representing the total cost (half the sum of squared errors).
    type Output = f64;

    /// Computes the total cost as half the sum of squared reprojection errors.
    ///
    /// This method calculates the reprojection error residuals using `apply` and then
    /// computes `0.5 * ||residuals||^2`.
    ///
    /// # Arguments
    ///
    /// * `p`: A `&DVector<f64>` containing the camera parameters for which to compute the cost.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<f64, argmin::core::Error>`. The `f64` value is the total cost.
    ///
    /// # Errors
    ///
    /// Propagates errors from the `apply` method (e.g., if model creation or projection fails).
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {
        let residuals = self.apply(p)?;
        Ok(residuals.norm_squared() / 2.0) // Standard least squares cost (0.5 * sum of squares)
    }
}

/// Implements the [`Gradient`] trait from `argmin` for [`KannalaBrandtOptimizationCost`].
///
/// The `Gradient` trait defines how to compute the gradient of the cost function.
impl Gradient for KannalaBrandtOptimizationCost {
    /// The parameter type is a `DVector<f64>` representing `[fx, fy, cx, cy, k1, k2, k3, k4]`.
    type Param = DVector<f64>;
    /// The gradient type is a `DVector<f64>` of the same dimension as `Param`.
    type Gradient = DVector<f64>;

    /// Computes the gradient of the cost function with respect to camera parameters.
    ///
    /// The gradient `g` is computed as `g = J^T * r`, where `J` is the Jacobian matrix of the
    /// residuals and `r` is the vector of residuals.
    ///
    /// # Arguments
    ///
    /// * `p`: A `&DVector<f64>` containing the camera parameters at which to evaluate the gradient.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<DVector<f64>, argmin::core::Error>`. The `DVector<f64>` is the
    /// computed gradient vector (8x1).
    ///
    /// # Errors
    ///
    /// Propagates errors from `jacobian` or `apply` methods.
    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        let jacobian = self.jacobian(p)?;
        let residuals = self.apply(p)?;
        Ok(jacobian.transpose() * residuals)
    }
}

/// Implements the [`Hessian`] trait from `argmin` for [`KannalaBrandtOptimizationCost`].
///
/// The `Hessian` trait defines how to compute the Hessian matrix (or its approximation)
/// of the cost function.
impl Hessian for KannalaBrandtOptimizationCost {
    /// The parameter type is a `DVector<f64>` representing `[fx, fy, cx, cy, k1, k2, k3, k4]`.
    type Param = DVector<f64>;
    /// The Hessian type is a `DMatrix<f64>`.
    type Hessian = DMatrix<f64>;

    /// Computes the Gauss-Newton approximation of the Hessian matrix.
    ///
    /// The Hessian `H` is approximated as `H = J^T * J`, where `J` is the Jacobian matrix
    /// of the residuals. This is standard for solvers like Gauss-Newton.
    ///
    /// # Arguments
    ///
    /// * `p`: A `&DVector<f64>` containing the camera parameters at which to evaluate the Hessian.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<DMatrix<f64>, argmin::core::Error>`. The `DMatrix<f64>` is the
    /// approximated Hessian matrix (8x8).
    ///
    /// # Errors
    ///
    /// Propagates errors from the `jacobian` method.
    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, ArgminError> {
        let jacobian = self.jacobian(p)?;
        Ok(jacobian.transpose() * jacobian) // Gauss-Newton approximation J^T * J
    }
}

+/// Unit tests for [`KannalaBrandtOptimizationCost`] and its trait implementations.
 #[cfg(test)]
 mod tests {
     use super::*;
     use crate::camera::{CameraModel, Intrinsics, KannalaBrandtModel as KBCameraModel, Resolution};
     use crate::optimization::Optimizer;
     use approx::assert_relative_eq;
-    use log::info;
+    use log::info; // For logging within tests if needed
     use nalgebra::{Matrix2xX, Matrix3xX, Vector2, Vector3};
 
-    // Helper function to get a sample KannalaBrandtModel instance
-    // This could load from YAML or use the new() function with sample parameters
+    /// Helper function to create a sample [`KBCameraModel`] instance for testing.
+    /// Parameters are based on "samples/kannala_brandt.yaml".
     fn get_sample_kb_camera_model() -> KBCameraModel {
         // Parameters from samples/kannala_brandt.yaml
         // intrinsics: [461.58688085556616, 460.2811732644195, 366.28603126815506, 249.08026891791644]
@@ -303,8 +735,9 @@
         model
     }
 
-    // Placeholder for geometry::sample_points or a simplified version
+    /// Helper function to generate sample 2D and 3D points for a given [`KBCameraModel`].
+    /// It projects synthetic 3D points to 2D using the model.
     fn sample_points_for_kb_model(
         model: &KBCameraModel,
         num_points: usize,
@@ -339,6 +772,9 @@
         )
     }
 
+    /// Tests basic functionality of [`KannalaBrandtOptimizationCost`] including `argmin` trait methods.
+    /// It checks if residuals, cost, Jacobian, gradient, and Hessian are computed correctly
+    /// when the model parameters perfectly match the data generation model.
     #[test]
     fn test_kannala_brandt_optimization_cost_basic() {
         let model_camera = get_sample_kb_camera_model();
@@ -384,6 +820,9 @@
         assert_eq!(hess.ncols(), 8);
     }
 
+    /// Tests the `Optimizer::optimize` method for [`KannalaBrandtOptimizationCost`].
+    /// It starts with a noisy model and checks if optimization brings parameters closer
+    /// to a reference model.
     #[test]
     fn test_kannala_brandt_optimize_trait_method() {
         let reference_model = get_sample_kb_camera_model();
@@ -421,6 +860,9 @@
         }
     }
 
+    /// Tests the `Optimizer::linear_estimation` method for [`KannalaBrandtOptimizationCost`].
+    /// It checks if the linear estimation of distortion parameters is reasonable when
+    /// intrinsics are assumed known.
     #[test]
     fn test_kannala_brandt_linear_estimation_optimizer_trait() {
         let reference_model = get_sample_kb_camera_model();
@@ -457,4 +899,4 @@
         assert_relative_eq!(estimated_model.intrinsics.cy, reference_model.intrinsics.cy, epsilon = 1e-9);
     }
 }
-
[end of src/optimization/kannala_brandt.rs]
