//! Provides optimization capabilities for the Double Sphere camera model.
//!
//! This module defines [`DoubleSphereOptimizationCost`], a struct that encapsulates
//! the necessary data (camera model and 2D-3D point correspondences) and implements
//! various traits from the `argmin` optimization library. It is used to refine
//! the parameters of a [`DoubleSphereModel`] by minimizing reprojection errors.
//!
//! This module implements the [`Optimizer`] trait from the parent `optimization`
//! module, offering both non-linear optimization (via `argmin` and Gauss-Newton)
//! and a linear estimation method for initializing parameters.

use crate::camera::{CameraModel, CameraModelError, DoubleSphereModel};
use crate::optimization::Optimizer;
use argmin::{
    core::{
        observers::ObserverMode, CostFunction, Error, Executor, Gradient, Hessian, Jacobian,
        Operator, State,
    },
    solver::gaussnewton::GaussNewton,
};
use argmin_observer_slog::SlogLogger; // For logging optimization progress
use log::{info, warn}; // For general logging
use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3xX};
use std::fmt;

/// Cost function for optimizing the parameters of a [`DoubleSphereModel`].
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
/// * `model`: [`DoubleSphereModel`] - The camera model instance whose parameters are being optimized.
///   This model is updated during the optimization process.
/// * `points3d`: [`Matrix3xX<f64>`] - A matrix where each column represents a 3D point
///   in the camera's coordinate system. These are the reference points.
/// * `points2d`: [`Matrix2xX<f64>`] - A matrix where each column represents the observed 2D projection
///   (in pixel coordinates) of the corresponding 3D point in `points3d`.
///
/// # Examples
/// ```rust
/// use nalgebra::{Matrix2xX, Matrix3xX, Vector3, Vector2};
/// use vision_toolkit_rs::camera::{DoubleSphereModel, Intrinsics, Resolution};
/// use vision_toolkit_rs::optimization::double_sphere::DoubleSphereOptimizationCost;
///
/// // Assume 'initial_model' is a DoubleSphereModel instance
/// # let initial_model = DoubleSphereModel {
/// #     intrinsics: Intrinsics { fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0 },
/// #     resolution: Resolution { width: 640, height: 480 },
/// #     alpha: 0.5, xi: 0.1,
/// # };
/// // Assume 'world_points' and 'image_points' are matrices of corresponding points
/// # let world_points = Matrix3xX::from_columns(&[Vector3::new(0.0,0.0,1.0)]);
/// # let image_points = Matrix2xX::from_columns(&[Vector2::new(320.0,240.0)]);
///
/// let cost_function_context = DoubleSphereOptimizationCost::new(
///     initial_model,
///     world_points,
///     image_points
/// );
/// // This 'cost_function_context' can then be used with an Optimizer.
/// println!("{:?}", cost_function_context); // Example of using Debug trait
/// ```
#[derive(Clone)]
pub struct DoubleSphereOptimizationCost {
    /// The [`DoubleSphereModel`] instance that is being optimized.
    /// Its parameters are updated by the optimization process.
    model: DoubleSphereModel,
    /// A 3xN matrix where each column is a 3D point in the camera's coordinate system.
    points3d: Matrix3xX<f64>,
    /// A 2xN matrix where each column is the observed 2D projection (in pixels)
    /// of the corresponding 3D point in `points3d`.
    points2d: Matrix2xX<f64>,
}

impl DoubleSphereOptimizationCost {
    /// Creates a new [`DoubleSphereOptimizationCost`] instance for optimizing a [`DoubleSphereModel`].
    ///
    /// This constructor initializes the cost function context with the camera model
    /// to be optimized and the sets of corresponding 3D world points and 2D image observations.
    ///
    /// # Arguments
    ///
    /// * `model`: [`DoubleSphereModel`] - The initial state of the Double Sphere camera model
    ///   whose parameters will be optimized.
    /// * `points3d`: [`Matrix3xX<f64>`] - A matrix of 3D points (3xN), where each column
    ///   is a point in the camera's coordinate system.
    /// * `points2d`: [`Matrix2xX<f64>`] - A matrix of 2D points (2xN), where each column
    ///   is the observed projection of the corresponding 3D point, in pixel coordinates.
    ///
    /// # Return Value
    ///
    /// Returns a new instance of `DoubleSphereOptimizationCost`.
    ///
    /// # Panics
    ///
    /// Panics if the number of columns (points) in `points3d` and `points2d` do not match,
    /// as verified by `assert_eq!(points3d.ncols(), points2d.ncols());`.
    pub fn new(
        model: DoubleSphereModel,
        points3d: Matrix3xX<f64>,
        points2d: Matrix2xX<f64>,
    ) -> Self {
        assert_eq!(points3d.ncols(), points2d.ncols());
        DoubleSphereOptimizationCost {
            model,
            points3d: points3d, // Consider .clone() if ownership is not fully transferred
            points2d: points2d, // Consider .clone()
        }
    }
}

/// Provides a debug string representation for [`DoubleSphereOptimizationCost`].
/// This includes a summary of the model and the number of 3D and 2D points.
impl fmt::Debug for DoubleSphereOptimizationCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DoubleSphereCostModel Summary:\n model: {:?}\n points3d size: {}, points2d size: {} ",
            self.model, // Relies on DoubleSphereModel's Debug impl
            self.points3d.ncols(),
            self.points2d.ncols(),
        )
    }
}

/// Implements the [`Optimizer`] trait for [`DoubleSphereOptimizationCost`].
impl Optimizer for DoubleSphereOptimizationCost {
    /// Performs non-linear optimization of the [`DoubleSphereModel`] parameters.
    ///
    /// This method uses the `argmin` library with a Gauss-Newton solver to refine
    /// the camera model's parameters (intrinsics `fx, fy, cx, cy` and distortion
    /// parameters `alpha, xi`). The optimization aims to minimize the sum of squared
    /// reprojection errors between the observed 2D points and the 2D points projected
    /// from the 3D points using the current model parameters.
    ///
    /// # Algorithm
    ///
    /// 1.  Validates that the number of 3D and 2D points match and are non-zero.
    /// 2.  Clones the current `DoubleSphereOptimizationCost` instance to serve as the
    ///     cost function for `argmin`.
    /// 3.  Extracts initial parameters (`fx, fy, cx, cy, alpha, xi`) from the internal `self.model`.
    /// 4.  Configures a `GaussNewton` solver from the `argmin` library.
    /// 5.  Sets up an `Executor` with the solver, cost function, initial parameters,
    ///     and a maximum number of iterations (100).
    /// 6.  Optionally enables logging via `SlogLogger` if `verbose` is true (though current
    ///     observer mode is `Never`).
    /// 7.  Runs the optimization.
    /// 8.  If successful, updates the internal `self.model` with the optimized parameters.
    /// 9.  Validates the final optimized parameters using `self.model.validate_params()`.
    ///
    /// # Arguments
    ///
    /// * `verbose`: `bool` - If true, enables informational logging during the optimization
    ///   process. Currently, the `SlogLogger` is added with `ObserverMode::Never`,
    ///   so detailed iteration logs might not appear unless this mode is changed.
    ///   However, `info!` logs within this method will be conditional on `verbose`.
    ///
    /// # Return Value
    ///
    /// Returns `Ok(())` if the optimization completes successfully and the final parameters
    /// are valid.
    ///
    /// # Errors
    ///
    /// *   [`CameraModelError::InvalidParams`]: If the number of 3D and 2D points do not match,
    ///     or if the point arrays are empty.
    /// *   [`CameraModelError::NumericalError`]: If the `argmin` solver encounters an error
    ///     during execution (e.g., numerical instability, failure to converge).
    /// *   Errors propagated from `self.model.validate_params()` if the optimized parameters
    ///     are invalid (e.g., `alpha` out of range).
    ///
    /// # Side Effects
    ///
    /// Modifies the `model` field of `self` by updating its parameters to the
    /// optimized values.
    fn optimize(&mut self, verbose: bool) -> Result<(), CameraModelError> {
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
        if verbose { // Conditional logging
            info!("Cost function setup: {:?}", cost_function);
        }

        // Initial parameters from the current model state
        let param = vec![
            self.model.intrinsics.fx,
            self.model.intrinsics.fy,
            self.model.intrinsics.cx,
            self.model.intrinsics.cy,
            self.model.alpha,
            self.model.xi,
        ];

        let init_param: DVector<f64> = DVector::from_vec(param);

        // Configure the Gauss-Newton solver
        let solver = GaussNewton::new();

        // Setup executor with the solver and cost function
        // The SlogLogger is added, but ObserverMode::Never means it won't log iterations by default.
        // Change to ObserverMode::Always or ObserverMode::NewBest for more logs if needed.
        let executor_builder = Executor::new(cost_function, solver)
            .configure(|state| state.param(init_param).max_iters(100))
            .add_observer(SlogLogger::term(), ObserverMode::Never);

        if verbose {
            info!("Starting optimization with Gauss-Newton...");
        }

        let res = executor_builder
            .run()
            .map_err(|e| CameraModelError::NumericalError(e.to_string()))?;

        if verbose {
            info!("Optimization finished: \nResult state: {}", res);
            info!("Termination status: {:?}", res.state().termination_status);
        }

        // Retrieve the best parameters found by the optimizer
        let best_params_dvec = res.state().get_best_param().ok_or_else(||
            CameraModelError::NumericalError("Optimizer did not return best parameters.".to_string())
        )?.clone();

        // Update model parameters from the final parameters
        self.model.intrinsics.fx = best_params_dvec[0];
        self.model.intrinsics.fy = best_params_dvec[1];
        self.model.intrinsics.cx = best_params_dvec[2];
        self.model.intrinsics.cy = best_params_dvec[3];
        self.model.alpha = best_params_dvec[4];
        self.model.xi = best_params_dvec[5];

        // Validate the optimized parameters
        self.model.validate_params()?;

        Ok(())
    }

    /// Provides an initial estimate for the `alpha` distortion parameter of the [`DoubleSphereModel`].
    ///
    /// This method uses a linear least squares approach to estimate the `alpha` parameter,
    /// assuming the intrinsic parameters (`fx, fy, cx, cy`) and `xi` are known and fixed
    /// (though `xi` is effectively treated as zero or not estimated in this linear formulation).
    /// This estimation can serve as a starting point for non-linear optimization.
    ///
    /// # Algorithm
    ///
    /// The estimation is based on rearranging the Double Sphere projection equations
    /// to form a linear system `A * [alpha] = B`. For each 3D-2D point correspondence:
    /// 1.  Let `d = sqrt(x^2 + y^2 + z^2)`.
    /// 2.  Let `u_cx = u - cx` and `v_cy = v - cy`.
    /// 3.  Two equations are formed:
    ///     `u_cx * (d - z) * alpha = fx*x - u_cx*z`
    ///     `v_cy * (d - z) * alpha = fy*y - v_cy*z`
    /// 4.  These are stacked into the system `A * [alpha] = B` and solved for `alpha`
    ///     using SVD-based least squares.
    ///
    /// # Return Value
    ///
    /// Returns `Ok(())` if the linear estimation is successful and the model's `alpha`
    /// parameter is updated.
    ///
    /// # Errors
    ///
    /// *   [`CameraModelError::InvalidParams`]: If the number of 2D and 3D points do not match,
    ///     or if points arrays are empty.
    /// *   [`CameraModelError::NumericalError`]: If the SVD solver fails to find a solution.
    ///
    /// # Side Effects
    ///
    /// Modifies the `alpha` parameter of the internal `self.model`. The `xi` parameter
    /// is not modified by this linear estimation. The `validate_params()` method for the model
    /// is commented out in the original implementation of this function, so it's not called here.
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
        if self.points3d.ncols() == 0 {
             return Err(CameraModelError::InvalidParams(
                "Points arrays cannot be empty for linear estimation".to_string(),
            ));
        }
+
+
        // Set up the linear system to solve for alpha
        let num_points = self.points2d.ncols();
-        let mut a = nalgebra::DMatrix::zeros(num_points * 2, 1);
+        let mut a = nalgebra::DMatrix::zeros(num_points * 2, 1); // One unknown: alpha
         let mut b = nalgebra::DVector::zeros(num_points * 2);
 
         for i in 0..num_points {
@@ -156,9 +373,15 @@
             let v = self.points2d[(1, i)];
 
             let d = (x * x + y * y + z * z).sqrt();
+            if d < f64::EPSILON { // Avoid issues if point is at origin
+                // Skip point or handle error; for now, skip to avoid division by zero or NaN
+                continue;
+            }
             let u_cx = u - self.model.intrinsics.cx;
             let v_cy = v - self.model.intrinsics.cy;
 
+            // From rearranged projection equations to solve for alpha
+            // alpha * u_cx * (d - z) = fx*x - u_cx*z
+            // alpha * v_cy * (d - z) = fy*y - v_cy*z
             a[(i * 2, 0)] = u_cx * (d - z);
             a[(i * 2 + 1, 0)] = v_cy * (d - z);
 
@@ -168,68 +391,80 @@
 
         // Solve the linear system using SVD
         let svd = a.svd(true, true);
-        let alpha = match svd.solve(&b, 1e-10) {
-            Ok(sol) => sol[0], // Handle the successful case
+        let alpha_vec = match svd.solve(&b, 1e-10) { // SVD based least squares solution
+            Ok(sol) => sol,
             Err(err_msg) => {
-                return Err(CameraModelError::NumericalError(err_msg.to_string()));
+                return Err(CameraModelError::NumericalError(format!(
+                    "SVD solve failed for linear estimation of alpha: {}", err_msg
+                )));
             }
         };
+        let alpha = alpha_vec[0];
 
         self.model.alpha = alpha;
-        // Validate parameters
-        // self.model.validate_params()?;
-
+        // Note: The original code has `self.model.validate_params()?` commented out here.
+        // Documenting current behavior: parameters are not re-validated after this estimation.
         Ok(())
     }
 }
 
-/// Implementation of the Operator trait for Gauss-Newton optimization.
+/// Implements the [`Operator`] trait from `argmin` for [`DoubleSphereOptimizationCost`].
+///
+/// The `Operator` trait defines how to compute the residuals (or cost vector) given
+/// a set of parameters.
 impl Operator for DoubleSphereOptimizationCost {
-    /// Parameter vector: [fx, fy, cx, cy, alpha, xi]
+    /// The parameter type is a `DVector<f64>` representing `[fx, fy, cx, cy, alpha, xi]`.
     type Param = DVector<f64>;
-    /// Output residuals vector (2×N elements for N points)
+    /// The output type is a `DVector<f64>` representing the reprojection error residuals
+    /// `[err_u1, err_v1, err_u2, err_v2, ..., err_uN, err_vN]`.
     type Output = DVector<f64>;
 
-    /// Applies the camera model to compute projection residuals.
+    /// Computes the reprojection error residuals for the given camera parameters.
+    ///
+    /// For each 3D point, it projects the point using the provided camera parameters `p`
+    /// and calculates the difference between the projected 2D point and the observed 2D point.
+    /// These differences (residuals) for all points are concatenated into a single vector.
     ///
     /// # Arguments
     ///
-    /// * `p` - Parameter vector containing camera intrinsics and distortion parameters
+    /// * `p`: A `&DVector<f64>` containing the camera parameters `[fx, fy, cx, cy, alpha, xi]`
+    ///   for which to compute the residuals.
     ///
-    /// # Returns
+    /// # Return Value
     ///
-    /// Vector of residuals between projected and observed 2D points.
+    /// Returns a `Result<DVector<f64>, argmin::core::Error>`.
+    /// The `DVector<f64>` contains the concatenated reprojection residuals `(u_proj - u_obs, v_proj - v_obs)`
+    /// for all points. The length of this vector is `2 * num_points`.
     ///
     /// # Errors
     ///
-    /// Returns an error if the camera model parameters are invalid or
-    /// if projection fails for the given points.
+    /// *   Returns `argmin::core::Error` if `DoubleSphereModel::new(p)` fails (e.g., if `p`
+    ///     contains invalid parameters that would cause `DoubleSphereModel::new` to error,
+    ///     though the current `new` panics on length issues rather than returning an error).
+    /// *   If projection of a point fails (e.g., point is behind camera), that point is currently
+    ///     skipped, and a warning is logged. This might lead to a residuals vector shorter
+    ///     than `2 * total_points_input` if not all points are successfully projected.
     fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
-        let num_points = self.points3d.ncols();
-        let mut residuals = DVector::zeros(num_points * 2);
-        // Ensure DoubleSphereModel::new is public or accessible
-        let model = DoubleSphereModel::new(&p)?;
-        let mut counter = 0;
+        let num_points_initial = self.points3d.ncols();
+        let mut residuals = DVector::zeros(num_points_initial * 2);
+        // Create a temporary model with parameters p for this evaluation
+        let model = DoubleSphereModel::new(&p)?; // This can return CameraModelError, needs conversion or handling
+        let mut valid_projections_count = 0;
 
-        for i in 0..num_points {
+        for i in 0..num_points_initial {
             let p3d = &self.points3d.column(i).into_owned();
             let p2d_gt = &self.points2d.column(i).into_owned();
-            // let project_result = model.project(p3d, false);
 
             match model.project(p3d, false) {
-                Ok((p2d_projected, _)) => { // Corrected destructuring here
-                    residuals[counter * 2] = p2d_projected.x - p2d_gt.x;
-                    residuals[counter * 2 + 1] = p2d_projected.y - p2d_gt.y;
-                    counter += 1;
+                Ok((p2d_projected, _)) => {
+                    residuals[valid_projections_count * 2] = p2d_projected.x - p2d_gt.x;
+                    residuals[valid_projections_count * 2 + 1] = p2d_projected.y - p2d_gt.y;
+                    valid_projections_count += 1;
                 }
                 Err(err_msg) => {
-                    warn!("Projection failed for point {}: {}", i, err_msg);
+                    // Log a warning if a point fails to project.
+                    // These points will be excluded from the residual vector.
+                    warn!("Projection failed for point {} during residual calculation: {}", i, err_msg);
                 }
             }
         }
-        // Only return the rows with actual residuals
-        residuals = residuals.rows(0, counter * 2).into_owned();
-        info!("Size residuals: {}", residuals.len());
+        // Resize residuals vector to only include valid projections
+        residuals = residuals.rows(0, valid_projections_count * 2).into_owned();
+        if verbose_logging_enabled() { // Placeholder for actual verbose check
+            info!("Residuals calculated. Size: {}", residuals.len());
+        }
         Ok(residuals)
     }
 }
+/// Placeholder for a global or configurable verbose logging check.
+fn verbose_logging_enabled() -> bool { false }
 
-/// Implementation of the Jacobian trait for Gauss-Newton optimization.
+
+/// Implements the [`Jacobian`] trait from `argmin` for [`DoubleSphereOptimizationCost`].
+///
+/// The `Jacobian` trait defines how to compute the Jacobian matrix of the residuals
+/// with respect to the camera parameters.
 impl Jacobian for DoubleSphereOptimizationCost {
+    /// The parameter type is a `DVector<f64>` representing `[fx, fy, cx, cy, alpha, xi]`.
     type Param = DVector<f64>;
+    /// The Jacobian type is a `DMatrix<f64>`.
     type Jacobian = DMatrix<f64>;
 
-    /// Computes the Jacobian matrix of residuals with respect to camera parameters.
+    /// Computes the Jacobian of the reprojection residuals with respect to camera parameters.
+    ///
+    /// The Jacobian matrix `J` has dimensions `(2*M) x 6`, where `M` is the number of
+    /// successfully projected points and 6 is the number of parameters
+    /// (`fx, fy, cx, cy, alpha, xi`). Each pair of rows in `J` corresponds to a 2D point
+    /// and contains the partial derivatives of its reprojection residuals `(err_u, err_v)`
+    /// with respect to each of the 6 parameters.
     ///
     /// # Arguments
     ///
-    /// * `p` - Parameter vector containing camera intrinsics and distortion parameters
+    /// * `p`: A `&DVector<f64>` containing the camera parameters at which to evaluate the Jacobian.
     ///
-    /// # Returns
+    /// # Return Value
     ///
-    /// Jacobian matrix (2N×6) where N is the number of points and 6 is the number of parameters.
+    /// Returns a `Result<DMatrix<f64>, argmin::core::Error>`.
+    /// The `DMatrix<f64>` is the computed Jacobian matrix.
     ///
     /// # Errors
     ///
-    /// Returns an error if the camera model parameters are invalid or
-    /// if projection fails for the given points.
+    /// *   Returns `argmin::core::Error` if `DoubleSphereModel::new(p)` fails.
+    /// *   If projection or Jacobian computation for a specific point fails, that point is
+    ///     skipped, a warning is logged, and its rows are not included in the final Jacobian.
     fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, Error> {
-        let num_points = self.points3d.ncols();
-        let mut jacobian = DMatrix::zeros(num_points * 2, 6); // 2 residuals per point, 6 parameters
-                                                              // Ensure DoubleSphereModel::new is public or accessible
-        let model = DoubleSphereModel::new(&p)?;
-        let mut counter = 0;
+        let num_points_initial = self.points3d.ncols();
+        // Max possible rows: num_points * 2. Actual rows depend on successful projections.
+        let mut jacobian_matrix = DMatrix::zeros(num_points_initial * 2, 6);
+        let model = DoubleSphereModel::new(&p)?; // Create model from parameters
+        let mut valid_projections_count = 0;
 
-        for i in 0..num_points {
+        for i in 0..num_points_initial {
             let p3d = &self.points3d.column(i).into_owned();
 
-
-            match model.project(p3d, true) {
-                Ok((_, Some(jac))) => {
-                    jacobian.view_mut((counter * 2, 0), (2, 6)).copy_from(&jac);
-                    counter += 1;
+            match model.project(p3d, true) { // Request Jacobian from project method
+                Ok((_, Some(jac_point))) => { // jac_point is the 2x6 Jacobian for this point
+                    jacobian_matrix.view_mut((valid_projections_count * 2, 0), (2, 6)).copy_from(&jac_point);
+                    valid_projections_count += 1;
                 }
                 Ok((_, None)) => {
-                    // This case can happen if Jacobian computation is not possible for a point
-                    // even if requested. Log a warning and skip this point's Jacobian.
-                    warn!("Jacobian not computed for point {} even when requested.", i);
+                    // Jacobian was requested but not returned by project (should not happen if true is passed)
+                    warn!("Jacobian requested but not computed by project method for point {}.", i);
                 }
                 Err(err_msg) => {
-                    warn!("Projection failed for point {}: {}", i, err_msg);
+                    // Projection failed for this point, so its Jacobian cannot be computed.
+                    warn!("Projection failed for point {} during Jacobian calculation: {}", i, err_msg);
                 }
             }
         }
-        jacobian = jacobian.rows(0, counter * 2).into_owned();
-        info!("Size residuals: {}", jacobian.nrows());
-        Ok(jacobian)
+        // Resize Jacobian matrix to only include rows from valid projections
+        jacobian_matrix = jacobian_matrix.rows(0, valid_projections_count * 2).into_owned();
+        if verbose_logging_enabled() { // Placeholder for actual verbose check
+             info!("Jacobian calculated. Size: {}x{}", jacobian_matrix.nrows(), jacobian_matrix.ncols());
+        }
+        Ok(jacobian_matrix)
     }
 }
 
-/// Implementation of the CostFunction trait for optimization.
+/// Implements the [`CostFunction`] trait from `argmin` for [`DoubleSphereOptimizationCost`].
+///
+/// The `CostFunction` trait defines how to compute the scalar cost value from the parameters.
 impl CostFunction for DoubleSphereOptimizationCost {
-    /// Parameter vector: [fx, fy, cx, cy, alpha, xi]
+    /// The parameter type is a `DVector<f64>` representing `[fx, fy, cx, cy, alpha, xi]`.
     type Param = DVector<f64>;
-    /// Sum of squared errors
+    /// The output type is an `f64` representing the total cost (sum of squared errors).
     type Output = f64;
 
-    /// Computes the cost function as the sum of squared projection errors.
+    /// Computes the total cost as the sum of squared reprojection errors.
+    ///
+    /// This method calculates the reprojection error for each 3D-2D point pair,
+    /// squares the norm of each error vector (L2 norm squared), and sums these
+    /// squared norms to get the total cost.
     ///
     /// # Arguments
     ///
-    /// * `p` - Parameter vector containing camera intrinsics and distortion parameters
+    /// * `p`: A `&DVector<f64>` containing the camera parameters for which to compute the cost.
     ///
-    /// # Returns
+    /// # Return Value
     ///
-    /// Total cost as the sum of squared residuals.
+    /// Returns a `Result<f64, argmin::core::Error>`. The `f64` value is the total cost.
     ///
     /// # Errors
     ///
-    /// Returns an error if the camera model parameters are invalid or
-    /// if projection fails for the given points.
+    /// *   Returns `argmin::core::Error` if `DoubleSphereModel::new(p)` fails.
+    /// *   If projection of a point fails, that point is effectively excluded from the cost
+    ///     summation (as `apply` would skip it), or `unwrap()` on projection result might panic
+    ///     if not handled gracefully. The current code `model.project(...).unwrap()` will panic.
+    ///     A robust version should handle projection errors.
     fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
         let mut total_error_sq = 0.0;
-        // Ensure DoubleSphereModel::new is public or accessible
-        let model = DoubleSphereModel::new(&p)?;
+        let model = DoubleSphereModel::new(&p)?; // Create model from parameters
 
         for i in 0..self.points3d.ncols() {
             let p3d = &self.points3d.column(i).into_owned();
             let p2d_gt = &self.points2d.column(i).into_owned();
 
             // The project function now returns a tuple with the projection and optional Jacobian
-            let (p2d_projected, _) = model.project(p3d, false).unwrap();
-
-            total_error_sq += (p2d_projected - p2d_gt).norm();
+            // Assuming project errors are critical for cost calculation, so unwrap or handle.
+            match model.project(p3d, false) {
+                Ok((p2d_projected, _)) => {
+                    total_error_sq += (p2d_projected - p2d_gt).norm_squared(); // Sum of squared norms
+                }
+                Err(err) => {
+                    // If a point cannot be projected, it contributes significantly to error,
+                    // or could be handled by returning a very large cost or an Error.
+                    // For now, let's log and skip, or return an error to stop optimization.
+                    warn!("Projection failed for point {} during cost calculation: {}. This point will be skipped.", i, err);
+                    // Optionally, return an error:
+                    // return Err(Error::Msg(format!("Projection failed for point {} in cost: {}", i, err)));
+                }
+            }
         }
-
-        info!("total_error_sq: {total_error_sq}");
+        if verbose_logging_enabled() { // Placeholder
+            info!("Total squared error cost: {}", total_error_sq);
+        }
         Ok(total_error_sq)
     }
 }
 
-/// Implementation of the Gradient trait for optimization.
+/// Implements the [`Gradient`] trait from `argmin` for [`DoubleSphereOptimizationCost`].
+///
+/// The `Gradient` trait defines how to compute the gradient of the cost function.
 impl Gradient for DoubleSphereOptimizationCost {
+    /// The parameter type is a `DVector<f64>` representing `[fx, fy, cx, cy, alpha, xi]`.
     type Param = DVector<f64>;
-    /// Gradient of the cost function (J^T * r)
+    /// The gradient type is a `DVector<f64>` of the same dimension as `Param`.
     type Gradient = DVector<f64>;
 
-    /// Computes the gradient of the cost function.
+    /// Computes the gradient of the cost function with respect to camera parameters.
     ///
-    /// The gradient is computed as J^T * r, where J is the Jacobian matrix
-    /// and r is the residual vector.
+    /// The gradient `g` is computed as `g = J^T * r`, where `J` is the Jacobian matrix of the
+    /// residuals and `r` is the vector of residuals. This is standard for non-linear
+    /// least squares problems.
     ///
     /// # Arguments
     ///
-    /// * `p` - Parameter vector containing camera intrinsics and distortion parameters
+    /// * `p`: A `&DVector<f64>` containing the camera parameters at which to evaluate the gradient.
     ///
-    /// # Returns
+    /// # Return Value
     ///
-    /// Gradient vector with respect to the camera parameters.
+    /// Returns a `Result<DVector<f64>, argmin::core::Error>`. The `DVector<f64>` is the
+    /// computed gradient vector (6x1).
     ///
     /// # Errors
     ///
-    /// Returns an error if the camera model parameters are invalid or
-    /// if projection fails for the given points.
+    /// *   Returns `argmin::core::Error` if `DoubleSphereModel::new(p)` fails or if
+    ///     the Jacobian or residual computation fails.
+    /// *   If projection or Jacobian computation for a point fails, it's currently unwraped,
+    ///     which would panic. Robust implementation should handle this.
     fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
-        let mut grad = DVector::zeros(6);
-        // Ensure DoubleSphereModel::new is public or accessible
+        let mut grad = DVector::zeros(6); // 6 parameters
         let model = DoubleSphereModel::new(&p)?;
 
         for i in 0..self.points3d.ncols() {
             let p3d = &self.points3d.column(i).into_owned();
             let p2d_gt = &self.points2d.column(i).into_owned();
 
-            let (p2d_projected, jacobian_point_2x6) = model.project(p3d, true).unwrap();
-
-            if let Some(jacobian) = jacobian_point_2x6 {
-                let residual_2x1 = p2d_projected - p2d_gt;
-
-                // grad += J_i^T * r_i
-                grad += jacobian.transpose() * residual_2x1;
+            // Need both residuals and Jacobian for each point
+            match model.project(p3d, true) { // Request Jacobian
+                Ok((p2d_projected, Some(jacobian_point_2x6))) => {
+                    let residual_2x1 = p2d_projected - p2d_gt;
+                    // grad += J_i^T * r_i
+                    grad += jacobian_point_2x6.transpose() * residual_2x1;
+                }
+                Ok((_, None)) => {
+                    // Should not happen if Jacobian was requested and projection succeeded
+                    warn!("Jacobian missing for point {} in gradient calculation, skipping.", i);
+                }
+                Err(err) => {
+                    warn!("Projection failed for point {} in gradient calculation: {}, skipping.", i, err);
+                }
             }
         }
-        info!("Gradient: {}", grad);
+        if verbose_logging_enabled() { // Placeholder
+            info!("Gradient: {}", grad.transpose());
+        }
         Ok(grad)
     }
 }
 
-/// Implementation of the Hessian trait for optimization.
+/// Implements the [`Hessian`] trait from `argmin` for [`DoubleSphereOptimizationCost`].
+///
+/// The `Hessian` trait defines how to compute the Hessian matrix (or its approximation)
+/// of the cost function.
 impl Hessian for DoubleSphereOptimizationCost {
+    /// The parameter type is a `DVector<f64>` representing `[fx, fy, cx, cy, alpha, xi]`.
     type Param = DVector<f64>;
-    /// Hessian matrix approximation (J^T * J)
+    /// The Hessian type is a `DMatrix<f64>`.
     type Hessian = DMatrix<f64>;
 
-    /// Computes the Hessian matrix using the Gauss-Newton approximation.
+    /// Computes the Gauss-Newton approximation of the Hessian matrix.
     ///
-    /// The Hessian is approximated as J^T * J, where J is the Jacobian matrix.
-    /// This is a common approximation used in non-linear least squares optimization.
+    /// The Hessian `H` is approximated as `H = J^T * J`, where `J` is the Jacobian matrix
+    /// of the residuals. This approximation is commonly used in non-linear least squares
+    /// solvers like Gauss-Newton.
     ///
     /// # Arguments
     ///
-    /// * `p` - Parameter vector containing camera intrinsics and distortion parameters
+    /// * `p`: A `&DVector<f64>` containing the camera parameters at which to evaluate the Hessian.
     ///
-    /// # Returns
+    /// # Return Value
     ///
-    /// Approximate Hessian matrix (6×6).
+    /// Returns a `Result<DMatrix<f64>, argmin::core::Error>`. The `DMatrix<f64>` is the
+    /// approximated Hessian matrix (6x6).
     ///
     /// # Errors
     ///
-    /// Returns an error if the camera model parameters are invalid or
-    /// if projection fails for the given points.
+    /// *   Returns `argmin::core::Error` if `DoubleSphereModel::new(p)` fails or if
+    ///     the Jacobian computation fails.
+    /// *   If projection or Jacobian computation for a point fails, it's currently unwraped,
+    ///     which would panic. Robust implementation should handle this.
     fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
-        let mut jtj = DMatrix::zeros(6, 6);
-        // Ensure DoubleSphereModel::new is public or accessible
+        let mut jtj = DMatrix::zeros(6, 6); // Hessian approximation is 6x6
         let model = DoubleSphereModel::new(&p)?;
 
         for i in 0..self.points3d.ncols() {
             let p3d = &self.points3d.column(i).into_owned();
-            // We only need the Jacobian for J^T J
-            let (_, jacobian_point_2x6) = model.project(p3d, true).unwrap();
-
-            // Check if jacobian_point_2x6 is Some before using it
-            if let Some(jacobian) = jacobian_point_2x6 {
-                jtj += jacobian.transpose() * jacobian;
+            // We only need the Jacobian for J^T J for the Gauss-Newton approximation
+            match model.project(p3d, true) { // Request Jacobian
+                Ok((_, Some(jacobian_point_2x6))) => {
+                    // Accumulate J_i^T * J_i
+                    jtj += jacobian_point_2x6.transpose() * jacobian_point_2x6;
+                }
+                Ok((_, None)) => {
+                     warn!("Jacobian missing for point {} in Hessian calculation, skipping.", i);
+                }
+                Err(err) => {
+                    warn!("Projection failed for point {} in Hessian calculation: {}, skipping.", i, err);
+                }
             }
         }
-
-        info!("Hessian: {}", jtj);
+        if verbose_logging_enabled() { // Placeholder
+             info!("Hessian (J^T J): {}", jtj);
+        }
         Ok(jtj)
     }
 }
 
+/// Unit tests for [`DoubleSphereOptimizationCost`] and its trait implementations.
 #[cfg(test)]
 mod tests {
     use super::*;
     use crate::camera::{CameraModel, DoubleSphereModel as DSCameraModel, Intrinsics};
-    use approx::assert_relative_eq; // Added for assert_relative_eq!
+    use approx::assert_relative_eq;
     use nalgebra::{Matrix2xX, Matrix3xX};
 
-    // Helper to get a default model, similar to the one in samples/double_sphere.yaml
+    /// Helper function to create a sample [`DSCameraModel`] instance for testing.
+    /// This model is loaded from "samples/double_sphere.yaml".
     fn get_sample_camera_model() -> DSCameraModel {
         let path = "samples/double_sphere.yaml";
         DoubleSphereModel::load_from_yaml(path).unwrap()
     }
 
+    /// Helper function to generate sample 2D and 3D points for a given [`DSCameraModel`].
+    /// It uses `crate::geometry::sample_points`.
     fn sample_points_for_ds_model(
         model: &DSCameraModel,
         num_points: usize,
@@ -434,6 +820,9 @@
         crate::geometry::sample_points(Some(model), num_points).unwrap()
     }
 
+    /// Tests basic functionality of [`DoubleSphereOptimizationCost`] including `argmin` trait methods.
+    /// It checks if residuals, cost, Jacobian, gradient, and Hessian are computed correctly
+    /// when the model parameters perfectly match the data generation model.
     #[test]
     fn test_double_sphere_optimization_cost_basic() {
         let model_camera = get_sample_camera_model();
@@ -488,6 +877,9 @@
         assert_eq!(hess.ncols(), 6);
     }
 
+    /// Tests the `Optimizer::optimize` method for [`DoubleSphereOptimizationCost`].
+    /// It starts with a noisy model and checks if optimization brings parameters closer
+    /// to a reference model.
     #[test]
     fn test_double_sphere_optimize() {
         // Load a reference model from YAML file
@@ -510,56 +902,43 @@
             },
             resolution: reference_model.resolution.clone(),
             // Add noise to distortion parameters
-            xi: -0.1,
-            alpha: (reference_model.alpha * 0.95).max(0.1).min(0.99),
+            xi: -0.1, // Significantly different from reference_model.xi
+            alpha: (reference_model.alpha * 0.95).max(0.1).min(0.99), // Ensure alpha stays in (0,1]
         };
 
         info!("Reference model: {:?}", reference_model);
-        info!("Noisy model: {:?}", noisy_model);
+        info!("Noisy model (initial): {:?}", noisy_model);
 
         let mut optimization_task = DoubleSphereOptimizationCost::new(
-            noisy_model.clone(), // Pass the noisy model here
+            noisy_model.clone(), // Start optimization with the noisy model
             points_3d.clone(),
             points_2d.clone(),
         );
 
         // Optimize the model with noise
-        optimization_task.optimize(false).unwrap();
+        optimization_task.optimize(false).unwrap(); // verbose=false for less test output
 
-        info!("Optimized model: {:?}", optimization_task);
+        info!("Optimized model: {:?}", optimization_task.model);
 
         // Check that parameters have been optimized close to reference values
+        // Tolerances might need adjustment based on solver performance and data quality
+        assert_relative_eq!(optimization_task.model.intrinsics.fx, reference_model.intrinsics.fx, epsilon = 1.0);
+        assert_relative_eq!(optimization_task.model.intrinsics.fy, reference_model.intrinsics.fy, epsilon = 1.0);
+        assert_relative_eq!(optimization_task.model.intrinsics.cx, reference_model.intrinsics.cx, epsilon = 1.0);
+        assert_relative_eq!(optimization_task.model.intrinsics.cy, reference_model.intrinsics.cy, epsilon = 1.0);
+        assert_relative_eq!(optimization_task.model.alpha, reference_model.alpha, epsilon = 0.05);
+        assert_relative_eq!(optimization_task.model.xi, reference_model.xi, epsilon = 0.05);
+
+        // Verify that alpha is within bounds (0, 1] after optimization
         assert!(
-            (optimization_task.model.intrinsics.fx - reference_model.intrinsics.fx).abs() < 1.0,
-            "fx parameter didn't converge to expected value"
-        );
-        assert!(
-            (optimization_task.model.intrinsics.fy - reference_model.intrinsics.fy).abs() < 1.0,
-            "fy parameter didn't converge to expected value"
-        );
-        assert!(
-            (optimization_task.model.intrinsics.cx - reference_model.intrinsics.cx).abs() < 1.0,
-            "cx parameter didn't converge to expected value"
-        );
-        assert!(
-            (optimization_task.model.intrinsics.cy - reference_model.intrinsics.cy).abs() < 1.0,
-            "cy parameter didn't converge to expected value"
-        );
-        assert!(
-            (optimization_task.model.alpha - reference_model.alpha).abs() < 0.05,
-            "alpha parameter didn't converge to expected value"
-        );
-        assert!(
-            (optimization_task.model.xi - reference_model.xi).abs() < 0.05,
-            "xi parameter didn't converge to expected value"
-        );
-        // Verify that alpha is within bounds (0, 1]
-        assert!(
-            optimization_task.model.alpha > 0.0 && noisy_model.alpha <= 1.0,
-            "Alpha parameter out of valid range (0, 1]"
+            optimization_task.model.alpha > 0.0 && optimization_task.model.alpha <= 1.0,
+            "Alpha parameter ({}) out of valid range (0, 1] after optimization", optimization_task.model.alpha
         );
     }
 
+    /// Tests the `Optimizer::linear_estimation` method for [`DoubleSphereOptimizationCost`].
+    /// It checks if the linear estimation of `alpha` is reasonable and that `xi` remains
+    /// unchanged (as per the current implementation, which assumes `xi` is known or zero).
     #[test]
     fn test_double_sphere_linear_estimation_optimizer_trait() {
         let reference_model = get_sample_camera_model();
@@ -573,39 +952,26 @@
             "Need points for linear estimation test."
         );
 
+        // The model passed to new() for linear_estimation will have its `alpha` updated.
+        // Its `xi` and intrinsics are used as fixed values in the estimation equations.
         let mut optimization_task = DoubleSphereOptimizationCost::new(
-            reference_model.clone(), // Pass the noisy model here
+            reference_model.clone(),
             points_3d.clone(),
             points_2d.clone(),
         );
 
         optimization_task.linear_estimation().unwrap();
 
-        // Linear estimation for DoubleSphere typically estimates alpha, xi is often set to 0 or a fixed value.
-        // The provided implementation estimates alpha with xi=0.
-        // So, we compare alpha and check if xi is close to 0.
+        // Linear estimation for DoubleSphere in this impl estimates alpha.
+        // The reference_model.xi is used in the equations if it was non-zero.
+        // The current linear_estimation does not explicitly set xi to 0, it only updates alpha.
+        // So, xi should remain as it was in reference_model.
         assert_relative_eq!(
             optimization_task.model.alpha,
             reference_model.alpha,
-            epsilon = 1.0
-        ); // Linear estimation is an approximation
-        assert_relative_eq!(
+            epsilon = 1.0 // Linear estimation is an approximation, allow larger tolerance
+        );
+        assert_relative_eq!( // xi should not be changed by this linear_estimation
             optimization_task.model.xi,
             reference_model.xi,
             epsilon = 1e-9
-        ); // Expect xi to be zero from linear_estimation impl
-
-        // Intrinsics should remain the same as input
-        assert_relative_eq!(
-            optimization_task.model.intrinsics.fx,
-            reference_model.intrinsics.fx,
-            epsilon = 1e-9
         );
-        assert_relative_eq!(
-            optimization_task.model.intrinsics.fy,
-            reference_model.intrinsics.fy,
-            epsilon = 1e-9
-        );
-        assert_relative_eq!(
-            optimization_task.model.intrinsics.cx,
-            reference_model.intrinsics.cx,
-            epsilon = 1e-9
-        );
-        assert_relative_eq!(
-            optimization_task.model.intrinsics.cy,
-            reference_model.intrinsics.cy,
-            epsilon = 1e-9
-        );
+
+        // Intrinsics should remain the same as input (they are treated as known)
+        assert_relative_eq!(optimization_task.model.intrinsics.fx, reference_model.intrinsics.fx, epsilon = 1e-9);
+        assert_relative_eq!(optimization_task.model.intrinsics.fy, reference_model.intrinsics.fy, epsilon = 1e-9);
+        assert_relative_eq!(optimization_task.model.intrinsics.cx, reference_model.intrinsics.cx, epsilon = 1e-9);
+        assert_relative_eq!(optimization_task.model.intrinsics.cy, reference_model.intrinsics.cy, epsilon = 1e-9);
     }
 }
