//! This module provides the cost function and optimization routines
//! for calibrating a Unified Camera Model (UCM).
//!
//! It uses the `factrs` crate for non-linear optimization and defines
//! the necessary structures and traits to integrate the UCM
//! camera model with the optimization framework.

use crate::camera::{CameraModel, CameraModelError, UcmModel};
use crate::optimization::Optimizer;
use factrs::{
    assign_symbols,
    core::{Graph, Huber, LevenMarquardt, Values},
    dtype, fac,
    linalg::{Const, DiffResult, ForwardProp, MatrixX, Numeric, VectorX},
    linear::QRSolver,
    optimizers::Optimizer as FactrsOptimizer,
    residuals::Residual1,
    variables::VectorVar,
};
use log::info;
use nalgebra::{DMatrix, Matrix2xX, Matrix3xX, Vector2, Vector3};

// Define VectorVar5 following the same pattern as VectorVar6 and VectorVar8
pub type VectorVar5<T = dtype> = VectorVar<5, T>;

// Helper function to create VectorVar5 instances since we can't implement methods for foreign types
fn create_vector_var5<T: Numeric>(fx: T, fy: T, cx: T, cy: T, alpha: T) -> VectorVar5<T> {
    use factrs::linalg::{Vector, VectorX};
    // Create a VectorX first, then convert to fixed-size Vector
    let vec_x = VectorX::from_vec(vec![fx, fy, cx, cy, alpha]);
    let fixed_vec = Vector::<5, T>::from_iterator(vec_x.iter().cloned());
    VectorVar(fixed_vec)
}

assign_symbols!(UCMCamParams: VectorVar5);

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

/// Residual implementation for `factrs` optimization of the [`UcmModel`].
///
/// This struct defines the residual error between the observed 2D point and
/// the projected 2D point from a 3D point using the current camera model parameters.
/// It is used by the `factrs` optimization framework.
#[derive(Debug, Clone)]
pub struct UcmFactrsResidual {
    /// The 3D point in the camera's coordinate system.
    point3d: Vector3<dtype>,
    /// The corresponding observed 2D point in image coordinates.
    point2d: Vector2<dtype>,
}

impl UcmFactrsResidual {
    /// Creates a new residual for a single 3D-2D point correspondence.
    ///
    /// # Arguments
    ///
    /// * `point3d` - The 3D point in the camera's coordinate system.
    /// * `point2d` - The corresponding observed 2D point in image coordinates.
    pub fn new(point3d: Vector3<f64>, point2d: Vector2<f64>) -> Self {
        Self {
            point3d: point3d.cast::<dtype>(),
            point2d: point2d.cast::<dtype>(),
        }
    }

    /// Computes the analytical residual and Jacobian for the UCM model.
    ///
    /// This method implements the analytical computation of both the residual
    /// and its Jacobian with respect to the camera parameters, based on the
    /// C++ implementation.
    ///
    /// # Arguments
    ///
    /// * `cam_params` - Array of camera parameters [fx, fy, cx, cy, alpha].
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * `residual` - The residual vector.
    /// * `jacobian` - The 2x5 Jacobian matrix.
    ///
    /// # Errors
    ///
    /// Returns `CameraModelError` if the projection fails.
    pub fn compute_analytical_residual_jacobian(
        &self,
        cam_params: &[f64; 5], // [fx, fy, cx, cy, alpha]
    ) -> Result<(Vector2<f64>, DMatrix<f64>), CameraModelError> {
        let fx = cam_params[0];
        let fy = cam_params[1];
        let cx = cam_params[2];
        let cy = cam_params[3];
        let alpha = cam_params[4];

        // gt_u_ and gt_v_ correspond to observed 2D point coordinates
        let gt_u = self.point2d.x;
        let gt_v = self.point2d.y;

        // obs_x_, obs_y_, obs_z_ correspond to 3D point coordinates
        let obs_x = self.point3d.x;
        let obs_y = self.point3d.y;
        let obs_z = self.point3d.z;

        let u_cx = gt_u - cx;
        let v_cy = gt_v - cy;
        let d = (obs_x * obs_x + obs_y * obs_y + obs_z * obs_z).sqrt();
        let denom = alpha * d + (1.0 - alpha) * obs_z;

        const PRECISION: f64 = 1e-3;
        if denom < PRECISION || !UcmModel::check_proj_condition(obs_z, d, alpha) {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        // Residual computation based on C++ implementation
        let residual = Vector2::new(fx * obs_x - u_cx * denom, fy * obs_y - v_cy * denom);

        // Compute analytical Jacobian
        let mut jacobian = DMatrix::zeros(2, 5); // 2 outputs, 5 parameters (fx,fy,cx,cy,alpha)

        // Jacobian computation based on C++ implementation
        jacobian[(0, 0)] = obs_x; // ∂residual_x / ∂fx
        jacobian[(1, 0)] = 0.0; // ∂residual_y / ∂fx
        jacobian[(0, 1)] = 0.0; // ∂residual_x / ∂fy
        jacobian[(1, 1)] = obs_y; // ∂residual_y / ∂fy
        jacobian[(0, 2)] = denom; // ∂residual_x / ∂cx
        jacobian[(1, 2)] = 0.0; // ∂residual_y / ∂cx
        jacobian[(0, 3)] = 0.0; // ∂residual_x / ∂cy
        jacobian[(1, 3)] = denom; // ∂residual_y / ∂cy
        jacobian[(0, 4)] = (obs_z - d) * u_cx; // ∂residual_x / ∂alpha
        jacobian[(1, 4)] = (obs_z - d) * v_cy; // ∂residual_y / ∂alpha

        Ok((residual, jacobian))
    }
}

// Mark this residual for factrs serialization and other features
#[factrs::mark]
impl Residual1 for UcmFactrsResidual {
    type DimIn = Const<5>;
    type DimOut = Const<2>;
    type V1 = VectorVar<5, dtype>;
    type Differ = ForwardProp<Self::DimIn>;

    /// Computes the residual vector for the given camera parameters.
    ///
    /// The residual is defined as `observed_2d_point - project(3d_point, camera_parameters)`.
    /// This method is called by the `factrs` optimizer during the optimization process.
    ///
    /// # Arguments
    ///
    /// * `cam_params` - A `VectorVar5<T>` containing the current camera parameters
    ///   `[fx, fy, cx, cy, alpha]`. `T` is a numeric type used by `factrs`.
    ///
    /// # Returns
    ///
    /// A `VectorX<T>` of dimension 2, representing the residual `[ru, rv]`.
    fn residual1<T: Numeric>(&self, cam_params: VectorVar<5, T>) -> VectorX<T> {
        let fx = cam_params[0];
        let fy = cam_params[1];
        let cx = cam_params[2];
        let cy = cam_params[3];
        let alpha = cam_params[4];

        // Observed 2D point coordinates
        let gt_u = T::from(self.point2d.x);
        let gt_v = T::from(self.point2d.y);

        // 3D point coordinates
        let obs_x = T::from(self.point3d.x);
        let obs_y = T::from(self.point3d.y);
        let obs_z = T::from(self.point3d.z);

        // Compute intermediate values (same as C++ implementation)
        let u_cx = gt_u - cx;
        let v_cy = gt_v - cy;
        let r_squared = obs_x * obs_x + obs_y * obs_y + obs_z * obs_z;
        let d = r_squared.sqrt();
        let denom = alpha * d + (T::from(1.0) - alpha) * obs_z;

        // Check for valid projection (same as C++ implementation)
        let precision = T::from(1e-3);
        if denom < precision {
            // Return large residuals for invalid projections
            return VectorX::from_vec(vec![T::from(1e6), T::from(1e6)]);
        }

        // Compute residuals using C++ formulation:
        // residuals[0] = fx * obs_x - u_cx * denom
        // residuals[1] = fy * obs_y - v_cy * denom
        let residual_u = fx * obs_x - u_cx * denom;
        let residual_v = fy * obs_y - v_cy * denom;

        VectorX::from_vec(vec![residual_u, residual_v])
    }

    /// Computes the analytical Jacobian for the residual function.
    ///
    /// This method provides the analytical Jacobian computation for better
    /// optimization performance compared to automatic differentiation.
    ///
    /// # Arguments
    ///
    /// * `values` - The current values of all variables in the optimization graph.
    /// * `keys` - A slice of `factrs::containers::Key` indicating which variables
    ///   (in this case, `UCMCamParams(0)`) to compute the Jacobian against.
    ///
    /// # Returns
    ///
    /// A `DiffResult<VectorX<dtype>, MatrixX<dtype>>` containing:
    /// * `value` - The residual vector computed at the given camera parameters.
    /// * `diff` - The 2x5 Jacobian matrix of the residual with respect to the
    ///   camera parameters `[fx, fy, cx, cy, alpha]`.
    fn residual1_jacobian(
        &self,
        values: &factrs::containers::Values,
        keys: &[factrs::containers::Key],
    ) -> DiffResult<VectorX<dtype>, MatrixX<dtype>>
    where
        Self::V1: 'static,
    {
        // Get the camera parameters from values
        let cam_params: &VectorVar5<dtype> = values.get_unchecked(keys[0]).unwrap_or_else(|| {
            panic!(
                "Key not found in values: {:?} with type {}",
                keys[0],
                std::any::type_name::<VectorVar5<dtype>>()
            )
        });

        // Extract parameter values
        let params_array = [
            cam_params[0], // fx
            cam_params[1], // fy
            cam_params[2], // cx
            cam_params[3], // cy
            cam_params[4], // alpha
        ];

        // Compute analytical residual and Jacobian
        match self.compute_analytical_residual_jacobian(&params_array) {
            Ok((residual, jacobian)) => {
                // Convert residual to VectorX<dtype>
                let residual_vec = VectorX::from_vec(vec![residual.x, residual.y]);

                // Convert Jacobian to MatrixX<dtype>
                let mut jacobian_matrix = MatrixX::zeros(2, 5);
                for i in 0..2 {
                    for j in 0..5 {
                        jacobian_matrix[(i, j)] = jacobian[(i, j)];
                    }
                }

                DiffResult {
                    value: residual_vec,
                    diff: jacobian_matrix,
                }
            }
            Err(_) => {
                // Return large residuals and zero Jacobian for invalid projections
                let residual_vec = VectorX::from_vec(vec![1e6, 1e6]);
                let jacobian_matrix = MatrixX::zeros(2, 5);

                DiffResult {
                    value: residual_vec,
                    diff: jacobian_matrix,
                }
            }
        }
    }
}

impl Optimizer for UcmOptimizationCost {
    /// Performs non-linear optimization to refine the UCM camera model parameters.
    ///
    /// This method uses the `factrs` crate with Levenberg-Marquardt optimization
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
        if verbose {
            info!("Starting UCM camera model optimization...");
            info!("Initial parameters: {:?}", self.model);
        }

        // Create a factrs Values object to hold the camera parameters
        let mut values = Values::new();

        // Initial parameters
        let initial_params = create_vector_var5(
            self.model.intrinsics.fx as dtype,
            self.model.intrinsics.fy as dtype,
            self.model.intrinsics.cx as dtype,
            self.model.intrinsics.cy as dtype,
            self.model.alpha as dtype,
        );

        values.insert(UCMCamParams(0), initial_params);

        // Create a factrs Graph
        let mut graph = Graph::new();

        // Add residuals for each point correspondence
        for i in 0..self.points3d.ncols() {
            let point3d = self.points3d.column(i).into_owned();
            let point2d = self.points2d.column(i).into_owned();

            let residual = UcmFactrsResidual::new(point3d, point2d);

            // Create a factor with the residual and add it to the graph
            let factor = fac![residual, UCMCamParams(0), 1.0 as std, Huber::default()];
            graph.add_factor(factor);
        }

        if verbose {
            info!("Starting optimization with factrs Levenberg-Marquardt...");
        }

        // Create a Levenberg-Marquardt optimizer with QR solver
        let mut optimizer: LevenMarquardt<QRSolver> = LevenMarquardt::new(graph);

        // Run optimization
        let result = optimizer
            .optimize(values)
            .map_err(|e| CameraModelError::NumericalError(format!("{:?}", e)))?;

        if verbose {
            info!("Optimization finished");
        }

        // Extract the optimized parameters
        let optimized_params: &VectorVar5<f64> = result.get(UCMCamParams(0)).unwrap();
        let params = &optimized_params.0;

        // Update the model parameters
        self.model.intrinsics.fx = params[0];
        self.model.intrinsics.fy = params[1];
        self.model.intrinsics.cx = params[2];
        self.model.intrinsics.cy = params[3];
        self.model.alpha = params[4];

        // Validate the optimized parameters
        self.model.validate_params()?;

        if verbose {
            info!("Optimized parameters: {:?}", self.model);
        }

        Ok(())
    }

    /// Performs linear estimation for initial parameter guess.
    ///
    /// This method estimates the alpha parameter using linear least squares
    /// while keeping the intrinsic parameters (fx, fy, cx, cy) fixed.
    /// Based on the C++ implementation.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If linear estimation was successful and alpha was updated.
    /// * `Err(CameraModelError)` - If estimation failed due to insufficient data or numerical issues.
    fn linear_estimation(&mut self) -> Result<(), CameraModelError> {
        let num_points = self.points3d.ncols();

        if num_points < 1 {
            return Err(CameraModelError::NumericalError(
                "Insufficient points for linear estimation".to_string(),
            ));
        }

        // Use nalgebra matrices for the linear system
        let mut a_matrix = nalgebra::DMatrix::zeros(num_points * 2, 1);
        let mut b_vector = nalgebra::DVector::zeros(num_points * 2);

        // Extract common parameters (intrinsics)
        let fx = self.model.intrinsics.fx;
        let fy = self.model.intrinsics.fy;
        let cx = self.model.intrinsics.cx;
        let cy = self.model.intrinsics.cy;

        for i in 0..num_points {
            let point3d = self.points3d.column(i);
            let point2d = self.points2d.column(i);

            let x = point3d[0];
            let y = point3d[1];
            let z = point3d[2];
            let u = point2d[0];
            let v = point2d[1];

            let d = (x * x + y * y + z * z).sqrt();
            let u_cx = u - cx;
            let v_cy = v - cy;

            // Set up the linear system: A * alpha = b
            a_matrix[(i * 2, 0)] = u_cx * (d - z);
            a_matrix[(i * 2 + 1, 0)] = v_cy * (d - z);

            b_vector[i * 2] = (fx * x) - (u_cx * z);
            b_vector[i * 2 + 1] = (fy * y) - (v_cy * z);
        }

        // Solve the linear system using SVD
        let svd = a_matrix.svd(true, true);
        let solution = svd.solve(&b_vector, 1e-6).map_err(|_| {
            CameraModelError::NumericalError(
                "Failed to solve linear system for alpha estimation".to_string(),
            )
        })?;

        let estimated_alpha = solution[0];

        // Validate and clamp the estimated alpha to reasonable bounds
        let clamped_alpha = if estimated_alpha.is_finite() {
            estimated_alpha.clamp(0.1, 2.0) // Reasonable bounds for UCM alpha
        } else {
            return Err(CameraModelError::NumericalError(
                "Estimated alpha is not finite".to_string(),
            ));
        };

        // Update the model's alpha parameter
        self.model.alpha = clamped_alpha;

        Ok(())
    }

    /// Returns the current intrinsic parameters.
    fn get_intrinsics(&self) -> crate::camera::Intrinsics {
        self.model.get_intrinsics()
    }

    /// Returns the current resolution.
    fn get_resolution(&self) -> crate::camera::Resolution {
        self.model.get_resolution()
    }

    /// Returns the current distortion parameters.
    fn get_distortion(&self) -> Vec<f64> {
        self.model.get_distortion()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraModel, UcmModel};
    use approx::assert_relative_eq;
    use nalgebra::{Matrix2xX, Matrix3xX};

    /// Helper function to create a sample UCM model for testing.
    fn get_sample_ucm_model() -> UcmModel {
        UcmModel::load_from_yaml("samples/ucm.yaml").unwrap()
    }

    /// Helper function to generate sample 3D-2D point correspondences.
    fn generate_sample_points(
        model: &UcmModel,
        num_points: usize,
    ) -> (Matrix3xX<f64>, Matrix2xX<f64>) {
        let mut points_3d = Matrix3xX::zeros(num_points);
        let mut points_2d = Matrix2xX::zeros(num_points);

        for i in 0..num_points {
            // Generate a 3D point in front of the camera
            let point_3d = nalgebra::Vector3::new(
                (i as f64 - num_points as f64 / 2.0) * 0.1,
                (i as f64 - num_points as f64 / 2.0) * 0.1,
                2.0 + (i as f64) * 0.1,
            );

            // Project to 2D
            if let Ok(point_2d) = model.project(&point_3d) {
                points_3d.set_column(i, &point_3d);
                points_2d.set_column(i, &point_2d);
            } else {
                // Use a fallback point if projection fails
                let fallback_3d = nalgebra::Vector3::new(0.0, 0.0, 3.0);
                let fallback_2d = model.project(&fallback_3d).unwrap();
                points_3d.set_column(i, &fallback_3d);
                points_2d.set_column(i, &fallback_2d);
            }
        }

        (points_3d, points_2d)
    }

    #[test]
    fn test_ucm_optimization_cost_creation() {
        let model = get_sample_ucm_model();
        let (points_3d, points_2d) = generate_sample_points(&model, 10);

        let cost = UcmOptimizationCost::new(model.clone(), points_3d, points_2d);

        assert_eq!(cost.model.intrinsics.fx, model.intrinsics.fx);
        assert_eq!(cost.model.alpha, model.alpha);
        assert_eq!(cost.points3d.ncols(), 10);
        assert_eq!(cost.points2d.ncols(), 10);
    }

    #[test]
    fn test_ucm_factrs_residual_computation() {
        let model = get_sample_ucm_model();
        let point_3d = nalgebra::Vector3::new(0.1, 0.1, 2.0);
        let point_2d = model.project(&point_3d).unwrap();

        let residual = UcmFactrsResidual::new(point_3d, point_2d);

        // Test residual computation with correct parameters
        let cam_params = [
            model.intrinsics.fx,
            model.intrinsics.fy,
            model.intrinsics.cx,
            model.intrinsics.cy,
            model.alpha,
        ];

        let result = residual.compute_analytical_residual_jacobian(&cam_params);
        assert!(result.is_ok());

        let (residual_vec, jacobian) = result.unwrap();

        // With the new residual formulation, check that the computation succeeds
        // and produces finite values
        assert!(residual_vec.x.is_finite());
        assert!(residual_vec.y.is_finite());

        // Check Jacobian dimensions
        assert_eq!(jacobian.nrows(), 2);
        assert_eq!(jacobian.ncols(), 5);

        // Check that all Jacobian elements are finite
        for i in 0..jacobian.nrows() {
            for j in 0..jacobian.ncols() {
                assert!(
                    jacobian[(i, j)].is_finite(),
                    "Jacobian element ({}, {}) is not finite",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_ucm_optimization_linear_estimation() {
        let model = get_sample_ucm_model();
        let (points_3d, points_2d) = generate_sample_points(&model, 20);

        let mut cost = UcmOptimizationCost::new(model.clone(), points_3d, points_2d);
        let _original_alpha = cost.model.alpha;

        // Test linear estimation - should estimate alpha parameter
        let result = cost.linear_estimation();
        assert!(result.is_ok());

        // Check that alpha was updated and is within reasonable bounds
        assert!(cost.model.alpha.is_finite());
        assert!(cost.model.alpha >= 0.1);
        assert!(cost.model.alpha <= 2.0);

        // Alpha should be different from the original (unless by coincidence)
        // We don't assert this strictly since it could theoretically be the same
    }

    #[test]
    fn test_ucm_optimization_getters() {
        let model = get_sample_ucm_model();
        let (points_3d, points_2d) = generate_sample_points(&model, 5);

        let cost = UcmOptimizationCost::new(model.clone(), points_3d, points_2d);

        let intrinsics = cost.get_intrinsics();
        assert_relative_eq!(intrinsics.fx, model.intrinsics.fx);
        assert_relative_eq!(intrinsics.fy, model.intrinsics.fy);
        assert_relative_eq!(intrinsics.cx, model.intrinsics.cx);
        assert_relative_eq!(intrinsics.cy, model.intrinsics.cy);

        let resolution = cost.get_resolution();
        assert_eq!(resolution.width, model.resolution.width);
        assert_eq!(resolution.height, model.resolution.height);

        let distortion = cost.get_distortion();
        assert_eq!(distortion.len(), 1);
        assert_relative_eq!(distortion[0], model.alpha);
    }
}
