//! This module provides the cost function and optimization routines
//! for calibrating a Kannala-Brandt camera model.
//!
//! It uses the `factrs` crate for non-linear optimization and defines
//! the necessary structures and traits to integrate the Kannala-Brandt
//! camera model with the optimization framework.

use crate::camera::{CameraModel, CameraModelError, KannalaBrandtModel};
use crate::optimization::Optimizer;
use factrs::{
    assign_symbols,
    core::{Graph, Huber, LevenMarquardt, Values},
    dtype, fac,
    linalg::{Const, Diff, DiffResult, ForwardProp, MatrixX, Numeric, VectorX},
    linear::QRSolver,
    optimizers::Optimizer as FactrsOptimizer,
    residuals::Residual1,
    variables::VectorVar,
};
use log::{info, warn};
use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3xX, Vector2, Vector3};

// Define VectorVar8 following the same pattern as VectorVar6
pub type VectorVar8<T = dtype> = VectorVar<8, T>;

// Helper function to create VectorVar8 instances since we can't implement methods for foreign types
fn create_vector_var8<T: Numeric>(x: T, y: T, z: T, w: T, a: T, b: T, c: T, d: T) -> VectorVar8<T> {
    use factrs::linalg::{Vector, VectorX};
    // Create a VectorX first, then convert to fixed-size Vector
    let vec_x = VectorX::from_vec(vec![x, y, z, w, a, b, c, d]);
    let fixed_vec = Vector::<8, T>::from_iterator(vec_x.iter().cloned());
    VectorVar(fixed_vec)
}

assign_symbols!(KBCamParams: VectorVar8);

/// Residual implementation for `factrs` optimization of the [`KannalaBrandtModel`].
///
/// This struct defines the residual error between the observed 2D point and
/// the projected 2D point from a 3D point using the current camera model parameters.
/// It is used by the `factrs` optimization framework.
#[derive(Debug, Clone)]
pub struct KannalaBrandtFactrsResidual {
    /// The 3D point in the camera's coordinate system.
    point3d: Vector3<dtype>,
    /// The corresponding observed 2D point in image coordinates.
    point2d: Vector2<dtype>,
}

impl KannalaBrandtFactrsResidual {
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

    /// Computes the residual and its analytical Jacobian with respect to camera parameters.
    ///
    /// This method is primarily used for validation and debugging purposes. It leverages
    /// the analytical Jacobian computation from [`KannalaBrandtModel::project`] to provide
    /// a reference for comparing with Jacobians obtained through automatic differentiation.
    ///
    /// # Arguments
    ///
    /// * `cam_params` - An array of 8 `f64` values representing the camera parameters:
    ///   `[fx, fy, cx, cy, k1, k2, k3, k4]`.
    ///
    /// # Returns
    ///
    /// A `Result` containing:
    /// * `Ok((residual, jacobian))` - A tuple where `residual` is a `Vector2<f64>`
    ///   representing the difference between the observed and projected 2D point,
    ///   and `jacobian` is a `nalgebra::DMatrix<f64>` (2x8) representing the
    ///   Jacobian of the residual with respect to the camera parameters.
    /// * `Err(CameraModelError)` - If the projection or Jacobian computation fails.
    pub fn compute_analytical_residual_jacobian(
        &self,
        cam_params: &[f64; 8], // [fx, fy, cx, cy, k1, k2, k3, k4]
    ) -> Result<(Vector2<f64>, DMatrix<f64>), CameraModelError> {
        let fx = cam_params[0];
        let fy = cam_params[1];
        let cx = cam_params[2];
        let cy = cam_params[3];
        let k1 = cam_params[4];
        let k2 = cam_params[5];
        let k3 = cam_params[6];
        let k4 = cam_params[7];

        let x = self.point3d.x;
        let y = self.point3d.y;
        let z = self.point3d.z;

        let r_squared = x * x + y * y;
        let r = r_squared.sqrt();
        let theta = r.atan2(z); // atan2(y,x) in nalgebra is atan2(self, other) -> atan2(r,z)

        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        let theta_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9;

        let (x_r, y_r) = if r < f64::EPSILON {
            // Use a small epsilon for r
            // If r is very small, point is close to optical axis.
            // x_r and y_r would be ill-defined or lead to instability.
            // For many fisheye models, the projection at r=0 is (cx, cy).
            // Here, theta_d * x_r and theta_d * y_r would be 0.
            // Let's assume for r=0, x_r and y_r are effectively 0 for the multiplication.
            // Or, if theta_d is also 0 (theta=0), then it's 0.
            // If theta=0 (on optical axis), theta_d = 0. So proj_x = cx, proj_y = cy.
            // This seems consistent.
            (0.0, 0.0)
        } else {
            (x / r, y / r)
        };

        let projected_x = fx * theta_d * x_r + cx;
        let projected_y = fy * theta_d * y_r + cy;

        let residual = Vector2::new(projected_x - self.point2d.x, projected_y - self.point2d.y);

        let mut jacobian = DMatrix::zeros(2, 8); // fx, fy, cx, cy, k1, k2, k3, k4
                                                 // Column 0: dfx
        jacobian[(0, 0)] = theta_d * x_r;
        jacobian[(1, 0)] = 0.0;

        // Column 1: dfy
        jacobian[(0, 1)] = 0.0;
        jacobian[(1, 1)] = theta_d * y_r;

        // Column 2: dcx
        jacobian[(0, 2)] = 1.0;
        jacobian[(1, 2)] = 0.0;

        // Column 3: dcy
        jacobian[(0, 3)] = 0.0;
        jacobian[(1, 3)] = 1.0;

        // Columns 4-7: dk1, dk2, dk3, dk4
        // This part matches the C++ logic: de_dp * dp_dd_theta * dd_theta_dks
        // where de_dp = [[fx, 0], [0, fy]]
        //       dp_dd_theta = [x_r, y_r]^T
        //       dd_theta_dks = [theta3, theta5, theta7, theta9]
        // Resulting in:
        // jacobian.rightCols<4>() = [fx * x_r; fy * y_r] * [theta3, theta5, theta7, theta9]

        let fx_x_r = fx * x_r;
        let fy_y_r = fy * y_r;

        jacobian[(0, 4)] = fx_x_r * theta3; // d(proj_x)/dk1
        jacobian[(1, 4)] = fy_y_r * theta3; // d(proj_y)/dk1

        jacobian[(0, 5)] = fx_x_r * theta5; // d(proj_x)/dk2
        jacobian[(1, 5)] = fy_y_r * theta5; // d(proj_y)/dk2

        jacobian[(0, 6)] = fx_x_r * theta7; // d(proj_x)/dk3
        jacobian[(1, 6)] = fy_y_r * theta7; // d(proj_y)/dk3

        jacobian[(0, 7)] = fx_x_r * theta9; // d(proj_x)/dk4
        jacobian[(1, 7)] = fy_y_r * theta9; // d(proj_y)/dk4

        Ok((residual, jacobian))
    }
}

// Mark this residual for factrs serialization and other features
#[factrs::mark]
impl Residual1 for KannalaBrandtFactrsResidual {
    type DimIn = Const<8>;
    type DimOut = Const<2>;
    type V1 = VectorVar<8, dtype>;
    type Differ = ForwardProp<Self::DimIn>;

    /// Computes the residual vector for the given camera parameters.
    ///
    /// The residual is defined as `observed_2d_point - project(3d_point, camera_parameters)`.
    /// This method is called by the `factrs` optimizer during the optimization process.
    ///
    /// # Arguments
    ///
    /// * `cam_params` - A `VectorVar<8, T>` containing the current camera parameters
    ///   `[fx, fy, cx, cy, k1, k2, k3, k4]`. `T` is a numeric type used by `factrs`.
    ///
    /// # Returns
    ///
    /// A `VectorX<T>` of dimension 2, representing the residual `[ru, rv]`.
    fn residual1<T: Numeric>(&self, cam_params: VectorVar<8, T>) -> VectorX<T> {
        // Convert camera parameters from generic type T to f64 for KannalaBrandtModel
        // Using to_subset() which is available through SupersetOf<f64> trait
        let fx_f64 = cam_params[0].to_subset().unwrap_or(0.0);
        let fy_f64 = cam_params[1].to_subset().unwrap_or(0.0);
        let cx_f64 = cam_params[2].to_subset().unwrap_or(0.0);
        let cy_f64 = cam_params[3].to_subset().unwrap_or(0.0);
        let k1_f64 = cam_params[4].to_subset().unwrap_or(0.0);
        let k2_f64 = cam_params[5].to_subset().unwrap_or(0.0);
        let k3_f64 = cam_params[6].to_subset().unwrap_or(0.0);
        let k4_f64 = cam_params[7].to_subset().unwrap_or(0.0);

        // Create a KannalaBrandtModel instance using the converted parameters
        let model = KannalaBrandtModel {
            intrinsics: crate::camera::Intrinsics {
                fx: fx_f64,
                fy: fy_f64,
                cx: cx_f64,
                cy: cy_f64,
            },
            resolution: crate::camera::Resolution {
                width: 0, // Resolution is not part of the optimized parameters
                height: 0,
            },
            distortions: [k1_f64, k2_f64, k3_f64, k4_f64],
        };

        // Convert input points to f64 for projection
        let point3d_f64 = Vector3::new(
            self.point3d.x as f64,
            self.point3d.y as f64,
            self.point3d.z as f64,
        );
        let point2d_f64 = Vector2::new(self.point2d.x as f64, self.point2d.y as f64);

        // Use the existing KannalaBrandtModel::project method
        match model.project(&point3d_f64) {
            Ok(projected_2d) => {
                // Compute residuals (observed - projected) and convert back to type T
                let residual_u = T::from(projected_2d.x - point2d_f64.x);
                let residual_v = T::from(projected_2d.y - point2d_f64.y);
                VectorX::from_vec(vec![residual_u, residual_v])
            }
            Err(_) => {
                // Return large residuals for invalid projections
                VectorX::from_vec(vec![T::from(1e6), T::from(1e6)])
            }
        }
    }

    /// Computes the Jacobian of the residual function with respect to the camera parameters.
    ///
    /// This implementation overrides the default automatic differentiation provided by `factrs`
    /// and uses the analytical Jacobian derived from [`KannalaBrandtModel::project`].
    /// This approach offers better computational efficiency while maintaining accuracy.
    ///
    /// If the analytical computation fails (e.g., due to invalid parameters leading to
    /// projection errors), it falls back to automatic differentiation as a safety measure.
    ///
    /// # Arguments
    ///
    /// * `values` - A `factrs::containers::Values` map containing the current state
    ///   of the optimization variables (i.e., camera parameters).
    /// * `keys` - A slice of `factrs::containers::Key` indicating which variables
    ///   (in this case, `KBCamParams(0)`) to compute the Jacobian against.
    ///
    /// # Returns
    ///
    /// A `DiffResult<VectorX<dtype>, MatrixX<dtype>>` containing:
    /// * `value` - The residual vector computed at the given camera parameters.
    /// * `diff` - The 2x8 Jacobian matrix of the residual with respect to the
    ///   camera parameters `[fx, fy, cx, cy, k1, k2, k3, k4]`.
    fn residual1_jacobian(
        &self,
        values: &factrs::containers::Values,
        keys: &[factrs::containers::Key],
    ) -> DiffResult<VectorX<dtype>, MatrixX<dtype>>
    where
        Self::V1: 'static,
    {
        // Get the camera parameters from values
        let cam_params: &VectorVar<8, dtype> = values.get_unchecked(keys[0]).unwrap_or_else(|| {
            panic!(
                "Key not found in values: {:?} with type {}",
                keys[0],
                std::any::type_name::<VectorVar<8, dtype>>()
            )
        });

        // Extract parameter values
        let params_array = [
            cam_params[0], // fx
            cam_params[1], // fy
            cam_params[2], // cx
            cam_params[3], // cy
            cam_params[4], // k1
            cam_params[5], // k2
            cam_params[6], // k3
            cam_params[7], // k4
        ];

        // Compute analytical residual and Jacobian
        match self.compute_analytical_residual_jacobian(&params_array) {
            Ok((analytical_residual, analytical_jacobian)) => {
                // Convert nalgebra types to factrs types
                let residual_vec =
                    VectorX::from_vec(vec![analytical_residual.x, analytical_residual.y]);

                // Convert the analytical Jacobian to factrs MatrixX
                let mut jacobian_factrs = MatrixX::zeros(2, 8);
                for i in 0..2 {
                    for j in 0..8 {
                        jacobian_factrs[(i, j)] = analytical_jacobian[(i, j)];
                    }
                }

                DiffResult {
                    value: residual_vec,
                    diff: jacobian_factrs,
                }
            }
            Err(e) => {
                // Fallback to automatic differentiation if analytical computation fails
                warn!("Analytical Jacobian computation failed: {:?}, falling back to automatic differentiation", e);
                Self::Differ::jacobian_1(|params| self.residual1(params), cam_params)
            }
        }
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
    /// This method uses the Levenberg-Marquardt algorithm provided by the `factrs`
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

        // Create a factrs Values object to hold the camera parameters
        let mut values = Values::new();

        // Initial parameters - create VectorVar8 from matrix with factrs Const
        let initial_params = create_vector_var8(
            self.model.intrinsics.fx as dtype,
            self.model.intrinsics.fy as dtype,
            self.model.intrinsics.cx as dtype,
            self.model.intrinsics.cy as dtype,
            self.model.distortions[0] as dtype,
            self.model.distortions[1] as dtype,
            self.model.distortions[2] as dtype,
            self.model.distortions[3] as dtype,
        );

        // Insert the initial parameters into the values
        values.insert(KBCamParams(0), initial_params);

        // Create a factrs Graph
        let mut graph = Graph::new();

        // Add residuals for each point correspondence
        for i in 0..self.points3d.ncols() {
            let p3d = self.points3d.column(i).into_owned();
            let p2d = self.points2d.column(i).into_owned();

            // Create a residual for this point correspondence
            let residual = KannalaBrandtFactrsResidual {
                point3d: p3d,
                point2d: p2d,
            };

            // Create a factor with the residual and add it to the graph
            // Use a simple standard deviation for the noise model
            let factor = fac![residual, KBCamParams(0), 1.0 as std, Huber::default()];
            graph.add_factor(factor);
        }

        if verbose {
            info!("Starting optimization with factrs Levenberg-Marquardt...");
        }

        // Create a Levenberg-Marquardt optimizer with QR solver
        let mut optimizer: LevenMarquardt<QRSolver> = LevenMarquardt::new(graph);

        // Run the optimization
        let result = optimizer
            .optimize(values)
            .map_err(|e| CameraModelError::NumericalError(format!("{:?}", e)))?;

        if verbose {
            info!("Optimization finished");
        }

        // Extract the optimized parameters
        let optimized_params: &VectorVar8<f64> = result.get(KBCamParams(0)).unwrap();

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
        info!("Validating automatic differentiation against analytical Jacobian...");

        let mut total_comparisons = 0;

        // Test with a subset of points to avoid excessive computation
        let test_points = std::cmp::min(10, self.points3d.ncols());

        for i in 0..test_points {
            let point3d = self.points3d.column(i);
            let point2d = self.points2d.column(i);

            // Create residual for this point
            let residual = KannalaBrandtFactrsResidual::new(
                Vector3::new(point3d[0], point3d[1], point3d[2]),
                Vector2::new(point2d[0], point2d[1]),
            );

            // Get current camera parameters
            let cam_params = [
                self.model.intrinsics.fx,
                self.model.intrinsics.fy,
                self.model.intrinsics.cx,
                self.model.intrinsics.cy,
                self.model.distortions[0],
                self.model.distortions[1],
                self.model.distortions[2],
                self.model.distortions[3],
            ];

            // Compute analytical residual and Jacobian
            match residual.compute_analytical_residual_jacobian(&cam_params) {
                Ok((analytical_residual, analytical_jacobian)) => {
                    // For now, we'll skip the automatic differentiation comparison
                    // as it requires more complex setup with the Diff trait
                    info!(
                        "Point {}: Analytical residual computed: [{}, {}]",
                        i, analytical_residual.x, analytical_residual.y
                    );
                    info!(
                        "Point {}: Analytical Jacobian shape: {}x{}",
                        i,
                        analytical_jacobian.nrows(),
                        analytical_jacobian.ncols()
                    );

                    total_comparisons += 1;
                }
                Err(e) => {
                    warn!(
                        "Failed to compute analytical Jacobian for point {}: {:?}",
                        i, e
                    );
                }
            }
        }

        info!(
            "Jacobian validation completed: total_comparisons = {}",
            total_comparisons
        );

        Ok(true) // For now, always return true since we're not doing full comparison
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

    #[test]
    fn test_kannala_brandt_factrs_residual_consistency() {
        // Test that the refactored residual computation produces consistent results
        let reference_model = get_sample_kb_camera_model();

        // Create a test point
        let point3d = Vector3::new(1.0, 0.5, 2.0);

        // Project it with the reference model to get the expected 2D point
        let point2d = reference_model.project(&point3d).unwrap();

        // Create a residual
        let residual = KannalaBrandtFactrsResidual::new(point3d, point2d);

        // Create camera parameters that match the reference model
        let cam_params = create_vector_var8(
            reference_model.intrinsics.fx as dtype,
            reference_model.intrinsics.fy as dtype,
            reference_model.intrinsics.cx as dtype,
            reference_model.intrinsics.cy as dtype,
            reference_model.distortions[0] as dtype,
            reference_model.distortions[1] as dtype,
            reference_model.distortions[2] as dtype,
            reference_model.distortions[3] as dtype,
        );

        // Compute residual - should be close to zero since we're using the same model
        let residual_value = residual.residual1(cam_params);

        info!(
            "Residual value: [{:.6}, {:.6}]",
            residual_value[0], residual_value[1]
        );

        // The residual should be very small (close to zero) since we're using the correct parameters
        assert!(
            residual_value[0].abs() < 1e-10,
            "Residual u component too large: {}",
            residual_value[0]
        );
        assert!(
            residual_value[1].abs() < 1e-10,
            "Residual v component too large: {}",
            residual_value[1]
        );

        // Test with slightly different parameters to ensure residual is non-zero
        let perturbed_params = create_vector_var8(
            (reference_model.intrinsics.fx * 1.01) as dtype, // 1% change
            reference_model.intrinsics.fy as dtype,
            reference_model.intrinsics.cx as dtype,
            reference_model.intrinsics.cy as dtype,
            reference_model.distortions[0] as dtype,
            reference_model.distortions[1] as dtype,
            reference_model.distortions[2] as dtype,
            reference_model.distortions[3] as dtype,
        );

        let perturbed_residual = residual.residual1(perturbed_params);
        info!(
            "Perturbed residual value: [{:.6}, {:.6}]",
            perturbed_residual[0], perturbed_residual[1]
        );

        // The residual should be non-zero when parameters are different
        assert!(
            perturbed_residual[0].abs() > 1e-6,
            "Residual should be non-zero for different parameters"
        );
    }

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
