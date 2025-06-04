//! This module provides the cost function and optimization routines
//! for calibrating an Extended Unified Camera Model (EUCM).
//!
//! It uses the `factrs` crate for non-linear optimization and defines
//! the necessary structures and traits to integrate the EUCM
//! camera model with the optimization framework.

use crate::camera::{CameraModel, CameraModelError, EucmModel};
use crate::optimization::Optimizer;
use factrs::{
    assign_symbols,
    core::{Graph, Huber, LevenMarquardt, Values},
    dtype, fac,
    linalg::{Const, ForwardProp, Numeric, SupersetOf, VectorX},
    linear::QRSolver,
    optimizers::Optimizer as FactrsOptimizer,
    residuals::Residual1,
    variables::VectorVar6,
};
use log::info;
use nalgebra::{Matrix2xX, Matrix3xX, Vector2, Vector3};

assign_symbols!(EUCMCamParams: VectorVar6);

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
    pub fn new(
        model: EucmModel,
        points3d: Matrix3xX<f64>,
        points2d: Matrix2xX<f64>,
    ) -> Self {
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

/// Residual implementation for `factrs` optimization of the [`EucmModel`].
///
/// This struct defines the residual error between the observed 2D point and
/// the projected 2D point from a 3D point using the current camera model parameters.
/// It is used by the `factrs` optimization framework.
#[derive(Debug, Clone)]
pub struct EucmFactrsResidual {
    /// The 3D point in the camera's coordinate system.
    point3d: Vector3<dtype>,
    /// The corresponding observed 2D point in image coordinates.
    point2d: Vector2<dtype>,
}

impl EucmFactrsResidual {
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


}

// Mark this residual for factrs serialization and other features
#[factrs::mark]
impl Residual1 for EucmFactrsResidual {
    type DimIn = Const<6>;
    type DimOut = Const<2>;
    type V1 = VectorVar6<dtype>;
    type Differ = ForwardProp<Self::DimIn>;

    /// Computes the residual vector for the given camera parameters.
    ///
    /// The residual is defined as `observed_2d_point - project(3d_point, camera_parameters)`.
    /// This method is called by the `factrs` optimizer during the optimization process.
    ///
    /// # Arguments
    ///
    /// * `cam_params` - A `VectorVar6<T>` containing the current camera parameters
    ///   `[fx, fy, cx, cy, alpha, beta]`. `T` is a numeric type used by `factrs`.
    ///
    /// # Returns
    ///
    /// A `VectorX<T>` of dimension 2, representing the residual `[ru, rv]`.
    fn residual1<T: Numeric>(&self, cam_params: VectorVar6<T>) -> VectorX<T> {
        // Convert camera parameters from generic type T to f64 for EucmModel
        // Using to_subset() which is available through SupersetOf<f64> trait
        let fx_f64 = cam_params[0].to_subset().unwrap();
        let fy_f64 = cam_params[1].to_subset().unwrap();
        let cx_f64 = cam_params[2].to_subset().unwrap();
        let cy_f64 = cam_params[3].to_subset().unwrap();
        let alpha_f64 = cam_params[4].to_subset().unwrap();
        let beta_f64 = cam_params[5].to_subset().unwrap();

        // Create a temporary EUCM model with current parameters
        let temp_model = EucmModel {
            intrinsics: crate::camera::Intrinsics {
                fx: fx_f64,
                fy: fy_f64,
                cx: cx_f64,
                cy: cy_f64,
            },
            resolution: crate::camera::Resolution { width: 0, height: 0 }, // Not used in projection
            alpha: alpha_f64,
            beta: beta_f64,
        };

        // Convert 3D point to f64 for projection
        let point3d_f64 = Vector3::new(
            self.point3d.x.to_subset().unwrap(),
            self.point3d.y.to_subset().unwrap(),
            self.point3d.z.to_subset().unwrap(),
        );

        // Project the 3D point using the temporary model
        let projected_2d = match temp_model.project(&point3d_f64) {
            Ok(p) => p,
            Err(_) => Vector2::new(-1.0, -1.0), // Return invalid projection on error
        };

        // Convert observed and projected points to generic type T
        let obs_u = T::from_subset(&self.point2d.x);
        let obs_v = T::from_subset(&self.point2d.y);
        let proj_u = T::from_subset(&projected_2d.x);
        let proj_v = T::from_subset(&projected_2d.y);

        // Return residual: observed - projected
        VectorX::from_vec(vec![obs_u - proj_u, obs_v - proj_v])
    }
}

impl Optimizer for EucmOptimizationCost {
    /// Performs non-linear optimization to refine the EUCM camera model parameters.
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
            info!("Starting EUCM camera model optimization...");
            info!("Initial model: {:?}", self.model);
            info!("Number of point correspondences: {}", self.points3d.ncols());
        }

        // Create a factrs Values object to hold the camera parameters
        let mut values = Values::new();

        // Initial parameters - create VectorVar6 from current model parameters
        let initial_params = VectorVar6::new(
            self.model.intrinsics.fx as dtype,
            self.model.intrinsics.fy as dtype,
            self.model.intrinsics.cx as dtype,
            self.model.intrinsics.cy as dtype,
            self.model.alpha as dtype,
            self.model.beta as dtype,
        );

        values.insert(EUCMCamParams(0), initial_params);

        // Create a factrs Graph
        let mut graph = Graph::new();

        // Add residuals for each point correspondence
        for i in 0..self.points3d.ncols() {
            let point3d = self.points3d.column(i).into_owned();
            let point2d = self.points2d.column(i).into_owned();

            let residual = EucmFactrsResidual::new(point3d, point2d);

            // Create a factor with the residual and add it to the graph
            let factor = fac![residual, EUCMCamParams(0), 1.0 as std, Huber::default()];
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
        let optimized_params: &VectorVar6<f64> = result.get(EUCMCamParams(0)).unwrap();
        let params = &optimized_params.0;

        // Update the model parameters
        self.model.intrinsics.fx = params[0] as f64;
        self.model.intrinsics.fy = params[1] as f64;
        self.model.intrinsics.cx = params[2] as f64;
        self.model.intrinsics.cy = params[3] as f64;
        self.model.alpha = params[4] as f64;
        self.model.beta = params[5] as f64;

        // Validate the optimized parameters
        self.model.validate_params()?;

        if verbose {
            info!("Optimized parameters: {:?}", self.model);
        }

        Ok(())
    }

    /// Performs linear estimation to initialize camera parameters.
    ///
    /// For the EUCM model, this method provides a basic linear estimation
    /// of the intrinsic parameters using a simplified approach.
    /// The distortion parameters (alpha, beta) are initialized to default values.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If linear estimation was successful and model parameters were updated.
    /// * `Err(CameraModelError)` - If estimation failed or parameters are invalid.
    fn linear_estimation(&mut self) -> Result<(), CameraModelError> {
        info!("Starting EUCM linear parameter estimation...");

        // For EUCM, we can use a simplified linear estimation approach
        // This is a basic implementation - in practice, you might want to use
        // more sophisticated methods like DLT (Direct Linear Transform)

        if self.points3d.ncols() < 4 {
            return Err(CameraModelError::InvalidParams(
                "At least 4 point correspondences required for linear estimation".to_string(),
            ));
        }

        // Simple estimation: use the center of the image as principal point
        // and estimate focal lengths from the point correspondences
        let mut sum_fx = 0.0;
        let mut sum_fy = 0.0;
        let mut count = 0;

        for i in 0..self.points3d.ncols() {
            let p3d = self.points3d.column(i);
            let p2d = self.points2d.column(i);

            if p3d.z > 0.0 {
                // Simple perspective projection approximation
                let fx_est = (p2d.x - self.model.intrinsics.cx) * p3d.z / p3d.x;
                let fy_est = (p2d.y - self.model.intrinsics.cy) * p3d.z / p3d.y;

                if fx_est.is_finite() && fy_est.is_finite() && fx_est > 0.0 && fy_est > 0.0 {
                    sum_fx += fx_est;
                    sum_fy += fy_est;
                    count += 1;
                }
            }
        }

        if count > 0 {
            self.model.intrinsics.fx = sum_fx / count as f64;
            self.model.intrinsics.fy = sum_fy / count as f64;
        }

        // Initialize distortion parameters to reasonable defaults
        self.model.alpha = 1.0;
        self.model.beta = 1.0;

        // Validate the estimated parameters
        self.model.validate_params()?;

        info!("Linear estimation completed: {:?}", self.model);

        Ok(())
    }

    /// Returns the intrinsic parameters of the camera model.
    fn get_intrinsics(&self) -> crate::camera::Intrinsics {
        self.model.get_intrinsics()
    }

    /// Returns the resolution of the camera model.
    fn get_resolution(&self) -> crate::camera::Resolution {
        self.model.get_resolution()
    }

    /// Returns the distortion parameters of the camera model.
    fn get_distortion(&self) -> Vec<f64> {
        self.model.get_distortion()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2xX, Matrix3xX};

    /// Test creating an EUCM optimization cost function.
    #[test]
    fn test_eucm_optimization_cost_creation() {
        let model = EucmModel {
            intrinsics: crate::camera::Intrinsics {
                fx: 1000.0,
                fy: 1000.0,
                cx: 320.0,
                cy: 240.0,
            },
            resolution: crate::camera::Resolution {
                width: 640,
                height: 480,
            },
            alpha: 1.0,
            beta: 1.0,
        };

        let points3d = Matrix3xX::from_columns(&[
            nalgebra::Vector3::new(1.0, 0.0, 5.0),
            nalgebra::Vector3::new(0.0, 1.0, 5.0),
            nalgebra::Vector3::new(-1.0, 0.0, 5.0),
            nalgebra::Vector3::new(0.0, -1.0, 5.0),
        ]);

        let points2d = Matrix2xX::from_columns(&[
            nalgebra::Vector2::new(520.0, 240.0),
            nalgebra::Vector2::new(320.0, 440.0),
            nalgebra::Vector2::new(120.0, 240.0),
            nalgebra::Vector2::new(320.0, 40.0),
        ]);

        let cost = EucmOptimizationCost::new(model, points3d, points2d);
        assert_eq!(cost.get_model().alpha, 1.0);
        assert_eq!(cost.get_model().beta, 1.0);
    }

    /// Test the getter methods for the optimization cost function.
    #[test]
    fn test_eucm_optimization_getters() {
        let model = EucmModel {
            intrinsics: crate::camera::Intrinsics {
                fx: 1000.0,
                fy: 1000.0,
                cx: 320.0,
                cy: 240.0,
            },
            resolution: crate::camera::Resolution {
                width: 640,
                height: 480,
            },
            alpha: 1.0,
            beta: 1.0,
        };

        let points3d = Matrix3xX::zeros(4);
        let points2d = Matrix2xX::zeros(4);

        let cost = EucmOptimizationCost::new(model, points3d, points2d);

        let intrinsics = cost.get_intrinsics();
        assert_eq!(intrinsics.fx, 1000.0);
        assert_eq!(intrinsics.fy, 1000.0);

        let resolution = cost.get_resolution();
        assert_eq!(resolution.width, 640);
        assert_eq!(resolution.height, 480);

        let distortion = cost.get_distortion();
        assert_eq!(distortion.len(), 2);
        assert_eq!(distortion[0], 1.0); // alpha
        assert_eq!(distortion[1], 1.0); // beta
    }

    /// Test linear estimation functionality.
    #[test]
    fn test_eucm_optimization_linear_estimation() {
        let model = EucmModel {
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
            alpha: 0.5,
            beta: 0.5,
        };

        // Create some synthetic 3D-2D correspondences
        let points3d = Matrix3xX::from_columns(&[
            nalgebra::Vector3::new(1.0, 0.0, 5.0),
            nalgebra::Vector3::new(0.0, 1.0, 5.0),
            nalgebra::Vector3::new(-1.0, 0.0, 5.0),
            nalgebra::Vector3::new(0.0, -1.0, 5.0),
        ]);

        let points2d = Matrix2xX::from_columns(&[
            nalgebra::Vector2::new(420.0, 240.0),
            nalgebra::Vector2::new(320.0, 340.0),
            nalgebra::Vector2::new(220.0, 240.0),
            nalgebra::Vector2::new(320.0, 140.0),
        ]);

        let mut cost = EucmOptimizationCost::new(model, points3d, points2d);

        // Test linear estimation
        let result = cost.linear_estimation();
        assert!(result.is_ok());

        // Check that parameters were updated
        let updated_model = cost.get_model();
        assert!(updated_model.alpha.is_finite());
        assert!(updated_model.beta.is_finite());
        assert!(updated_model.intrinsics.fx > 0.0);
        assert!(updated_model.intrinsics.fy > 0.0);
    }
}
