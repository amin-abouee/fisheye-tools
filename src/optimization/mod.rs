//! The `optimization` module provides tools and traits for calibrating camera models.
//!
//! This module defines the [`Optimizer`] trait, which outlines the common interface
//! for different camera model optimization tasks. Each camera model that supports
//! calibration (e.g., Double Sphere, Kannala-Brandt, RadTan) will have a corresponding
//! optimization cost structure that implements this trait.
//!
//! The primary goal of these optimizers is to refine the camera model's parameters
//! (intrinsics and distortion coefficients) by minimizing the reprojection error
//! between observed 2D image points and 3D world points.
//!
//! The optimization process typically involves:
//! 1. An optional linear estimation step to get a rough initial guess for some parameters.
//! 2. A non-linear optimization step (usually Levenberg-Marquardt) to refine all parameters.
//!
//! This module re-exports the main optimization cost structures from its submodules.

use serde::{Deserialize, Serialize};

pub mod double_sphere;
pub mod eucm;
pub mod kannala_brandt;
pub mod rad_tan;
pub mod ucm;

pub use double_sphere::DoubleSphereOptimizationCost;
pub use eucm::EucmOptimizationCost;
pub use kannala_brandt::KannalaBrandtOptimizationCost;
pub use rad_tan::RadTanOptimizationCost;
pub use ucm::UcmOptimizationCost;

use crate::camera::{CameraModelError, Intrinsics, Resolution};

#[derive(Clone, Serialize, Deserialize)]
pub struct ProjectionError {
    pub rmse: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub stddev: f64,
    pub median: f64,
}

/// A trait for camera model optimization tasks.
///
/// Types implementing `Optimizer` are responsible for refining the parameters
/// of a specific camera model. This typically involves minimizing the
/// reprojection error given a set of 3D-2D point correspondences.
///
pub trait Optimizer {
    /// Performs non-linear optimization to refine the camera model parameters.
    ///
    /// This method should adjust the internal camera model's parameters (intrinsics
    /// and distortion coefficients) to minimize the reprojection error.
    ///
    /// # Arguments
    ///
    /// * `verbose` - If `true`, the optimizer may print progress information
    ///   and results to the console.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the optimization was successful and the model parameters
    ///   have been updated.
    /// * `Err(CameraModelError)` - If an error occurred during optimization,
    ///   such as invalid input parameters, numerical issues, or if the
    ///   optimization failed to converge.
    fn optimize(&mut self, verbose: bool) -> Result<(), CameraModelError>;

    /// Performs a linear estimation of some camera model parameters.
    ///
    /// This method provides an initial guess for a subset of the camera model
    /// parameters by solving a linear system. It's often used as a preliminary
    /// step before non-linear optimization. The specific parameters estimated
    /// depend on the camera model.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the linear estimation was successful and the relevant
    ///   model parameters have been updated.
    /// * `Err(CameraModelError)` - If an error occurred, such as insufficient
    ///   data or numerical issues.
    fn linear_estimation(&mut self) -> Result<(), CameraModelError>
    where
        Self: Sized;

    // fn compute_reprojection_error(&self) -> Result<ProjectionError, CameraModelError>;

    /// Retrieves the current intrinsic parameters from the underlying camera model.
    ///
    /// # Returns
    ///
    /// An [`Intrinsics`] struct containing `fx, fy, cx, cy`.
    fn get_intrinsics(&self) -> Intrinsics;

    /// Retrieves the current resolution from the underlying camera model.
    ///
    /// # Returns
    ///
    /// A [`Resolution`] struct containing `width` and `height`.
    fn get_resolution(&self) -> Resolution;

    /// Retrieves the current distortion parameters from the underlying camera model.
    ///
    /// The number and meaning of these parameters depend on the specific camera model.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` containing the distortion coefficients.
    fn get_distortion(&self) -> Vec<f64>;
}
