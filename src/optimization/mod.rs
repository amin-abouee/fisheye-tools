// src/optimization/mod.rs

//! Defines the core optimization functionalities for camera models.
//!
//! This module provides the [`Optimizer`] trait, which outlines a common interface
//! for optimizing the parameters of different camera models. The goal of optimization
//! is typically to refine camera parameters (intrinsics and distortion coefficients)
//! by minimizing a cost function, often based on reprojection errors from observed
//! 3D-2D point correspondences.
//!
//! It re-exports specific optimization cost function implementations from its submodules:
//! *   [`double_sphere`]: Optimization for the [`DoubleSphereModel`].
//! *   [`kannala_brandt`]: Optimization for the [`KannalaBrandtModel`].
//! *   [`rad_tan`]: Optimization for the [`RadTanModel`].
//!
//! The optimization process often involves two stages:
//! 1.  **Linear Estimation**: A preliminary estimation of parameters using linear methods,
//!     which provides a good starting point for non-linear optimization.
//! 2.  **Non-linear Optimization**: An iterative process that refines the parameters by
//!     minimizing the chosen cost function, typically using a numerical solver.
//!
//! This module relies on types from the `camera` module, such as [`CameraModelError`],
//! as optimization can encounter issues like numerical instability or invalid parameters.

pub mod double_sphere;
pub mod kannala_brandt;
pub mod rad_tan;

pub use double_sphere::DoubleSphereOptimizationCost;
pub use kannala_brandt::KannalaBrandtOptimizationCost;
pub use rad_tan::RadTanOptimizationCost;

use crate::camera::CameraModelError;

/// Defines a common interface for optimizing camera model parameters.
///
/// Implementations of this trait are responsible for refining the parameters of a
/// specific camera model. This usually involves minimizing a cost function, such as
/// the reprojection error between observed 2D image points and projected 3D world points.
/// The optimization process might include an initial linear estimation step followed by
/// an iterative non-linear optimization.
pub trait Optimizer {
    /// Runs the main non-linear optimization process to refine camera parameters.
    ///
    /// This method typically employs an iterative numerical solver (e.g., Gauss-Newton,
    /// Levenberg-Marquardt) to minimize a cost function (often reprojection error)
    /// and find the optimal set of camera parameters. It modifies the internal state
    /// of the implementing camera model object.
    ///
    /// # Arguments
    ///
    /// *   `verbose`: `bool` - A flag to control whether detailed logging or output
    ///     should be enabled during the optimization process.
    ///
    /// # Return Value
    ///
    /// Returns `Ok(())` if the optimization converges successfully.
    ///
    /// # Errors
    ///
    /// Returns a [`CameraModelError`] if the optimization fails. Potential reasons include:
    /// *   Numerical instability during the solving process.
    /// *   Failure to converge within the maximum number of iterations.
    /// *   Invalid or inconsistent data (e.g., insufficient point correspondences).
    fn optimize(&mut self, verbose: bool) -> Result<(), CameraModelError>;

    /// Provides an initial estimate for camera parameters using linear methods.
    ///
    /// This method is often used as a precursor to non-linear optimization.
    /// It computes an initial guess for the camera parameters based on a set of
    /// point correspondences, typically using simpler, non-iterative techniques.
    /// The quality of this initial estimate can significantly impact the convergence
    /// and success of the subsequent non-linear optimization.
    ///
    /// # Return Value
    ///
    /// Returns `Ok(())` if the linear estimation is successful and parameters are updated.
    ///
    /// # Errors
    ///
    /// Returns a [`CameraModelError`] if the linear estimation fails. This could be due to:
    /// *   Insufficient or degenerate data (e.g., co-linear points).
    /// *   Numerical issues in the linear algebraic solutions.
    ///
    /// # Generic Constraints
    ///
    /// *   `where Self: Sized`: This constraint indicates that the method can only be
    ///     called on types that have a known size at compile time. It is a common
    ///     default constraint for methods that operate on `self`.
    fn linear_estimation(&mut self) -> Result<(), CameraModelError>
    where
        Self: Sized;
}
