// src/optimization/mod.rs

pub mod double_sphere; // Added this line
pub mod kannala_brandt; // Added this line
pub mod rad_tan; // Added this line

pub use double_sphere::DoubleSphereOptimizationCost;
pub use kannala_brandt::KannalaBrandtOptimizationCost;
pub use rad_tan::RadTanOptimizationCost;

use crate::camera::{CameraModelError, Intrinsics, Resolution};

pub trait Optimizer {
    fn optimize(&mut self, verbose: bool) -> Result<(), CameraModelError>;

    fn linear_estimation(&mut self,) -> Result<(), CameraModelError>
    where
        Self: Sized;

    /// Get the intrinsic parameters from the underlying camera model
    fn get_intrinsics(&self) -> Intrinsics;

    /// Get the resolution from the underlying camera model
    fn get_resolution(&self) -> Resolution;

    /// Get the distortion parameters from the underlying camera model
    fn get_distortion(&self) -> Vec<f64>;
}
