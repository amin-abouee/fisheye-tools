// src/optimization/mod.rs

pub mod double_sphere; // Added this line
pub mod kannala_brandt; // Added this line
pub mod rad_tan; // Added this line

pub use double_sphere::DoubleSphereOptimizationCost;
pub use kannala_brandt::KannalaBrandtOptimizationCost;
pub use rad_tan::RadTanOptimizationCost;

use crate::camera::{CameraModelError};

pub trait Optimizer {
    fn optimize(&mut self, verbose: bool) -> Result<(), CameraModelError>;

    fn linear_estimation(&mut self,) -> Result<(), CameraModelError>
    where
        Self: Sized;
}
