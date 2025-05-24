// src/optimization/mod.rs

pub mod double_sphere; // Added this line
pub mod kannala_brandt; // Added this line
pub mod rad_tan; // Added this line

pub use double_sphere::DoubleSphereOptimizationCost;
pub use kannala_brandt::KannalaBrandtOptimizationCost;
pub use rad_tan::RadTanOptimizationCost;

use crate::camera::{CameraModelError, Intrinsics, Resolution};
use nalgebra::{Matrix2xX, Matrix3xX};

pub trait Optimizer {
    fn optimize(
        &mut self,
        points_3d: &Matrix3xX<f64>,
        points_2d: &Matrix2xX<f64>,
        verbose: bool,
    ) -> Result<(), CameraModelError>;

    fn linear_estimation(
        intrinsics: &Intrinsics,
        resolution: &Resolution,
        points_2d: &Matrix2xX<f64>,
        points_3d: &Matrix3xX<f64>,
    ) -> Result<Self, CameraModelError>
    where
        Self: Sized;
}
