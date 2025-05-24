// src/optimization/mod.rs

pub mod double_sphere; // Added this line
pub mod kannala_brandt; // Added this line
pub mod rad_tan; // Added this line

use nalgebra::{Matrix2xX, Matrix3xX};
use crate::camera::{CameraModelError, Intrinsics, Resolution};
use argmin::core::{CostFunction, Gradient, Hessian, Jacobian, Operator};

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

pub trait OptimizationCost:
    CostFunction<Param = Self::Param, Output = Self::Output>
    + Gradient<Param = Self::Param, Gradient = Self::Jacobian> // Or define a separate associated type for Gradient if it's not the Jacobian
    + Hessian<Param = Self::Param, Hessian = Self::Hessian>
    + Jacobian<Param = Self::Param, Jacobian = Self::Jacobian>
    + Operator<Param = Self::Param, Output = Self::Output>
{
    type Param;
    type Output;
    type Jacobian; // As per argmin::core::Jacobian
    type Hessian; // As per argmin::core::Hessian
}
