use nalgebra::{Point2, Point3};
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

/// Trait defining the core functionality for camera models
pub trait CameraModel {
    /// Project a 3D point to 2D image coordinates
    fn project(&self, point_3d: &Point3<f64>) -> Result<Point2<f64>, Box<dyn Error>>;
    
    /// Unproject 2D image coordinates to a 3D ray
    fn unproject(&self, point_2d: &Point2<f64>) -> Result<Point3<f64>, Box<dyn Error>>;
    
    /// Load camera parameters from a YAML file
    fn load_from_yaml(path: &str) -> Result<Self, Box<dyn Error>> where Self: Sized;
    
    /// Validate camera parameters
    fn validate_params(&self) -> Result<(), Box<dyn Error>>;
}

/// Common validation functions for camera parameters
pub mod validation {
    use super::*;

    pub fn validate_intrinsics(intrinsics: &Intrinsics) -> Result<(), Box<dyn Error>> {
        if intrinsics.fx <= 0.0 || intrinsics.fy <= 0.0 {
            return Err("Focal length must be positive".into());
        }
        if !intrinsics.cx.is_finite() || !intrinsics.cy.is_finite() {
            return Err("Principal point must be finite".into());
        }
        Ok(())
    }
} 