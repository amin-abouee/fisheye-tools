use nalgebra::{Point2, Point3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

#[derive(thiserror::Error, Debug)]
pub enum CameraModelError {
    #[error("Projection is outside the image")]
    ProjectionOutSideImage,
    #[error("Input point is outside the image")]
    PointIsOutSideImage,
    #[error("z is close to zero, point is at camera center")]
    PointAtCameraCenter,
    #[error("Focal length must be positive")]
    FocalLengthMustBePositive,
    #[error("Principal point must be finite")]
    PrincipalPointMustBeFinite,
    #[error("Invalid camera parameters: {0}")]
    InvalidParams(String),
    #[error("Failed to load YAML: {0}")]
    YamlError(String),
    #[error("IO Error: {0}")]
    IOError(String),
}

impl From<std::io::Error> for CameraModelError {
    fn from(err: std::io::Error) -> Self {
        CameraModelError::IOError(err.to_string())
    }
}

impl From<yaml_rust::ScanError> for CameraModelError {
    fn from(err: yaml_rust::ScanError) -> Self {
        CameraModelError::YamlError(err.to_string())
    }
}

/// Trait defining the core functionality for camera models
pub trait CameraModel {
    /// Project a 3D point to 2D image coordinates
    fn project(&self, point_3d: &Point3<f64>) -> Result<Point2<f64>, CameraModelError>;

    /// Unproject 2D image coordinates to a 3D ray
    fn unproject(&self, point_2d: &Point2<f64>) -> Result<Point3<f64>, CameraModelError>;

    /// Load camera parameters from a YAML file
    fn load_from_yaml(path: &str) -> Result<Self, CameraModelError>
    where
        Self: Sized;

    /// Validate camera parameters
    fn validate_params(&self) -> Result<(), CameraModelError>;
}

/// Common validation functions for camera parameters
pub mod validation {
    use super::*;

    pub fn validate_intrinsics(intrinsics: &Intrinsics) -> Result<(), CameraModelError> {
        if intrinsics.fx <= 0.0 || intrinsics.fy <= 0.0 {
            return Err(CameraModelError::FocalLengthMustBePositive);
        }
        if !intrinsics.cx.is_finite() || !intrinsics.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointMustBeFinite);
        }
        Ok(())
    }
}
