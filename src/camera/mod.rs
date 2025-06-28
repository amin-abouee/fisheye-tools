//! This module defines the core functionalities for various camera models.
//!
//! It provides a unified interface for camera operations such as projecting 3D points
//! to 2D image coordinates and unprojecting 2D image coordinates to 3D rays.
//! The module also includes definitions for camera intrinsic parameters, resolution,
//! and error handling for camera operations.
//!
//! This module re-exports several specific camera model implementations from its submodules:
//! - `double_sphere`: Implements the Double Sphere camera model.
//! - `kannala_brandt`: Implements the Kannala-Brandt camera model.
//! - `pinhole`: Implements the Pinhole camera model.
//! - `rad_tan`: Implements the Radial-Tangential distortion model (often used with pinhole).
//!
//! It also contains a `validation` submodule for common parameter validation logic.

use nalgebra::{Vector2, Vector3};
use serde::{Deserialize, Serialize};

// Camera model modules
pub mod double_sphere;
pub mod eucm;
pub mod kannala_brandt;
pub mod pinhole;
pub mod rad_tan;
pub mod ucm;

// Re-export camera models
pub use double_sphere::DoubleSphereModel;
pub use eucm::EucmModel;
pub use kannala_brandt::KannalaBrandtModel;
pub use pinhole::PinholeModel;
pub use rad_tan::RadTanModel;
pub use ucm::UcmModel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CameraModelEnum {
    DoubleSphere(DoubleSphereModel),
    Eucm(EucmModel),
    KannalaBrandt(KannalaBrandtModel),
    Pinhole(PinholeModel),
    RadTan(RadTanModel),
    Ucm(UcmModel),
}

/// Represents the intrinsic parameters of a camera.
///
/// These parameters define the internal geometry of the camera,
/// including focal length and principal point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intrinsics {
    /// The focal length along the x-axis, in pixels.
    pub fx: f64,
    /// The focal length along the y-axis, in pixels.
    pub fy: f64,
    /// The x-coordinate of the principal point (optical center), in pixels.
    pub cx: f64,
    /// The y-coordinate of the principal point (optical center), in pixels.
    pub cy: f64,
}

/// Represents the resolution of a camera image.
///
/// This struct holds the width and height of the image sensor in pixels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resolution {
    /// The width of the image in pixels.
    pub width: u32,
    /// The height of the image in pixels.
    pub height: u32,
}

/// Defines the possible errors that can occur during camera model operations.
///
/// This enum covers errors related to projection, unprojection, parameter validation,
/// file I/O, and numerical issues.
#[derive(thiserror::Error, Debug)]
pub enum CameraModelError {
    /// Error indicating that a 3D point projects outside the valid image area.
    #[error("Projection is outside the image")]
    ProjectionOutSideImage,
    /// Error indicating that an input 2D point for unprojection is outside the valid image area.
    #[error("Input point is outside the image")]
    PointIsOutSideImage,
    /// Error indicating that a 3D point is too close to the camera center (z-coordinate is near zero),
    /// making projection or unprojection numerically unstable or undefined.
    #[error("z is close to zero, point is at camera center")]
    PointAtCameraCenter,
    /// Error indicating that a focal length parameter (fx or fy) is not positive.
    #[error("Focal length must be positive")]
    FocalLengthMustBePositive,
    /// Error indicating that a principal point coordinate (cx or cy) is not a finite number.
    #[error("Principal point must be finite")]
    PrincipalPointMustBeFinite,
    /// Error indicating that one or more camera parameters are invalid.
    /// Contains a string describing the specific parameter issue.
    #[error("Invalid camera parameters: {0}")]
    InvalidParams(String),
    /// Error indicating a failure during YAML deserialization when loading camera parameters.
    /// Contains a string describing the YAML parsing error.
    #[error("Failed to load YAML: {0}")]
    YamlError(String),
    /// Error indicating a failure during file input/output operations.
    /// Contains a string describing the I/O error.
    #[error("IO Error: {0}")]
    IOError(String),
    /// Error indicating a numerical instability or issue during calculations.
    /// Contains a string describing the numerical error.
    #[error("NumericalError: {0}")]
    NumericalError(String),
}

/// Implements the conversion from `std::io::Error` to `CameraModelError::IOError`.
///
/// This allows for seamless handling of I/O errors within the camera model context.
impl From<std::io::Error> for CameraModelError {
    fn from(err: std::io::Error) -> Self {
        CameraModelError::IOError(err.to_string())
    }
}

/// Implements the conversion from `yaml_rust::ScanError` to `CameraModelError::YamlError`.
///
/// This allows for seamless handling of YAML parsing errors when loading camera parameters.
impl From<yaml_rust::ScanError> for CameraModelError {
    fn from(err: yaml_rust::ScanError) -> Self {
        CameraModelError::YamlError(err.to_string())
    }
}

/// Defines the core functionality and interface for all camera models.
///
/// This trait provides a common set of methods that any camera model implementation
/// must provide, such as projection, unprojection, parameter loading/saving,
/// validation, and retrieval of intrinsic and distortion parameters.
pub trait CameraModel {
    /// Projects a 3D point from the camera's coordinate system to 2D image coordinates.
    ///
    /// # Arguments
    /// * `point_3d` - A reference to a `Vector3<f64>` representing the 3D point (X, Y, Z) in camera coordinates.
    /// * `compute_jacobian` - A boolean flag indicating whether to compute the Jacobian of the projection function.
    ///
    /// # Returns
    /// A `Result` containing:
    /// * `Ok((Vector2<f64>, Option<DMatrix<f64>>))`: A tuple where the first element is the
    ///   projected 2D point (u, v) in pixel coordinates, and the second element is an `Option`
    ///   containing the Jacobian matrix (2x3) if `compute_jacobian` was true, otherwise `None`.
    /// * `Err(CameraModelError)`: An error if the projection fails (e.g., point is behind the camera,
    ///   projects outside the image, or numerical issues). Possible errors include
    ///   `ProjectionOutSideImage` and `PointAtCameraCenter`.
    fn project(&self, point_3d: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError>;

    /// Unprojects a 2D point from image coordinates to a 3D ray in the camera's coordinate system.
    ///
    /// The resulting 3D vector is a direction ray originating from the camera center.
    /// Its Z component is typically normalized to 1, but this can vary by model.
    ///
    /// # Arguments
    /// * `point_2d` - A reference to a `Vector2<f64>` representing the 2D point (u, v) in pixel coordinates.
    ///
    /// # Returns
    /// A `Result` containing:
    /// * `Ok(Vector3<f64>)`: The 3D ray (direction vector) corresponding to the 2D point.
    /// * `Err(CameraModelError)`: An error if the unprojection fails (e.g., point is outside the image,
    ///   or numerical issues). Possible errors include `PointIsOutSideImage`.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError>;

    /// Loads camera parameters from a YAML file.
    ///
    /// This method should parse a YAML file specified by `path` and populate the
    /// camera model's parameters.
    ///
    /// # Arguments
    /// * `path` - A string slice representing the path to the YAML file.
    ///
    /// # Returns
    /// A `Result` containing:
    /// * `Ok(Self)`: An instance of the camera model with parameters loaded from the file.
    /// * `Err(CameraModelError)`: An error if loading fails (e.g., file not found, YAML parsing error,
    ///   invalid parameters). Possible errors include `IOError`, `YamlError`, and `InvalidParams`.
    fn load_from_yaml(path: &str) -> Result<Self, CameraModelError>
    where
        Self: Sized;

    /// Saves the camera model's parameters to a YAML file.
    ///
    /// # Arguments
    /// * `path` - A string slice representing the path to the YAML file where parameters will be saved.
    ///
    /// # Returns
    /// A `Result` containing:
    /// * `Ok(())`: If saving was successful.
    /// * `Err(CameraModelError)`: An error if saving fails (e.g., I/O error). Possible errors include `IOError`.
    fn save_to_yaml(&self, path: &str) -> Result<(), CameraModelError>;

    /// Validates the current camera parameters.
    ///
    /// This method checks if the intrinsic parameters, distortion coefficients (if any),
    /// and other model-specific parameters are valid.
    ///
    /// # Returns
    /// A `Result` containing:
    /// * `Ok(())`: If all parameters are valid.
    /// * `Err(CameraModelError)`: An error describing the validation failure. Possible errors include
    ///   `FocalLengthMustBePositive`, `PrincipalPointMustBeFinite`, and `InvalidParams`.
    fn validate_params(&self) -> Result<(), CameraModelError>;

    /// Returns the resolution of the camera.
    ///
    /// # Returns
    /// A `Resolution` struct containing the width and height of the camera image.
    fn get_resolution(&self) -> Resolution;

    /// Returns the intrinsic parameters of the camera.
    ///
    /// # Returns
    /// An `Intrinsics` struct containing the focal lengths (fx, fy) and principal point (cx, cy).
    fn get_intrinsics(&self) -> Intrinsics;

    /// Returns the distortion parameters of the camera.
    ///
    /// The specific meaning and number of distortion parameters depend on the camera model.
    ///
    /// # Returns
    /// A `Vec<f64>` containing the distortion coefficients.
    fn get_distortion(&self) -> Vec<f64>;
}

/// Provides common validation functions for camera parameters.
///
/// This module groups utility functions used to validate parts of camera models,
/// such as intrinsic parameters, ensuring they meet common criteria (e.g., positive focal length).
pub mod validation {
    use super::*;

    /// Validates the intrinsic camera parameters.
    ///
    /// Checks if the focal lengths (fx, fy) are positive and if the principal
    /// point coordinates (cx, cy) are finite numbers.
    ///
    /// # Arguments
    /// * `intrinsics` - A reference to an `Intrinsics` struct containing the parameters to validate.
    ///
    /// # Returns
    /// A `Result` containing:
    /// * `Ok(())`: If the intrinsic parameters are valid.
    /// * `Err(CameraModelError)`: An error if validation fails. Possible errors include
    ///   `FocalLengthMustBePositive` and `PrincipalPointMustBeFinite`.
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
