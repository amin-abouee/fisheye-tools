//! Implements the Pinhole camera model.
//!
//! This module provides the [`PinholeModel`] struct and its associated methods
//! for representing and working with a simple pinhole camera. It adheres to the
//! [`CameraModel`] trait defined in the parent `camera` module ([`crate::camera`]).
//! The pinhole model is the simplest camera model, assuming no lens distortion.

use crate::camera::{validation, CameraModel, CameraModelError, Intrinsics, Resolution};
use nalgebra::{DVector, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use yaml_rust::YamlLoader;

/// Represents a Pinhole camera model.
///
/// This struct holds the intrinsic parameters (focal length, principal point)
/// and image resolution for a pinhole camera. It assumes no lens distortion.
///
/// # Examples
///
/// ```rust
/// use nalgebra::DVector;
/// use fisheye_tools::camera::pinhole::PinholeModel;
/// use fisheye_tools::camera::{Intrinsics, Resolution, CameraModelError};
///
/// // Create a PinholeModel using the new constructor
/// let params = DVector::from_vec(vec![500.0, 500.0, 320.0, 240.0]); // fx, fy, cx, cy
/// let mut pinhole_model = PinholeModel::new(&params).unwrap();
/// // Set a resolution, as `new` initializes it to 0x0
/// pinhole_model.resolution = Resolution { width: 640, height: 480 };
///
/// assert_eq!(pinhole_model.intrinsics.fx, 500.0);
/// assert_eq!(pinhole_model.resolution.width, 640);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinholeModel {
    /// The intrinsic parameters of the camera, [`Intrinsics`] (fx, fy, cx, cy).
    pub intrinsics: Intrinsics,
    /// The resolution of the camera image, [`Resolution`] (width, height).
    pub resolution: Resolution,
}

impl PinholeModel {
    /// Creates a new [`PinholeModel`] from a vector of parameters.
    ///
    /// The resolution is initialized to 0x0 and should be set manually or by loading from YAML.
    ///
    /// # Arguments
    ///
    /// * `parameters` - A `&DVector<f64>` containing the intrinsic parameters in the order:
    ///   `fx` (focal length x), `fy` (focal length y), `cx` (principal point x), `cy` (principal point y).
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Self, CameraModelError>`. On success, it provides the [`PinholeModel`] instance.
    ///
    /// # Errors
    ///
    /// This function can return a [`CameraModelError`] if the provided parameters are invalid
    /// (e.g., non-positive focal length, non-finite principal point), as checked by `validate_params`.
    /// Specifically, it can return:
    /// * [`CameraModelError::FocalLengthMustBePositive`]
    /// * [`CameraModelError::PrincipalPointMustBeFinite`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::DVector;
    /// use fisheye_tools::camera::pinhole::PinholeModel;
    /// use fisheye_tools::camera::{Resolution, CameraModelError};
    ///
    /// let params = DVector::from_vec(vec![500.0, 500.0, 320.0, 240.0]);
    /// let mut model = PinholeModel::new(&params).expect("Failed to create PinholeModel");
    /// model.resolution = Resolution { width: 640, height: 480 }; // Set resolution
    ///
    /// assert_eq!(model.intrinsics.fx, 500.0);
    /// assert_eq!(model.intrinsics.fy, 500.0);
    /// assert_eq!(model.intrinsics.cx, 320.0);
    /// assert_eq!(model.intrinsics.cy, 240.0);
    /// assert_eq!(model.resolution.width, 640);
    /// assert_eq!(model.resolution.height, 480);
    /// ```
    pub fn new(parameters: &DVector<f64>) -> Result<Self, CameraModelError> {
        let model = PinholeModel {
            intrinsics: Intrinsics {
                fx: parameters[0],
                fy: parameters[1],
                cx: parameters[2],
                cy: parameters[3],
            },
            resolution: Resolution {
                width: 0,  // Resolution is typically set after creation or by loading.
                height: 0, // It's not part of the minimal parameter set for `new`.
            },
        };

        model.validate_params()?;

        Ok(model)
    }
}

impl CameraModel for PinholeModel {
    // fn initialize(&mut self, parameters: &DVector<f64>) -> Result<(), CameraModelError> {
    //     self.intrinsics = Intrinsics {
    //         fx: parameters[0],
    //         fy: parameters[1],
    //         cx: parameters[2],
    //         cy: parameters[3],
    //     };
    //     self.resolution = Resolution {
    //         width: 0,
    //         height: 0,
    //     };
    //     Ok(())
    // }

    /// Projects a 3D point from camera coordinates to 2D image coordinates.
    ///
    /// This method applies the pinhole camera projection equations:
    /// `u = fx * X / Z + cx`
    /// `v = fy * Y / Z + cy`
    ///
    /// # Arguments
    ///
    /// * `point_3d` - A `&Vector3<f64>` representing the 3D point (X, Y, Z) in camera coordinates.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Vector2<f64>, CameraModelError>`.
    /// On success, it provides the projected 2D point (`Vector2<f64>`) in pixel coordinates (u, v).
    ///
    /// # Errors
    ///
    /// * [`CameraModelError::PointAtCameraCenter`]: If the 3D point's Z-coordinate is too close to zero.
    /// * [`CameraModelError::ProjectionOutSideImage`]: If the projected 2D point falls outside the camera's resolution.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::{DVector, Vector3};
    /// use fisheye_tools::camera::pinhole::PinholeModel;
    /// use fisheye_tools::camera::{CameraModel, Resolution};
    ///
    /// let params = DVector::from_vec(vec![500.0, 500.0, 320.0, 240.0]);
    /// let mut model = PinholeModel::new(&params).unwrap();
    /// model.resolution = Resolution { width: 640, height: 480 };
    ///
    /// let point_3d = Vector3::new(0.1, 0.2, 1.0); // X, Y, Z in meters
    /// match model.project(&point_3d) {
    ///     Ok(point_2d) => {
    ///         println!("Projected point: ({}, {})", point_2d.x, point_2d.y);
    ///         // Expected: u = 500 * 0.1 / 1.0 + 320 = 50 + 320 = 370
    ///         // Expected: v = 500 * 0.2 / 1.0 + 240 = 100 + 240 = 340
    ///         assert!((point_2d.x - 370.0).abs() < 1e-6);
    ///         assert!((point_2d.y - 340.0).abs() < 1e-6);
    ///     }
    ///     Err(e) => println!("Projection failed: {:?}", e),
    /// }
    /// ```
    fn project(&self, point_3d: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        // If z is very small, the point is at the camera center
        if point_3d.z < f64::EPSILON.sqrt() {
            return Err(CameraModelError::PointAtCameraCenter);
        }
        let u: f64 = self.intrinsics.fx * point_3d.x / point_3d.z + self.intrinsics.cx;
        let v: f64 = self.intrinsics.fy * point_3d.y / point_3d.z + self.intrinsics.cy;

        if u < 0.0
            || u >= self.resolution.width as f64
            || v < 0.0
            || v >= self.resolution.height as f64
        {
            return Err(CameraModelError::ProjectionOutSideImage);
        }

        Ok(Vector2::new(u, v))
    }

    /// Unprojects a 2D image point to a 3D ray in camera coordinates.
    ///
    /// This method applies the inverse pinhole camera equations to find a 3D direction vector:
    /// `mx = (u - cx) / fx`
    /// `my = (v - cy) / fy`
    /// The resulting 3D vector `(mx, my, 1.0)` is then normalized.
    ///
    /// # Arguments
    ///
    /// * `point_2d` - A `&Vector2<f64>` representing the 2D point (u, v) in pixel coordinates.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Vector3<f64>, CameraModelError>`.
    /// On success, it provides the 3D ray (`Vector3<f64>`) as a normalized direction vector
    /// originating from the camera center.
    ///
    /// # Errors
    ///
    /// * [`CameraModelError::PointIsOutSideImage`]: If the input 2D point is outside the camera's resolution.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::{DVector, Vector2};
    /// use fisheye_tools::camera::pinhole::PinholeModel;
    /// use fisheye_tools::camera::{CameraModel, Resolution};
    ///
    /// let params = DVector::from_vec(vec![500.0, 500.0, 320.0, 240.0]);
    /// let mut model = PinholeModel::new(&params).unwrap();
    /// model.resolution = Resolution { width: 640, height: 480 };
    ///
    /// let point_2d = Vector2::new(370.0, 340.0); // u, v in pixels
    /// match model.unproject(&point_2d) {
    ///     Ok(ray_3d) => {
    ///         println!("Unprojected ray: ({}, {}, {})", ray_3d.x, ray_3d.y, ray_3d.z);
    ///         // Expected mx = (370 - 320) / 500 = 50 / 500 = 0.1
    ///         // Expected my = (340 - 240) / 500 = 100 / 500 = 0.2
    ///         // Expected ray before normalization: (0.1, 0.2, 1.0)
    ///         // Normalizing (0.1, 0.2, 1.0) gives approx (0.098, 0.196, 0.976)
    ///     }
    ///     Err(e) => println!("Unprojection failed: {:?}", e),
    /// }
    /// ```
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        if point_2d.x < 0.0
            || point_2d.x >= self.resolution.width as f64
            || point_2d.y < 0.0
            || point_2d.y >= self.resolution.height as f64
        {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let mx: f64 = (point_2d.x - self.intrinsics.cx) / self.intrinsics.fx;
        let my: f64 = (point_2d.y - self.intrinsics.cy) / self.intrinsics.fy;

        let r2: f64 = mx * mx + my * my;

        let norm: f64 = (1.0 as f64 + r2).sqrt();
        let norm_inv: f64 = 1.0 as f64 / norm;

        Ok(Vector3::new(mx * norm_inv, my * norm_inv, norm_inv))
    }

    /// Loads camera parameters from a YAML file.
    ///
    /// The YAML file is expected to have a specific structure, typically including
    /// `cam0`, `intrinsics` (fx, fy, cx, cy), and `resolution` (width, height).
    ///
    /// # Arguments
    ///
    /// * `path` - A string slice representing the path to the YAML file.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Self, CameraModelError>`. On success, it provides an instance
    /// of [`PinholeModel`] populated with parameters from the file.
    ///
    /// # Errors
    ///
    /// This function can return various [`CameraModelError`] variants:
    /// * [`CameraModelError::IOError`]: If there's an issue reading the file.
    /// * [`CameraModelError::YamlError`]: If the YAML content is malformed or cannot be parsed.
    /// * [`CameraModelError::InvalidParams`]: If the YAML structure is missing expected fields
    ///   (e.g., "intrinsics", "resolution") or if parameter values are of incorrect types.
    /// * Errors from `validate_params` (e.g., [`CameraModelError::FocalLengthMustBePositive`])
    ///   if the loaded parameters are invalid.
    ///
    /// # Related
    /// * [`PinholeModel::save_to_yaml()`]
    fn load_from_yaml(path: &str) -> Result<Self, CameraModelError> {
        let contents = fs::read_to_string(path)?;
        let docs = YamlLoader::load_from_str(&contents)?;
        let doc = &docs[0];

        let intrinsics_yaml = doc["cam0"]["intrinsics"].as_vec().ok_or_else(|| {
            CameraModelError::InvalidParams("YAML missing 'intrinsics' or not an array".to_string())
        })?;
        let resolution_yaml = doc["cam0"]["resolution"].as_vec().ok_or_else(|| {
            CameraModelError::InvalidParams("YAML missing 'resolution' or not an array".to_string())
        })?;

        let intrinsics = Intrinsics {
            fx: intrinsics_yaml[0].as_f64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid fx: not a float".to_string())
            })?,
            fy: intrinsics_yaml[1].as_f64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid fy: not a float".to_string())
            })?,
            cx: intrinsics_yaml[2].as_f64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid cx: not a float".to_string())
            })?,
            cy: intrinsics_yaml[3].as_f64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid cy: not a float".to_string())
            })?,
        };

        let resolution = Resolution {
            width: resolution_yaml[0].as_i64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid width: not an integer".to_string())
            })? as u32,
            height: resolution_yaml[1].as_i64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid height: not an integer".to_string())
            })? as u32,
        };

        let model = PinholeModel {
            intrinsics,
            resolution,
        };

        // Validate parameters
        model.validate_params()?;

        Ok(model)
    }

    /// Saves the camera model's parameters to a YAML file.
    ///
    /// The output YAML file will include the camera model type ("pinhole"),
    /// intrinsic parameters (fx, fy, cx, cy), and resolution (width, height).
    ///
    /// # Arguments
    ///
    /// * `path` - A string slice representing the path to the YAML file where parameters will be saved.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<(), CameraModelError>`. `Ok(())` indicates success.
    ///
    /// # Errors
    ///
    /// This function can return:
    /// * [`CameraModelError::YamlError`]: If there's an issue serializing the data to YAML format.
    /// * [`CameraModelError::IOError`]: If there's an issue creating or writing to the file.
    ///
    /// # Related
    /// * [`PinholeModel::load_from_yaml()`]
    fn save_to_yaml(&self, path: &str) -> Result<(), CameraModelError> {
        // Create the YAML structure using serde_yaml
        let yaml = serde_yaml::to_value(&serde_yaml::Mapping::from_iter([(
            serde_yaml::Value::String("cam0".to_string()),
            serde_yaml::to_value(&serde_yaml::Mapping::from_iter([
                (
                    serde_yaml::Value::String("camera_model".to_string()),
                    serde_yaml::Value::String("pinhole".to_string()),
                ),
                (
                    serde_yaml::Value::String("intrinsics".to_string()),
                    serde_yaml::to_value(vec![
                        self.intrinsics.fx,
                        self.intrinsics.fy,
                        self.intrinsics.cx,
                        self.intrinsics.cy,
                    ])
                    .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
                ),
                (
                    serde_yaml::Value::String("resolution".to_string()),
                    serde_yaml::to_value(vec![self.resolution.width, self.resolution.height])
                        .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
                ),
            ]))
            .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
        )]))
        .map_err(|e| CameraModelError::YamlError(e.to_string()))?;

        // Convert to string
        let yaml_string =
            serde_yaml::to_string(&yaml).map_err(|e| CameraModelError::YamlError(e.to_string()))?;

        // Write to file
        let mut file =
            fs::File::create(path).map_err(|e| CameraModelError::IOError(e.to_string()))?;

        file.write_all(yaml_string.as_bytes())
            .map_err(|e| CameraModelError::IOError(e.to_string()))?;

        Ok(())
    }

    /// Validates the intrinsic parameters of the camera model.
    ///
    /// This method checks if the focal lengths (fx, fy) are positive and
    /// if the principal point coordinates (cx, cy) are finite numbers.
    /// It utilizes `validation::validate_intrinsics`.
    ///
    /// # Return Value
    ///
    /// Returns `Ok(())` if the parameters are valid.
    ///
    /// # Errors
    ///
    /// Returns a [`CameraModelError`] if validation fails, specifically:
    /// * [`CameraModelError::FocalLengthMustBePositive`]
    /// * [`CameraModelError::PrincipalPointMustBeFinite`]
    fn validate_params(&self) -> Result<(), CameraModelError> {
        validation::validate_intrinsics(&self.intrinsics)?;
        Ok(())
    }

    /// Returns a clone of the camera's resolution.
    ///
    /// # Return Value
    /// A [`Resolution`] struct containing the width and height of the camera image.
    fn get_resolution(&self) -> Resolution {
        self.resolution.clone()
    }

    /// Returns a clone of the camera's intrinsic parameters.
    ///
    /// # Return Value
    /// An [`Intrinsics`] struct containing fx, fy, cx, and cy.
    fn get_intrinsics(&self) -> Intrinsics {
        self.intrinsics.clone()
    }

    /// Returns the distortion parameters of the camera.
    ///
    /// For the Pinhole model, there are no distortion parameters.
    ///
    /// # Return Value
    /// An empty `Vec<f64>`.
    fn get_distortion(&self) -> Vec<f64> {
        vec![] // Pinhole model has no distortion parameters
    }

    // linear_estimation removed from impl CameraModel for PinholeModel
    // optimize removed from impl CameraModel for PinholeModel
}

/// Contains unit tests for the Pinhole camera model.
#[cfg(test)]
mod tests {
    use super::*;

    /// Tests loading [`PinholeModel`] parameters from a YAML file.
    #[test]
    fn test_pinhole_load_from_yaml() {
        let path = "samples/pinhole.yaml";
        let model = PinholeModel::load_from_yaml(path).unwrap();

        assert_eq!(model.intrinsics.fx, 461.629);
        assert_eq!(model.intrinsics.fy, 460.152);
        assert_eq!(model.intrinsics.cx, 362.680);
        assert_eq!(model.intrinsics.cy, 246.049);
        assert_eq!(model.resolution.width, 752);
        assert_eq!(model.resolution.height, 480);
    }

    /// Tests the projection and unprojection consistency of the [`PinholeModel`].
    #[test]
    fn test_pinhole_project_unproject() {
        let path = "samples/pinhole.yaml";
        let model = PinholeModel::load_from_yaml(path).unwrap();

        // 3D point in camera coordinates
        let point_3d = Vector3::new(1.0, 1.0, 5.0);
        let norm_3d = point_3d.normalize();

        // Project the 3D point to 2D
        let point_2d = model.project(&point_3d).unwrap();

        // Unproject the 2D point back to 3D
        let point_3d_unprojected = model.unproject(&point_2d).unwrap();

        // Check if the unprojected point is close to the original normalized point
        assert!((norm_3d.x - point_3d_unprojected.x).abs() < 1e-6);
        assert!((norm_3d.y - point_3d_unprojected.y).abs() < 1e-6);
        assert!((norm_3d.z - point_3d_unprojected.z).abs() < 1e-6);
    }
}
