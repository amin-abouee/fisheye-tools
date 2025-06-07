//! Extended Unified Camera Model (EUCM) Implementation
//!
//! This module implements the Extended Unified Camera Model (EUCM), which extends
//! the Unified Camera Model (UCM) with an additional parameter for better modeling
//! of wide-angle and fisheye cameras. The model uses two parameters (alpha, beta)
//! to handle the distortion characteristics of such cameras. It adheres to the
//! [`CameraModel`] trait defined in the parent `camera` module ([`crate::camera`]).
//!
//! # References
//!
//! The Extended Unified Camera Model is based on:
//! "A Generic Camera Model and Calibration Method for Conventional, Wide-Angle,
//! and Fish-Eye Lenses" by Bogdan Khomutenko, Gaëtan Garcia, and Philippe Martinet.

use crate::camera::{validation, CameraModel, CameraModelError, Intrinsics, Resolution};
use log::info;
use nalgebra::{DVector, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::io::Write;
use yaml_rust::YamlLoader;

/// Implements the Extended Unified Camera Model (EUCM) for wide-angle/fisheye lenses.
///
/// The EUCM model is designed for cameras with significant distortion,
/// common in wide-angle or fisheye lenses. It represents the camera using
/// standard intrinsic parameters ([`Intrinsics`]: fx, fy, cx, cy), image [`Resolution`],
/// and two distortion parameters: `alpha` and `beta`.
/// `alpha` and `beta` control the projection onto a unit sphere.
///
/// # Fields
///
/// *   `intrinsics`: [`Intrinsics`] - Holds the focal lengths (fx, fy) and principal point (cx, cy).
/// *   `resolution`: [`Resolution`] - The width and height of the camera image in pixels.
/// *   `alpha`: `f64` - The first distortion parameter, controlling the sphere projection.
///     It must be a finite number.
/// *   `beta`: `f64` - The second distortion parameter, extending the UCM model.
///     It must be a finite number.
///
/// # References
///
/// *   Khomutenko, B., Garcia, G., & Martinet, P. (2015). A generic camera model and
///     calibration method for conventional, wide-angle, and fish-eye lenses.
///     In *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
///
/// # Examples
///
/// ```rust
/// use nalgebra::DVector;
/// use fisheye_tools::camera::eucm::EucmModel;
/// use fisheye_tools::camera::{Intrinsics, Resolution, CameraModel, CameraModelError};
///
/// // Parameters: fx, fy, cx, cy, alpha, beta
/// let params = DVector::from_vec(vec![1313.83, 1313.27, 960.471, 546.981, 1.01674, 0.5]);
/// let mut eucm_model = EucmModel::new(&params).unwrap();
/// eucm_model.resolution = Resolution { width: 752, height: 480 };
///
/// println!("Created EUCM model: {:?}", eucm_model);
/// assert_eq!(eucm_model.intrinsics.fx, 1313.83);
/// assert_eq!(eucm_model.alpha, 1.01674);
/// assert_eq!(eucm_model.beta, 0.5);
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct EucmModel {
    /// Camera intrinsic parameters: `fx`, `fy`, `cx`, `cy`.
    pub intrinsics: Intrinsics,
    /// Image resolution as width and height in pixels.
    pub resolution: Resolution,
    /// First distortion parameter, controlling the sphere projection.
    /// Must be a finite `f64` value.
    pub alpha: f64,
    /// Second distortion parameter, extending the UCM model.
    /// Must be a finite `f64` value.
    pub beta: f64,
}

impl EucmModel {
    /// Creates a new [`EucmModel`] instance from a parameter vector.
    ///
    /// This constructor initializes the EUCM camera model using the provided parameters.
    /// The parameter vector should contain exactly 6 elements in the following order:
    /// `[fx, fy, cx, cy, alpha, beta]`.
    ///
    /// # Arguments
    ///
    /// * `parameters`: A `&DVector<f64>` containing the camera parameters.
    ///   - `parameters[0]`: `fx` - Focal length in the x-direction (pixels).
    ///   - `parameters[1]`: `fy` - Focal length in the y-direction (pixels).
    ///   - `parameters[2]`: `cx` - Principal point x-coordinate (pixels).
    ///   - `parameters[3]`: `cy` - Principal point y-coordinate (pixels).
    ///   - `parameters[4]`: `alpha` - First distortion parameter.
    ///   - `parameters[5]`: `beta` - Second distortion parameter.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Self, CameraModelError>`. Currently, this function
    /// always returns `Ok(Self)` as no validation that can fail is performed within `new` itself.
    ///
    /// # Panics
    ///
    /// This function will panic if `parameters.len()` is less than 6, due to direct
    /// indexing (`parameters[0]` through `parameters[5]`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::DVector;
    /// use fisheye_tools::camera::eucm::EucmModel;
    /// use fisheye_tools::camera::Resolution;
    ///
    /// let params_vec = DVector::from_vec(vec![
    ///     1313.83, // fx
    ///     1313.27, // fy
    ///     960.471, // cx
    ///     546.981, // cy
    ///     1.01674, // alpha
    ///     0.5      // beta
    /// ]);
    /// let mut model = EucmModel::new(&params_vec).unwrap();
    /// model.resolution = Resolution { width: 752, height: 480 }; // Set resolution manually
    ///
    /// assert_eq!(model.intrinsics.fx, 1313.83);
    /// assert_eq!(model.alpha, 1.01674);
    /// assert_eq!(model.beta, 0.5);
    /// assert_eq!(model.resolution.width, 752);
    /// ```
    pub fn new(parameters: &DVector<f64>) -> Result<Self, CameraModelError> {
        let model = EucmModel {
            intrinsics: Intrinsics {
                fx: parameters[0],
                fy: parameters[1],
                cx: parameters[2],
                cy: parameters[3],
            },
            resolution: Resolution {
                width: 0, // Resolution is typically set after creation or by loading.
                height: 0,
            },
            alpha: parameters[4],
            beta: parameters[5],
        };

        info!("new EUCM model is: {:?}", model);
        Ok(model)
    }

    /// Checks the geometric condition for a valid projection in the EUCM model.
    ///
    /// This helper function determines if a 3D point can be validly projected
    /// based on its z-coordinate (`z`), the denominator (`denom`), and the model's
    /// `alpha` and `beta` parameters.
    ///
    /// # Arguments
    ///
    /// * `z`: `f64` - The Z-coordinate of the 3D point in camera space.
    /// * `denom`: `f64` - The denominator from the projection equation.
    /// * `alpha`: `f64` - The first distortion parameter.
    /// * `beta`: `f64` - The second distortion parameter.
    ///
    /// # Return Value
    ///
    /// Returns `true` if the point satisfies the projection condition, `false` otherwise.
    pub fn check_proj_condition(z: f64, denom: f64, alpha: f64, _beta: f64) -> bool {
        let mut condition = true;
        if alpha > 0.5 {
            let _zn = z / denom;
            let c = (alpha - 1.0) / (2.0 * alpha - 1.0);
            if z < denom * c {
                condition = false;
            }
        }
        condition
    }

    /// Checks the geometric condition for a valid unprojection in the EUCM model.
    ///
    /// This private helper function determines if a 2D point (represented by its
    /// squared radial distance from the principal point) can be validly unprojected.
    /// The condition depends on the model's `alpha` and `beta` parameters.
    ///
    /// # Arguments
    ///
    /// * `r_squared`: `f64` - The squared radial distance of the normalized 2D point from the principal point.
    /// * `alpha`: `f64` - The first distortion parameter.
    /// * `beta`: `f64` - The second distortion parameter.
    ///
    /// # Return Value
    ///
    /// Returns `true` if the point satisfies the unprojection condition, `false` otherwise.
    fn check_unproj_condition(r_squared: f64, alpha: f64, beta: f64) -> bool {
        let mut condition = true;
        if alpha > 0.5 && r_squared > (1.0 / beta * (2.0 * alpha - 1.0)) {
            condition = false;
        }
        condition
    }
}

/// Provides a debug string representation for [`EucmModel`].
impl fmt::Debug for EucmModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EUCM [fx: {} fy: {} cx: {} cy: {} alpha: {} beta: {}]",
            self.intrinsics.fx,
            self.intrinsics.fy,
            self.intrinsics.cx,
            self.intrinsics.cy,
            self.alpha,
            self.beta
        )
    }
}

impl CameraModel for EucmModel {
    /// Projects a 3D point from camera coordinates to 2D image coordinates.
    ///
    /// This method applies the EUCM projection equations. It first checks
    /// if the point is projectable using the EUCM geometric constraints.
    /// If valid, it computes the normalized image coordinates and then
    /// scales them by the focal lengths and adds the principal point offsets.
    ///
    /// # Arguments
    ///
    /// * `point_3d`: A `&Vector3<f64>` representing the 3D point (X, Y, Z) in camera coordinates.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Vector2<f64>, CameraModelError>`.
    /// On success, it provides the projected 2D point (`Vector2<f64>`) in pixel coordinates (u, v).
    ///
    /// # Errors
    ///
    /// * [`CameraModelError::PointIsOutSideImage`]: If the 3D point cannot be validly projected
    ///   according to the EUCM model's geometric constraints.
    fn project(&self, point_3d: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        const PRECISION: f64 = 1e-3;

        let x = point_3d.x;
        let y = point_3d.y;
        let z = point_3d.z;

        let d = (self.beta * (x * x + y * y) + z * z).sqrt();
        let denom = self.alpha * d + (1.0 - self.alpha) * z;

        // Check if the projection is valid
        if denom < PRECISION || !Self::check_proj_condition(z, denom, self.alpha, self.beta) {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let projected_x = self.intrinsics.fx * (x / denom) + self.intrinsics.cx;
        let projected_y = self.intrinsics.fy * (y / denom) + self.intrinsics.cy;

        Ok(Vector2::new(projected_x, projected_y))
    }

    /// Unprojects a 2D image point to a 3D ray in camera coordinates.
    ///
    /// This method applies the inverse EUCM model equations to convert
    /// a 2D pixel coordinate back into a 3D direction vector (ray) originating
    /// from the camera center. The resulting vector is normalized.
    ///
    /// # Arguments
    ///
    /// * `point_2d`: A `&Vector2<f64>` representing the 2D point (u, v) in pixel coordinates.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Vector3<f64>, CameraModelError>`.
    /// On success, it provides the normalized 3D ray (`Vector3<f64>`) corresponding to the 2D point.
    ///
    /// # Errors
    ///
    /// * [`CameraModelError::PointIsOutSideImage`]: If the 2D point cannot be validly unprojected
    ///   according to the EUCM model's geometric constraints.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        const PRECISION: f64 = 1e-3;

        let fx = self.intrinsics.fx;
        let fy = self.intrinsics.fy;
        let cx = self.intrinsics.cx;
        let cy = self.intrinsics.cy;
        let alpha = self.alpha;
        let beta = self.beta;
        let u = point_2d.x;
        let v = point_2d.y;

        let mx = (u - cx) / fx;
        let my = (v - cy) / fy;

        let r_squared = mx * mx + my * my;
        let gamma = 1.0 - alpha;
        let num = 1.0 - r_squared * alpha * alpha * beta;
        let det = 1.0 - (alpha - gamma) * beta * r_squared;
        let denom = gamma + alpha * det.sqrt();

        // Check if we can unproject this point
        if det < PRECISION || !Self::check_unproj_condition(r_squared, alpha, beta) {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let mz = num / denom;
        let norm = (mx * mx + my * my + mz * mz).sqrt();

        Ok(Vector3::new(mx / norm, my / norm, mz / norm))
    }

    /// Loads [`EucmModel`] parameters from a YAML file.
    ///
    /// The YAML file is expected to follow a structure where camera parameters are nested
    /// under `cam0`. The intrinsic parameters (`fx`, `fy`, `cx`, `cy`) and the EUCM
    /// specific distortion parameters (`alpha`, `beta`) are typically grouped
    /// together in an `intrinsics` array in the YAML file: `[fx, fy, cx, cy, alpha, beta]`.
    /// The `resolution` (width, height) is also expected under `cam0`.
    ///
    /// # Arguments
    ///
    /// * `path`: A string slice representing the path to the YAML file.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Self, CameraModelError>`. On success, it provides an instance
    /// of [`EucmModel`] populated with parameters from the file.
    ///
    /// # Errors
    ///
    /// This function can return:
    /// * [`CameraModelError::IOError`]: If there's an issue reading the file.
    /// * [`CameraModelError::YamlError`]: If the YAML content is malformed or cannot be parsed.
    /// * [`CameraModelError::InvalidParams`]: If the YAML structure is missing expected fields
    ///   or if parameter values are of incorrect types or counts.
    fn load_from_yaml(path: &str) -> Result<Self, CameraModelError> {
        let contents = fs::read_to_string(path)?;
        let docs = YamlLoader::load_from_str(&contents)?;

        if docs.is_empty() {
            return Err(CameraModelError::InvalidParams(
                "Empty YAML document".to_string(),
            ));
        }

        let doc = &docs[0];

        let intrinsics_yaml_vec = doc["cam0"]["intrinsics"].as_vec().ok_or_else(|| {
            CameraModelError::InvalidParams(
                "YAML missing 'intrinsics' array under 'cam0'".to_string(),
            )
        })?;
        let resolution_yaml_vec = doc["cam0"]["resolution"].as_vec().ok_or_else(|| {
            CameraModelError::InvalidParams(
                "YAML missing 'resolution' array under 'cam0'".to_string(),
            )
        })?;

        if intrinsics_yaml_vec.len() < 6 {
            return Err(CameraModelError::InvalidParams(
                "Intrinsics array in YAML must have at least 6 elements (fx, fy, cx, cy, alpha, beta)".to_string()
            ));
        }

        let alpha = intrinsics_yaml_vec[4].as_f64().ok_or_else(|| {
            CameraModelError::InvalidParams("Invalid alpha in YAML: not a float".to_string())
        })?;

        let beta = intrinsics_yaml_vec[5].as_f64().ok_or_else(|| {
            CameraModelError::InvalidParams("Invalid beta in YAML: not a float".to_string())
        })?;

        let intrinsics = Intrinsics {
            fx: intrinsics_yaml_vec[0].as_f64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid fx in YAML: not a float".to_string())
            })?,
            fy: intrinsics_yaml_vec[1].as_f64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid fy in YAML: not a float".to_string())
            })?,
            cx: intrinsics_yaml_vec[2].as_f64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid cx in YAML: not a float".to_string())
            })?,
            cy: intrinsics_yaml_vec[3].as_f64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid cy in YAML: not a float".to_string())
            })?,
        };

        if resolution_yaml_vec.len() < 2 {
            return Err(CameraModelError::InvalidParams(
                "Resolution array in YAML must have at least 2 elements (width, height)"
                    .to_string(),
            ));
        }
        let resolution = Resolution {
            width: resolution_yaml_vec[0].as_i64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid width in YAML: not an integer".to_string())
            })? as u32,
            height: resolution_yaml_vec[1].as_i64().ok_or_else(|| {
                CameraModelError::InvalidParams(
                    "Invalid height in YAML: not an integer".to_string(),
                )
            })? as u32,
        };

        let model = EucmModel {
            intrinsics,
            resolution,
            alpha,
            beta,
        };

        // Validate parameters
        model.validate_params()?;
        Ok(model)
    }

    /// Saves the [`EucmModel`] parameters to a YAML file.
    ///
    /// The parameters are saved under the `cam0` key. The `intrinsics` YAML array
    /// will contain `fx, fy, cx, cy, alpha, beta` in that order.
    ///
    /// # Arguments
    ///
    /// * `path`: A string slice representing the path to the YAML file where parameters will be saved.
    ///
    /// # Return Value
    ///
    /// Returns `Ok(())` on successful save.
    ///
    /// # Errors
    ///
    /// This function can return:
    /// * [`CameraModelError::YamlError`]: If there's an issue serializing the data to YAML format.
    /// * [`CameraModelError::IOError`]: If there's an issue creating or writing to the file.
    fn save_to_yaml(&self, path: &str) -> Result<(), CameraModelError> {
        // Create the YAML structure using serde_yaml
        let yaml = serde_yaml::to_value(serde_yaml::Mapping::from_iter([(
            serde_yaml::Value::String("cam0".to_string()),
            serde_yaml::to_value(serde_yaml::Mapping::from_iter([
                (
                    serde_yaml::Value::String("camera_model".to_string()),
                    serde_yaml::Value::String("eucm".to_string()),
                ),
                (
                    serde_yaml::Value::String("intrinsics".to_string()),
                    serde_yaml::to_value(vec![
                        self.intrinsics.fx,
                        self.intrinsics.fy,
                        self.intrinsics.cx,
                        self.intrinsics.cy,
                        self.alpha,
                        self.beta,
                    ])
                    .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
                ),
                (
                    serde_yaml::Value::String("rostopic".to_string()),
                    serde_yaml::Value::String("/cam0/image_raw".to_string()),
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

    /// Validates the parameters of the [`EucmModel`].
    ///
    /// This method checks the validity of the core intrinsic parameters (focal lengths,
    /// principal point) using [`validation::validate_intrinsics`]. It also validates
    /// the EUCM specific parameters:
    /// *   `alpha` must be a finite `f64` value (not NaN or infinity).
    /// *   `beta` must be a finite `f64` value (not NaN or infinity).
    ///
    /// # Return Value
    ///
    /// Returns `Ok(())` if all parameters are valid.
    ///
    /// # Errors
    ///
    /// Returns a [`CameraModelError`] if any parameter is invalid:
    /// * [`CameraModelError::InvalidParams`]: If `alpha` or `beta` is not a finite number.
    /// * Errors propagated from [`validation::validate_intrinsics`].
    fn validate_params(&self) -> Result<(), CameraModelError> {
        validation::validate_intrinsics(&self.intrinsics)?;

        if !self.alpha.is_finite() {
            return Err(CameraModelError::InvalidParams(
                "alpha must be finite".to_string(),
            ));
        }

        if !self.beta.is_finite() {
            return Err(CameraModelError::InvalidParams(
                "beta must be finite".to_string(),
            ));
        }

        Ok(())
    }

    /// Returns a clone of the camera's image resolution.
    fn get_resolution(&self) -> Resolution {
        self.resolution.clone()
    }

    /// Returns a clone of the camera's intrinsic parameters.
    fn get_intrinsics(&self) -> Intrinsics {
        self.intrinsics.clone()
    }

    /// Returns the distortion parameters of the EUCM model.
    ///
    /// The parameters are returned as a vector containing `[alpha, beta]`.
    ///
    /// # Return Value
    ///
    /// A `Vec<f64>` containing the distortion parameters: `[alpha, beta]`.
    fn get_distortion(&self) -> Vec<f64> {
        vec![self.alpha, self.beta]
    }
}

/// Unit tests for the [`EucmModel`].
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq; // For floating point comparisons

    /// Helper function to create a sample [`EucmModel`] instance for testing.
    /// This model is based on typical EUCM parameters.
    fn get_sample_model() -> EucmModel {
        EucmModel {
            intrinsics: Intrinsics {
                fx: 1313.83,
                fy: 1313.27,
                cx: 960.471,
                cy: 546.981,
            },
            resolution: Resolution {
                width: 752,
                height: 480,
            },
            alpha: 1.01674,
            beta: 0.5,
        }
    }

    /// Tests the consistency of projection and unprojection for the [`EucmModel`].
    #[test]
    fn test_eucm_project_unproject() {
        let model = get_sample_model();

        // Create a 3D point in camera coordinates - use a point closer to the center
        let point_3d = Vector3::new(0.1, 0.1, 3.0);
        let norm_3d = point_3d.normalize();

        // Project the 3D point to pixel coordinates
        let point_2d = model.project(&point_3d).unwrap();

        // Debug print the projected coordinates
        println!("Projected point: ({}, {})", point_2d.x, point_2d.y);
        println!(
            "Image bounds: {}x{}",
            model.resolution.width, model.resolution.height
        );

        // Check if the pixel coordinates are finite
        assert!(point_2d.x.is_finite() && point_2d.y.is_finite());

        // Unproject the pixel point back to a 3D ray direction
        let point_3d_unprojected = model.unproject(&point_2d).unwrap();

        // Check if the unprojected point (normalized ray) is close to the original normalized point
        // Use a more relaxed epsilon for EUCM due to numerical precision in the complex equations
        assert_relative_eq!(norm_3d.x, point_3d_unprojected.x, epsilon = 1e-4);
        assert_relative_eq!(norm_3d.y, point_3d_unprojected.y, epsilon = 1e-4);
        assert_relative_eq!(norm_3d.z, point_3d_unprojected.z, epsilon = 1e-4);
    }

    /// Tests projection and unprojection with a point near the image center.
    #[test]
    fn test_eucm_project_unproject_center() {
        let model = get_sample_model();

        // Use a point that should project near the image center
        let point_3d = Vector3::new(0.0, 0.0, 1.0);
        let norm_3d = point_3d.normalize();

        // Project the 3D point to pixel coordinates
        let point_2d = model.project(&point_3d).unwrap();

        // Should project near the principal point
        assert_relative_eq!(point_2d.x, model.intrinsics.cx, epsilon = 1.0);
        assert_relative_eq!(point_2d.y, model.intrinsics.cy, epsilon = 1.0);

        // Unproject back
        let point_3d_unprojected = model.unproject(&point_2d).unwrap();

        // Check round-trip consistency
        assert_relative_eq!(norm_3d.x, point_3d_unprojected.x, epsilon = 1e-6);
        assert_relative_eq!(norm_3d.y, point_3d_unprojected.y, epsilon = 1e-6);
        assert_relative_eq!(norm_3d.z, point_3d_unprojected.z, epsilon = 1e-6);
    }

    /// Tests projection of points near or at the camera center.
    #[test]
    fn test_project_point_at_center() {
        let model = get_sample_model();
        // Point very close to origin on Z axis
        let point_3d_on_z = Vector3::new(0.0, 0.0, 1e-9);
        let result_origin = model.project(&point_3d_on_z);

        // For EUCM, if `denom` becomes too small or `check_proj_condition` fails,
        // it should return `PointIsOutSideImage`. Otherwise, it might project near (cx, cy).
        if let Ok(p) = result_origin {
            assert_relative_eq!(p.x, model.intrinsics.cx, epsilon = 1e-3);
            assert_relative_eq!(p.y, model.intrinsics.cy, epsilon = 1e-3);
        } else {
            assert!(matches!(
                result_origin,
                Err(CameraModelError::PointIsOutSideImage)
            ));
        }

        // Point exactly at (0,0,0)
        let result_exact_origin = model.project(&Vector3::new(0.0, 0.0, 0.0));
        // This should typically result in an error due to d=0, leading to small denom or failed check.
        assert!(
            matches!(
                result_exact_origin,
                Err(CameraModelError::PointIsOutSideImage)
            ),
            "Projecting (0,0,0) should result in PointIsOutSideImage or similar error."
        );
    }

    /// Tests projection of a point located behind the camera.
    #[test]
    fn test_project_point_behind_camera() {
        let model = get_sample_model();
        let point_3d = Vector3::new(0.1, 0.2, -1.0); // Point behind camera
        let result = model.project(&point_3d);
        // Expect an error as the point is behind the camera and likely fails check_proj_condition.
        assert!(matches!(result, Err(CameraModelError::PointIsOutSideImage)));
    }

    /// Tests `validate_params` with a valid model.
    #[test]
    fn test_validate_params_valid() {
        let model = get_sample_model();
        assert!(model.validate_params().is_ok());
    }

    /// Tests `validate_params` with invalid `alpha` values (NaN, Infinity).
    #[test]
    fn test_validate_params_invalid_alpha() {
        let mut model = get_sample_model();
        model.alpha = f64::NAN;
        assert!(matches!(
            model.validate_params(),
            Err(CameraModelError::InvalidParams(msg)) if msg == "alpha must be finite"
        ));

        model.alpha = f64::INFINITY;
        assert!(matches!(
            model.validate_params(),
            Err(CameraModelError::InvalidParams(msg)) if msg == "alpha must be finite"
        ));
    }

    /// Tests `validate_params` with invalid `beta` values (NaN, Infinity).
    #[test]
    fn test_validate_params_invalid_beta() {
        let mut model = get_sample_model();
        model.beta = f64::NAN;
        assert!(matches!(
            model.validate_params(),
            Err(CameraModelError::InvalidParams(msg)) if msg == "beta must be finite"
        ));

        model.beta = f64::INFINITY;
        assert!(matches!(
            model.validate_params(),
            Err(CameraModelError::InvalidParams(msg)) if msg == "beta must be finite"
        ));
    }

    /// Tests `validate_params` with invalid core intrinsic parameters.
    #[test]
    fn test_validate_params_invalid_intrinsics() {
        let mut model = get_sample_model();
        model.intrinsics.fx = 0.0; // Invalid: fx must be > 0
        assert!(matches!(
            model.validate_params(),
            Err(CameraModelError::FocalLengthMustBePositive)
        ));
    }

    /// Tests the getter methods: `get_intrinsics`, `get_resolution`, and `get_distortion`.
    #[test]
    fn test_getters() {
        let model = get_sample_model();
        let intrinsics = model.get_intrinsics();
        assert_relative_eq!(intrinsics.fx, model.intrinsics.fx);
        assert_relative_eq!(intrinsics.fy, model.intrinsics.fy);
        assert_relative_eq!(intrinsics.cx, model.intrinsics.cx);
        assert_relative_eq!(intrinsics.cy, model.intrinsics.cy);

        let resolution = model.get_resolution();
        assert_eq!(resolution.width, model.resolution.width);
        assert_eq!(resolution.height, model.resolution.height);

        let distortion = model.get_distortion();
        assert_eq!(distortion.len(), 2);
        assert_relative_eq!(distortion[0], model.alpha);
        assert_relative_eq!(distortion[1], model.beta);
    }
}
