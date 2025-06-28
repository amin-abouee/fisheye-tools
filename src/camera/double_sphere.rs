//! Double Sphere Camera Model Implementation
//!
//! This module implements the Double Sphere camera model, which is particularly useful
//! for wide-angle and fisheye cameras. The model uses two sphere projections to handle
//! the distortion characteristics of such cameras. It adheres to the [`CameraModel`]
//! trait defined in the parent `camera` module ([`crate::camera`]).
//!
//! # References
//!
//! The Double Sphere model is based on:
//! "The Double Sphere Camera Model" by Vladyslav Usenko and Nikolaus Demmel.

use crate::camera::{validation, CameraModel, CameraModelError, Intrinsics, Resolution};
use log::info;
use nalgebra::{DVector, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use std::{fmt, fs, io::Write};
use yaml_rust::YamlLoader;

/// Implements the Double Sphere camera model for wide-angle/fisheye lenses.
///
/// The Double Sphere model is designed for cameras with significant distortion,
/// common in wide-angle or fisheye lenses. It represents the camera using
/// standard intrinsic parameters ([`Intrinsics`]: fx, fy, cx, cy), image [`Resolution`],
/// and two special distortion parameters: `alpha` and `xi`.
/// `alpha` controls the transition between two conceptual spheres used in the projection,
/// and `xi` represents the displacement between their centers.
///
/// # Fields
///
/// *   `intrinsics`: [`Intrinsics`] - Holds the focal lengths (fx, fy) and principal point (cx, cy).
/// *   `resolution`: [`Resolution`] - The width and height of the camera image in pixels.
/// *   `alpha`: `f64` - The first distortion parameter, controlling the blend between the
///     two sphere projections. It must be in the range (0, 1] (i.e., `0 < alpha <= 1`).
/// *   `xi`: `f64` - The second distortion parameter, representing the displacement
///     between the centers of the two spheres. It must be a finite number.
///
/// # References
///
/// *   Usenko, V., Demmel, N., & Cremers, D. (2018). The Double Sphere Camera Model.
///     In *2018 International Conference on 3D Vision (3DV)*. IEEE.
///
/// # Examples
///
/// ```rust
/// use nalgebra::DVector;
/// use fisheye_tools::camera::double_sphere::DoubleSphereModel;
/// use fisheye_tools::camera::{Intrinsics, Resolution, CameraModel, CameraModelError};
///
/// // Parameters: fx, fy, cx, cy, alpha, xi
/// let params = DVector::from_vec(vec![350.0, 350.0, 320.0, 240.0, 0.58, -0.18]);
/// let mut ds_model = DoubleSphereModel::new(&params).unwrap();
/// ds_model.resolution = Resolution { width: 640, height: 480 };
///
/// println!("Created Double Sphere model: {:?}", ds_model);
/// assert_eq!(ds_model.intrinsics.fx, 350.0);
/// assert_eq!(ds_model.alpha, 0.58);
///
/// // Example of loading from a (hypothetical) YAML - actual loading depends on file content
/// // let model_from_yaml = DoubleSphereModel::load_from_yaml("path/to/your_ds_camera.yaml");
/// // if let Ok(model) = model_from_yaml {
/// //     println!("Loaded model: {:?}", model.get_intrinsics());
/// // }
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct DoubleSphereModel {
    /// Camera intrinsic parameters: `fx`, `fy`, `cx`, `cy`.
    pub intrinsics: Intrinsics,
    /// Image resolution as width and height in pixels.
    pub resolution: Resolution,
    /// First distortion parameter, controlling the transition between the two spheres.
    /// Must be in the range (0, 1] (i.e., `0.0 < alpha <= 1.0`).
    pub alpha: f64,
    /// Second distortion parameter, representing the displacement between the centers of the two spheres.
    /// Must be a finite `f64` value.
    pub xi: f64,
}

impl DoubleSphereModel {
    /// Creates a new [`DoubleSphereModel`] from a DVector of parameters.
    ///
    /// This constructor initializes the model with the provided parameters.
    /// The image resolution is initialized to 0x0 and should be set explicitly
    /// or by loading from a configuration file like YAML.
    ///
    /// Note: The current implementation does not validate the length of the `parameters`
    /// vector or the validity of `alpha` and `xi` during construction (though `validate_params`
    /// can be called separately). Direct indexing is used.
    ///
    /// # Arguments
    ///
    /// * `parameters`: A `&DVector<f64>` containing the camera parameters in the following order:
    ///   1.  `fx`: Focal length along the x-axis.
    ///   2.  `fy`: Focal length along the y-axis.
    ///   3.  `cx`: Principal point x-coordinate.
    ///   4.  `cy`: Principal point y-coordinate.
    ///   5.  `alpha`: The first distortion parameter.
    ///   6.  `xi`: The second distortion parameter.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Self, CameraModelError>`. In the current implementation, this
    /// always returns `Ok(Self)` as no validation that can fail is performed within `new` itself.
    /// However, it retains the `Result` type for future compatibility or if internal
    /// validation were added.
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
    /// use fisheye_tools::camera::double_sphere::DoubleSphereModel;
    /// use fisheye_tools::camera::Resolution;
    ///
    /// let params_vec = DVector::from_vec(vec![
    ///     348.11, // fx
    ///     347.11, // fy
    ///     365.81, // cx
    ///     249.35, // cy
    ///     0.56,   // alpha
    ///     -0.24   // xi
    /// ]);
    /// let mut model = DoubleSphereModel::new(&params_vec).unwrap();
    /// model.resolution = Resolution { width: 752, height: 480 }; // Set resolution manually
    ///
    /// assert_eq!(model.intrinsics.fx, 348.11);
    /// assert_eq!(model.alpha, 0.56);
    /// assert_eq!(model.resolution.width, 752);
    /// ```
    pub fn new(parameters: &DVector<f64>) -> Result<Self, CameraModelError> {
        let model = DoubleSphereModel {
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
            xi: parameters[5],
        };

        // model.validate_params()?; // Original code has this commented out.
        // Documenting current behavior: validate_params is not called here.
        info!("new model is: {model:?}");
        Ok(model)
    }

    /// Checks the geometric condition for a valid projection in the Double Sphere model.
    ///
    /// This private helper function determines if a 3D point can be validly projected
    /// based on its z-coordinate (`z`), its distance from the origin (`d1`),
    /// and the model's `alpha` and `xi` parameters. The condition ensures that
    /// the point is in front of a plane defined by the model's geometry.
    ///
    /// # Arguments
    ///
    /// * `z`: `f64` - The Z-coordinate of the 3D point in camera space.
    /// * `d1`: `f64` - The Euclidean distance of the 3D point from the camera origin.
    ///
    /// # Return Value
    ///
    /// Returns `true` if the point satisfies the projection condition, `false` otherwise.
    fn check_projection_condition(&self, z: f64, d1: f64) -> bool {
        let w1 = match self.alpha <= 0.5 {
            true => self.alpha / (1.0 - self.alpha),
            false => (1.0 - self.alpha) / self.alpha,
        };
        let w2 = (w1 + self.xi) / (2.0 * w1 * self.xi + self.xi * self.xi + 1.0).sqrt();
        z > -w2 * d1
    }

    /// Checks the geometric condition for a valid unprojection in the Double Sphere model.
    ///
    /// This private helper function determines if a 2D point (represented by its
    /// squared radial distance from the principal point) can be validly unprojected.
    /// The condition depends on the model's `alpha` parameter and is relevant when
    /// `alpha > 0.5`.
    ///
    /// # Arguments
    ///
    /// * `r_squared`: `f64` - The squared radial distance of the normalized 2D point from the principal point.
    ///
    /// # Return Value
    ///
    /// Returns `true` if the point satisfies the unprojection condition, `false` otherwise.
    fn check_unprojection_condition(&self, r_squared: f64) -> bool {
        let mut condition = true;
        if self.alpha > 0.5 {
            // If alpha > 0.5, the point must be within a certain radius for unprojection to be valid.
            if r_squared > 1.0 / (2.0 * self.alpha - 1.0) {
                condition = false;
            }
        }
        condition
    }
}

/// Provides a debug string representation for [`DoubleSphereModel`].
impl fmt::Debug for DoubleSphereModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DoubleSphere [fx: {} fy: {} cx: {} cy: {} alpha: {} xi: {}]",
            self.intrinsics.fx,
            self.intrinsics.fy,
            self.intrinsics.cx,
            self.intrinsics.cy,
            self.alpha,
            self.xi
        )
    }
}

impl CameraModel for DoubleSphereModel {
    /// Projects a 3D point from camera coordinates to 2D image coordinates.
    ///
    /// This method applies the Double Sphere projection equations. It first checks
    /// if the point is projectable using [`DoubleSphereModel::check_projection_condition`].
    /// If valid, it computes the normalized image coordinates (mx, my) and then
    /// scales them by the focal lengths and adds the principal point offsets.
    /// The Jacobian of the projection function can optionally be computed.
    ///
    /// # Arguments
    ///
    /// * `point_3d`: A `&Vector3<f64>` representing the 3D point (X, Y, Z) in camera coordinates.
    /// * `compute_jacobian`: A boolean flag. If true, the Jacobian of the projection
    ///   with respect to the camera parameters (fx, fy, cx, cy, alpha, xi) is computed.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<(Vector2<f64>, Option<DMatrix<f64>>), CameraModelError>`.
    /// On success, it provides a tuple containing:
    /// *   The projected 2D point (`Vector2<f64>`) in pixel coordinates (u, v).
    /// *   An `Option<DMatrix<f64>>` which is `Some(jacobian)` if `compute_jacobian` was true,
    ///     or `None` otherwise. The Jacobian matrix is 2x6, representing the partial
    ///     derivatives of the projected (u,v) coordinates with respect to
    ///     (fx, fy, cx, cy, alpha, xi).
    ///
    /// # Errors
    ///
    /// * [`CameraModelError::PointIsOutSideImage`]: If the 3D point cannot be validly projected
    ///   according to the Double Sphere model's geometric constraints (e.g., point is behind
    ///   the valid projection plane or results in a denominator close to zero).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::{DVector, Vector3};
    /// use fisheye_tools::camera::double_sphere::DoubleSphereModel;
    /// use fisheye_tools::camera::{CameraModel, Resolution, CameraModelError};
    ///
    /// let params = DVector::from_vec(vec![350.0, 350.0, 320.0, 240.0, 0.58, -0.18]);
    /// let mut model = DoubleSphereModel::new(&params).unwrap();
    /// model.resolution = Resolution { width: 640, height: 480 };
    ///
    /// let point_3d = Vector3::new(0.1, 0.2, 1.0); // X, Y, Z in meters
    /// match model.project(&point_3d) {
    ///     Ok(point_2d) => {
    ///         println!("Projected point: ({}, {})", point_2d.x, point_2d.y);
    ///         // Expected values would depend on the exact DS parameters and equations
    ///         assert!(point_2d.x > 0.0 && point_2d.y > 0.0);
    ///     }
    ///     Err(e) => println!("Projection failed: {:?}", e),
    /// }
    /// ```
    fn project(&self, point_3d: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        const PRECISION: f64 = 1e-3;

        let x = point_3d.x;
        let y = point_3d.y;
        let z = point_3d.z;

        let r_squared = (x * x) + (y * y);
        let d1 = (r_squared + (z * z)).sqrt();
        let gamma = self.xi * d1 + z; // Note: Original paper might use 'zeta' for xi.
        let d2 = (r_squared + gamma * gamma).sqrt();

        let denom = self.alpha * d2 + (1.0 - self.alpha) * gamma;

        // Check if the projection is valid
        if denom < PRECISION || !self.check_projection_condition(z, d1) {
            // This error indicates the point is outside the valid projection area
            // or results in an unstable projection.
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let mx = x / denom;
        let my = y / denom;

        // Project the point
        let projected_x = self.intrinsics.fx * (mx) + self.intrinsics.cx;
        let projected_y = self.intrinsics.fy * (my) + self.intrinsics.cy;

        Ok(Vector2::new(projected_x, projected_y))
    }

    /// Unprojects a 2D image point to a 3D ray in camera coordinates.
    ///
    /// This method applies the inverse Double Sphere model equations to convert
    /// a 2D pixel coordinate back into a 3D direction vector (ray) originating
    /// from the camera center. The resulting vector is normalized.
    /// It first checks if the point can be unprojected using [`DoubleSphereModel::check_unprojection_condition`].
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
    ///   according to the Double Sphere model's geometric constraints (e.g., point is outside
    ///   the valid unprojection radius if `alpha > 0.5`, or results in a denominator close to zero).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::{DVector, Vector2};
    /// use fisheye_tools::camera::double_sphere::DoubleSphereModel;
    /// use fisheye_tools::camera::{CameraModel, Resolution, CameraModelError};
    ///
    /// let params = DVector::from_vec(vec![350.0, 350.0, 320.0, 240.0, 0.58, -0.18]);
    /// let mut model = DoubleSphereModel::new(&params).unwrap();
    /// model.resolution = Resolution { width: 640, height: 480 };
    ///
    /// // A point near the center of the image
    /// let point_2d = Vector2::new(330.0, 250.0);
    /// match model.unproject(&point_2d) {
    ///     Ok(ray_3d) => {
    ///         println!("Unprojected ray: ({}, {}, {})", ray_3d.x, ray_3d.y, ray_3d.z);
    ///         // Check that it's a unit vector
    ///         assert!((ray_3d.norm() - 1.0).abs() < 1e-6);
    ///     }
    ///     Err(e) => println!("Unprojection failed: {:?}", e),
    /// }
    /// ```
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        const PRECISION: f64 = 1e-3; // Used to check for small denominators

        let fx = self.intrinsics.fx;
        let fy = self.intrinsics.fy;
        let cx = self.intrinsics.cx;
        let cy = self.intrinsics.cy;
        let alpha = self.alpha;
        let xi = self.xi;

        let u = point_2d.x;
        let v = point_2d.y;
        let gamma_ds = 1.0 - alpha; // Renamed to avoid conflict with 'gamma' in project
        let mx = (u - cx) / fx;
        let my = (v - cy) / fy;
        let r_squared = (mx * mx) + (my * my);

        // Check if we can unproject this point based on alpha and r_squared
        if alpha != 0.0 && !self.check_unprojection_condition(r_squared) {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let mz = (1.0 - alpha * alpha * r_squared)
            / (alpha * (1.0 - (2.0 * alpha - 1.0) * r_squared).sqrt() + gamma_ds);
        let mz_squared = mz * mz;

        let num = mz * xi + (mz_squared + (1.0 - xi * xi) * r_squared).sqrt();
        let denom = mz_squared + r_squared;

        // Check if denominator is too small, indicating potential instability
        if denom < PRECISION {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let coeff = num / denom;

        // Calculate the unprojected 3D point
        let point3d = Vector3::new(coeff * mx, coeff * my, coeff * mz - xi);

        Ok(point3d.normalize())
    }

    /// Loads [`DoubleSphereModel`] parameters from a YAML file.
    ///
    /// The YAML file is expected to follow a structure where camera parameters are nested
    /// under `cam0`. The intrinsic parameters (`fx`, `fy`, `cx`, `cy`) and the Double
    /// Sphere specific distortion parameters (`alpha`, `xi`) are typically grouped
    /// together in an `intrinsics` array in the YAML file: `[fx, fy, cx, cy, alpha, xi]`.
    /// The `resolution` (width, height) is also expected under `cam0`.
    ///
    /// # Arguments
    ///
    /// * `path`: A string slice representing the path to the YAML file.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Self, CameraModelError>`. On success, it provides an instance
    /// of [`DoubleSphereModel`] populated with parameters from the file.
    ///
    /// # Errors
    ///
    /// This function can return:
    /// * [`CameraModelError::IOError`]: If there's an issue reading the file.
    /// * [`CameraModelError::YamlError`]: If the YAML content is malformed or cannot be parsed.
    /// * [`CameraModelError::InvalidParams`]: If the YAML structure is missing expected fields
    ///   (e.g., "intrinsics", "resolution") or if parameter values are of incorrect types or counts.
    ///   This includes cases where `alpha` or `xi` cannot be extracted as `f64` from the
    ///   `intrinsics` array in the YAML.
    /// * Errors from [`DoubleSphereModel::validate_params()`] if the loaded parameters are invalid
    ///   (e.g., `alpha` out of range, non-finite `xi`, or invalid core intrinsics).
    ///
    /// # Related
    /// * [`DoubleSphereModel::save_to_yaml()`]
    fn load_from_yaml(path: &str) -> Result<Self, CameraModelError> {
        let contents = fs::read_to_string(path)?;
        let docs = YamlLoader::load_from_str(&contents)?;

        if docs.is_empty() {
            return Err(CameraModelError::InvalidParams(
                "Empty YAML document".to_string(),
            ));
        }

        let doc = &docs[0];

        let intrinsics_yaml_vec = doc["cam0"]["intrinsics"] // Renamed for clarity
            .as_vec()
            .ok_or_else(|| {
                CameraModelError::InvalidParams(
                    "YAML missing 'intrinsics' array under 'cam0'".to_string(),
                )
            })?;
        let resolution_yaml_vec = doc["cam0"]["resolution"] // Renamed for clarity
            .as_vec()
            .ok_or_else(|| {
                CameraModelError::InvalidParams(
                    "YAML missing 'resolution' array under 'cam0'".to_string(),
                )
            })?;

        if intrinsics_yaml_vec.len() < 6 {
            return Err(CameraModelError::InvalidParams(
                "Intrinsics array in YAML must have at least 6 elements (fx, fy, cx, cy, alpha, xi)".to_string()
            ));
        }

        let alpha = intrinsics_yaml_vec[4].as_f64().ok_or_else(|| {
            CameraModelError::InvalidParams("Invalid alpha in YAML: not a float".to_string())
        })?;

        let xi = intrinsics_yaml_vec[5].as_f64().ok_or_else(|| {
            CameraModelError::InvalidParams("Invalid xi in YAML: not a float".to_string())
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

        let model = DoubleSphereModel {
            intrinsics,
            resolution,
            alpha,
            xi,
        };

        // Validate parameters
        model.validate_params()?;
        Ok(model)
    }

    /// Saves the [`DoubleSphereModel`] parameters to a YAML file.
    ///
    /// The parameters are saved under the `cam0` key. The `intrinsics` YAML array
    /// will contain `fx, fy, cx, cy, alpha, xi` in that order.
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
    ///
    /// # Related
    /// * [`DoubleSphereModel::load_from_yaml()`]
    fn save_to_yaml(&self, path: &str) -> Result<(), CameraModelError> {
        // Create the YAML structure using serde_yaml
        let yaml = serde_yaml::to_value(serde_yaml::Mapping::from_iter([(
            serde_yaml::Value::String("cam0".to_string()),
            serde_yaml::to_value(serde_yaml::Mapping::from_iter([
                (
                    serde_yaml::Value::String("camera_model".to_string()),
                    serde_yaml::Value::String("double_sphere".to_string()),
                ),
                (
                    serde_yaml::Value::String("intrinsics".to_string()),
                    serde_yaml::to_value(vec![
                        self.intrinsics.fx,
                        self.intrinsics.fy,
                        self.intrinsics.cx,
                        self.intrinsics.cy,
                        self.alpha, // alpha is 5th element
                        self.xi,    // xi is 6th element
                    ])
                    .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
                ),
                (
                    serde_yaml::Value::String("rostopic".to_string()), // Often included in Kalibr format
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

    /// Validates the parameters of the [`DoubleSphereModel`].
    ///
    /// This method checks the validity of the core intrinsic parameters (focal lengths,
    /// principal point) using [`validation::validate_intrinsics`]. It also validates
    /// the Double Sphere specific parameters:
    /// *   `alpha` must be in the range (0, 1] (i.e., `0.0 < alpha <= 1.0`).
    /// *   `xi` must be a finite `f64` value (not NaN or infinity).
    ///
    /// # Return Value
    ///
    /// Returns `Ok(())` if all parameters are valid.
    ///
    /// # Errors
    ///
    /// Returns a [`CameraModelError`] if any parameter is invalid:
    /// * [`CameraModelError::InvalidParams`]: If `alpha` is out of its valid range (0, 1],
    ///   or if `xi` is not a finite number. The error message will specify the issue.
    /// * Errors propagated from [`validation::validate_intrinsics`] (e.g.,
    ///   [`CameraModelError::FocalLengthMustBePositive`],
    ///   [`CameraModelError::PrincipalPointMustBeFinite`]).
    fn validate_params(&self) -> Result<(), CameraModelError> {
        validation::validate_intrinsics(&self.intrinsics)?;

        if self.alpha <= 0.0 || self.alpha > 1.0 {
            return Err(CameraModelError::InvalidParams(
                "alpha must be in (0, 1]".to_string(),
            ));
        }

        if !self.xi.is_finite() {
            return Err(CameraModelError::InvalidParams(
                "xi must be finite".to_string(),
            ));
        }

        Ok(())
    }

    /// Returns a clone of the camera's image resolution.
    ///
    /// # Return Value
    ///
    /// A [`Resolution`] struct containing the width and height of the camera image.
    fn get_resolution(&self) -> Resolution {
        self.resolution.clone()
    }

    /// Returns a clone of the camera's intrinsic parameters.
    ///
    /// # Return Value
    ///
    /// An [`Intrinsics`] struct containing `fx`, `fy`, `cx`, and `cy`.
    fn get_intrinsics(&self) -> Intrinsics {
        self.intrinsics.clone()
    }

    /// Returns the distortion parameters of the Double Sphere model.
    ///
    /// The parameters are returned as a vector containing `[xi, alpha]`.
    /// Note the order: `xi` is the first element, and `alpha` is the second.
    ///
    /// # Return Value
    ///
    /// A `Vec<f64>` containing the two distortion parameters: `[xi, alpha]`.
    fn get_distortion(&self) -> Vec<f64> {
        vec![self.alpha, self.xi] // Order: xi, then alpha
    }

    // linear_estimation removed from impl CameraModel for DoubleSphereModel
    // optimize removed from impl CameraModel for DoubleSphereModel
}

/// Unit tests for the [`DoubleSphereModel`].
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq; // For floating point comparisons

    /// Helper function to create a sample [`DoubleSphereModel`] instance for testing.
    /// This model is based on parameters similar to those in "samples/double_sphere.yaml".
    fn get_sample_model() -> DoubleSphereModel {
        DoubleSphereModel {
            intrinsics: Intrinsics {
                fx: 348.112754378549,
                fy: 347.1109973814674,
                cx: 365.8121721753254,
                cy: 249.3555778487899,
            },
            resolution: Resolution {
                width: 752,
                height: 480,
            },
            alpha: 0.5657413673629862,
            xi: -0.24425190195168348,
        }
    }

    /// Tests loading [`DoubleSphereModel`] parameters from "samples/double_sphere.yaml".
    #[test]
    fn test_double_sphere_load_from_yaml() {
        let path = "samples/double_sphere.yaml";
        let model = DoubleSphereModel::load_from_yaml(path).unwrap();

        assert_eq!(model.intrinsics.fx, 348.112754378549);
        assert_eq!(model.intrinsics.fy, 347.1109973814674);
        assert_eq!(model.intrinsics.cx, 365.8121721753254);
        assert_eq!(model.intrinsics.cy, 249.3555778487899);
        assert_eq!(model.alpha, 0.5657413673629862);
        assert_eq!(model.xi, -0.24425190195168348);
        assert_eq!(model.resolution.width, 752);
        assert_eq!(model.resolution.height, 480);
    }

    /// Tests saving [`DoubleSphereModel`] parameters to a YAML file and then reloading them.
    #[test]
    fn test_double_sphere_save_to_yaml() {
        use std::fs;

        // Create output directory if it doesn't exist
        fs::create_dir_all("output").unwrap_or_else(|_| {
            // This info! macro might not be visible depending on test runner's log capture.
            // For robust tests, usually prefer expect or panic on critical setup errors.
            info!("Output directory already exists or couldn't be created");
        });

        // Define input and output paths
        let input_path = "samples/double_sphere.yaml"; // Source of truth for this test
        let output_path = "output/double_sphere_saved.yaml";

        // Load the camera model from the original YAML
        let model = DoubleSphereModel::load_from_yaml(input_path).unwrap();

        // Save the model to the output path
        model.save_to_yaml(output_path).unwrap();

        // Load the model from the saved file
        let saved_model = DoubleSphereModel::load_from_yaml(output_path).unwrap();

        // Compare the original and saved models
        assert_eq!(model.intrinsics.fx, saved_model.intrinsics.fx);
        assert_eq!(model.intrinsics.fy, saved_model.intrinsics.fy);
        assert_eq!(model.intrinsics.cx, saved_model.intrinsics.cx);
        assert_eq!(model.intrinsics.cy, saved_model.intrinsics.cy);
        assert_eq!(model.xi, saved_model.xi);
        assert_eq!(model.alpha, saved_model.alpha);
        assert_eq!(model.resolution.width, saved_model.resolution.width);
        assert_eq!(model.resolution.height, saved_model.resolution.height);

        // Clean up the saved file (optional, but good practice for tests)
        fs::remove_file(output_path).unwrap();
    }

    /// Tests the consistency of projection and unprojection for the [`DoubleSphereModel`].
    #[test]
    fn test_double_sphere_project_unproject() {
        // Load the camera model from YAML
        let path = "samples/double_sphere.yaml";
        let model = DoubleSphereModel::load_from_yaml(path).unwrap();

        // Create a 3D point in camera coordinates
        let point_3d = Vector3::new(0.5, -0.3, 2.0);
        let norm_3d = point_3d.normalize();

        // Project the 3D point to pixel coordinates
        let point_2d = model.project(&point_3d).unwrap();

        // Check if the pixel coordinates are within the image bounds (basic sanity check)
        assert!(point_2d.x >= 0.0 && point_2d.x < model.resolution.width as f64);
        assert!(point_2d.y >= 0.0 && point_2d.y < model.resolution.height as f64);

        // Unproject the pixel point back to a 3D ray direction
        let point_3d_unprojected = model.unproject(&point_2d).unwrap();

        // Check if the unprojected point (normalized ray) is close to the original normalized point
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

        // Behavior for points very close to the center can be model-specific.
        // For Double Sphere, if `denom` becomes too small or `check_projection_condition` fails,
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
        // This should typically result in an error due to d1=0, leading to small denom or failed check.
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
        // Expect an error as the point is behind the camera and likely fails check_projection_condition.
        assert!(matches!(result, Err(CameraModelError::PointIsOutSideImage)));
    }

    /// Tests `validate_params` with a valid model.
    #[test]
    fn test_validate_params_valid() {
        let model = get_sample_model();
        assert!(model.validate_params().is_ok());
    }

    /// Tests `validate_params` with invalid `alpha` values.
    #[test]
    fn test_validate_params_invalid_alpha() {
        let mut model = get_sample_model();
        model.alpha = 0.0; // Invalid: alpha must be > 0
        assert!(matches!(
            model.validate_params(),
            Err(CameraModelError::InvalidParams(msg)) if msg == "alpha must be in (0, 1]"
        ));

        model.alpha = 1.1; // Invalid: alpha must be <= 1
        assert!(matches!(
            model.validate_params(),
            Err(CameraModelError::InvalidParams(msg)) if msg == "alpha must be in (0, 1]"
        ));
    }

    /// Tests `validate_params` with invalid `xi` values (NaN, Infinity).
    #[test]
    fn test_validate_params_invalid_xi() {
        let mut model = get_sample_model();
        model.xi = f64::NAN;
        assert!(matches!(
            model.validate_params(),
            Err(CameraModelError::InvalidParams(msg)) if msg == "xi must be finite"
        ));

        model.xi = f64::INFINITY;
        assert!(matches!(
            model.validate_params(),
            Err(CameraModelError::InvalidParams(msg)) if msg == "xi must be finite"
        ));
    }

    /// Tests `validate_params` with invalid core intrinsic parameters.
    #[test]
    fn test_validate_params_invalid_intrinsics() {
        let mut model = get_sample_model();
        model.intrinsics.fx = 0.0; // Invalid: fx must be > 0
                                   // This error comes from `validation::validate_intrinsics`.
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
        assert_relative_eq!(distortion[0], model.alpha); // alpha is second
        assert_relative_eq!(distortion[1], model.xi); // xi is first
    }
}
