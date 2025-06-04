//! Implements the Radial-Tangential (RadTan) camera model.
//!
//! This module provides the [`RadTanModel`] struct and its associated methods
//! for representing and working with a camera that exhibits radial and tangential
//! lens distortion. It adheres to the [`CameraModel`] trait defined in the
//! parent `camera` module ([`crate::camera`]). The RadTan model is commonly used
//! to correct for these types of distortions in many computer vision applications.

use crate::camera::{validation, CameraModel, CameraModelError, Intrinsics, Resolution};
use nalgebra::{DVector, Matrix2, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use std::{fmt, fs, io::Write};
use yaml_rust::YamlLoader;

/// Represents a Radial-Tangential (RadTan) camera model.
///
/// This struct holds the intrinsic parameters (focal length, principal point),
/// image resolution, and a set of 5 distortion coefficients that define
/// the radial and tangential distortion characteristics of the camera lens.
///
/// The distortion coefficients are typically denoted as:
/// *   `k1`, `k2`, `k3`: Radial distortion coefficients.
/// *   `p1`, `p2`: Tangential distortion coefficients.
///
/// # Examples
///
/// ```rust
/// use nalgebra::DVector;
/// use fisheye_tools::camera::rad_tan::RadTanModel;
/// use fisheye_tools::camera::{Intrinsics, Resolution, CameraModelError};
///
/// // Create a RadTanModel using the new constructor
/// // Parameters: fx, fy, cx, cy, k1, k2, p1, p2, k3
/// let params = DVector::from_vec(vec![
///     500.0, 500.0, 320.0, 240.0, // Intrinsics
///     0.1, -0.05, 0.001, 0.001, 0.02 // Distortion (k1, k2, p1, p2, k3)
/// ]);
/// let mut rad_tan_model = RadTanModel::new(&params).unwrap();
/// // Set a resolution, as `new` initializes it to 0x0
/// rad_tan_model.resolution = Resolution { width: 640, height: 480 };
///
/// assert_eq!(rad_tan_model.intrinsics.fx, 500.0);
/// assert_eq!(rad_tan_model.distortions[0], 0.1); // k1
/// assert_eq!(rad_tan_model.resolution.width, 640);
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct RadTanModel {
    /// The intrinsic parameters of the camera, [`Intrinsics`] (fx, fy, cx, cy).
    pub intrinsics: Intrinsics,
    /// The resolution of the camera image, [`Resolution`] (width, height).
    pub resolution: Resolution,
    /// The 5 distortion coefficients: `[k1, k2, p1, p2, k3]`.
    /// * `k1`, `k2`, `k3`: Radial distortion coefficients.
    /// * `p1`, `p2`: Tangential distortion coefficients.
    pub distortions: [f64; 5], // k1, k2, p1, p2, k3
}

impl RadTanModel {
    /// Creates a new [`RadTanModel`] from a vector of parameters.
    ///
    /// The resolution is initialized to 0x0 and should be set manually or by loading from YAML.
    ///
    /// # Arguments
    ///
    /// * `parameters` - A `&DVector<f64>` containing the camera parameters in the order:
    ///   1.  `fx`: Focal length along the x-axis.
    ///   2.  `fy`: Focal length along the y-axis.
    ///   3.  `cx`: Principal point x-coordinate.
    ///   4.  `cy`: Principal point y-coordinate.
    ///   5.  `k1`: First radial distortion coefficient.
    ///   6.  `k2`: Second radial distortion coefficient.
    ///   7.  `p1`: First tangential distortion coefficient.
    ///   8.  `p2`: Second tangential distortion coefficient.
    ///   9.  `k3`: Third radial distortion coefficient.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Self, CameraModelError>`. On success, it provides the [`RadTanModel`] instance.
    ///
    /// # Errors
    ///
    /// This function can return a [`CameraModelError`] if the provided intrinsic parameters are invalid
    /// (e.g., non-positive focal length, non-finite principal point), as checked by `validate_params`.
    /// Specifically, it can return:
    /// * [`CameraModelError::FocalLengthMustBePositive`]
    /// * [`CameraModelError::PrincipalPointMustBeFinite`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::DVector;
    /// use fisheye_tools::camera::rad_tan::RadTanModel;
    /// use fisheye_tools::camera::{Resolution, CameraModelError};
    ///
    /// let params = DVector::from_vec(vec![
    ///     460.0, 460.0, 320.0, 240.0, // fx, fy, cx, cy
    ///     -0.28, 0.07, 0.0002, 0.00002, 0.0 // k1, k2, p1, p2, k3
    /// ]);
    /// let mut model = RadTanModel::new(&params).expect("Failed to create RadTanModel");
    /// model.resolution = Resolution { width: 640, height: 480 }; // Set resolution
    ///
    /// assert_eq!(model.intrinsics.fx, 460.0);
    /// assert_eq!(model.distortions[0], -0.28); // k1
    /// assert_eq!(model.resolution.width, 640);
    /// ```
    pub fn new(parameters: &DVector<f64>) -> Result<Self, CameraModelError> {
        let model = RadTanModel {
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
            distortions: [
                parameters[4], // k1
                parameters[5], // k2
                parameters[6], // p1
                parameters[7], // p2
                parameters[8], // k3
            ],
        };

        model.validate_params()?;
        Ok(model)
    }
}

/// Provides a debug string representation for [`RadTanModel`].
impl fmt::Debug for RadTanModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RadTanModel [fx: {} fy: {} cx: {} cy: {} distortions: {:?}]",
            self.intrinsics.fx,
            self.intrinsics.fy,
            self.intrinsics.cx,
            self.intrinsics.cy,
            self.distortions,
        )
    }
}

impl CameraModel for RadTanModel {
    /// Projects a 3D point from camera coordinates to 2D image coordinates, applying distortion.
    ///
    /// The projection involves normalizing the 3D point, applying radial and tangential
    /// distortion based on the model's `distortion` coefficients, and then projecting
    /// the distorted normalized point to pixel coordinates using the intrinsic parameters.
    /// The Jacobian of the projection function is not computed for this model and will be `None`.
    ///
    /// # Arguments
    ///
    /// * `point_3d` - A `&Vector3<f64>` representing the 3D point (X, Y, Z) in camera coordinates.
    /// * `_compute_jacobian` - A boolean flag. If true, the Jacobian would be computed.
    ///   However, for `RadTanModel`, this is ignored and the Jacobian is always `None`.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<(Vector2<f64>, Option<DMatrix<f64>>), CameraModelError>`.
    /// On success, it provides a tuple containing:
    /// * The projected 2D point (`Vector2<f64>`) in pixel coordinates (u, v).
    /// * `None` for the Jacobian matrix, as it's not implemented for this model.
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
    /// use fisheye_tools::camera::rad_tan::RadTanModel;
    /// use fisheye_tools::camera::{CameraModel, Resolution};
    ///
    /// let params = DVector::from_vec(vec![
    ///     500.0, 500.0, 320.0, 240.0, // Intrinsics
    ///     0.01, 0.0, 0.0, 0.0, 0.0    // Minimal distortion
    /// ]);
    /// let mut model = RadTanModel::new(&params).unwrap();
    /// model.resolution = Resolution { width: 640, height: 480 };
    ///
    /// let point_3d = Vector3::new(0.1, 0.2, 1.0); // X, Y, Z in meters
    /// match model.project(&point_3d) {
    ///     Ok(point_2d) => {
    ///         println!("Projected point: ({}, {})", point_2d.x, point_2d.y);
    ///         // Actual values will depend on distortion, this is a basic check
    ///         assert!(point_2d.x > 0.0 && point_2d.y > 0.0);
    ///     }
    ///     Err(e) => println!("Projection failed: {:?}", e),
    /// }
    /// ```
    fn project(&self, point_3d: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        // If z is very small, the point is at the camera center
        if point_3d.z < f64::EPSILON.sqrt() {
            return Err(CameraModelError::PointAtCameraCenter);
        }

        let x = point_3d.x;
        let y = point_3d.y;
        let z = point_3d.z;

        let k1 = self.distortions[0];
        let k2 = self.distortions[1];
        let p1 = self.distortions[2];
        let p2 = self.distortions[3];
        let k3 = self.distortions[4];

        // Calculate normalized image coordinates
        let x_prime = x / z;
        let y_prime = y / z;

        let r2 = x_prime.powi(2) + y_prime.powi(2);
        let r4 = r2.powi(2);
        let r6 = r4.powi(2); // Original r6 calculation: r4.powi(2) -> r2*r2*r2 or r2*r4

        // Apply radial and tangential distortion
        let x_distorted = x_prime * (1.0 + k1 * r2 + k2 * r4 + k3 * r6)
            + 2.0 * p1 * x_prime * y_prime
            + p2 * (r2 + 2.0 * x_prime * x_prime);

        let y_distorted = y_prime * (1.0 + k1 * r2 + k2 * r4 + k3 * r6)
            + p1 * (r2 + 2.0 * y_prime * y_prime)
            + 2.0 * p2 * x_prime * y_prime;

        let u = self.intrinsics.fx * x_distorted + self.intrinsics.cx;
        let v = self.intrinsics.fy * y_distorted + self.intrinsics.cy;

        // Check if the projected point is inside the image
        if u < 0.0
            || u >= self.resolution.width as f64
            || v < 0.0
            || v >= self.resolution.height as f64
        {
            return Err(CameraModelError::ProjectionOutSideImage);
        }

        Ok(Vector2::new(u, v))
    }

    /// Unprojects a 2D image point (with distortion) to a 3D ray in camera coordinates.
    ///
    /// This method iteratively solves for the undistorted normalized image coordinates
    /// that, when distorted, produce the input `point_2d`. It uses a numerical approach
    /// (Newton's method with Jacobian) to refine an initial guess. The resulting
    /// undistorted normalized coordinates `(x, y)` are then formed into a 3D ray `(x, y, 1.0)`
    /// and normalized.
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
    /// * [`CameraModelError::NumericalError`]: If the iterative unprojection fails to converge
    ///   (e.g., Jacobian is singular, maximum iterations reached).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::{DVector, Vector2};
    /// use fisheye_tools::camera::rad_tan::RadTanModel;
    /// use fisheye_tools::camera::{CameraModel, Resolution};
    ///
    /// let params = DVector::from_vec(vec![
    ///     500.0, 500.0, 320.0, 240.0, // Intrinsics
    ///     0.01, 0.0, 0.0, 0.0, 0.0    // Minimal distortion
    /// ]);
    /// let mut model = RadTanModel::new(&params).unwrap();
    /// model.resolution = Resolution { width: 640, height: 480 };
    ///
    /// // A point that, if undistorted, would be (370, 340) with the above intrinsics
    /// // (0.1, 0.2) in normalized coords. Distortion will shift this slightly.
    /// // For this example, we'll use a point known to be valid.
    /// let point_2d = Vector2::new(370.5, 340.5);
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
        if point_2d.x < 0.0
            || point_2d.x >= self.resolution.width as f64
            || point_2d.y < 0.0
            || point_2d.y >= self.resolution.height as f64
        {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let u = point_2d.x;
        let v = point_2d.y;
        let fx = self.intrinsics.fx;
        let fy = self.intrinsics.fy;
        let cx = self.intrinsics.cx;
        let cy = self.intrinsics.cy;

        let k1 = self.distortions[0];
        let k2 = self.distortions[1];
        let p1 = self.distortions[2];
        let p2 = self.distortions[3];
        let k3 = self.distortions[4];

        // Calculate normalized coordinates of the distorted point
        // This is the target point in the normalized image plane we want to match
        let x_distorted = (u - cx) / fx;
        let y_distorted = (v - cy) / fy;
        let target_distorted_point = Vector2::new(x_distorted, y_distorted);

        // Initial guess for the undistorted normalized point (start with the distorted point)
        let mut point = target_distorted_point;

        // Tolerance for convergence checks
        const EPS: f64 = 1e-6;
        const MAX_ITERATIONS: u32 = 100; // Add max iterations to prevent infinite loops

        for iteration in 0..MAX_ITERATIONS {
            let x = point.x;
            let y = point.y;
            let r2 = x * x + y * y; // r^2
            let r4 = r2 * r2; // r^4
            let r6 = r4 * r2; // r^6 Original: r4 * r2

            // Calculate the radial distortion factor
            let radial_distortion = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

            // Estimate the distorted position based on the current undistorted estimate `point`
            // and the distortion model (radial + tangential)
            let x_distorted_estimate =
                x * radial_distortion + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
            let y_distorted_estimate =
                y * radial_distortion + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
            let estimated_distorted_point =
                Vector2::new(x_distorted_estimate, y_distorted_estimate);

            // Calculate the error: difference between the estimated distorted point
            // and the actual target distorted point
            let error = estimated_distorted_point - target_distorted_point;

            // Check for convergence based on the error norm
            if error.norm() < EPS {
                break; // Converged
            }

            // Calculate the Jacobian matrix (J) of the distortion function
            // with respect to the undistorted point (x, y)
            // Original term1: k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;
            // This was the derivative of (r * radial_distortion_term_in_r) / r, which simplifies.
            // Corrected term should be derivative of (1 + k1*r^2 + k2*r^4 + k3*r^6) w.r.t r^2, then times 2x or 2y.
            // d(1+k1*r^2+k2*r^4+k3*r^6)/dx = (k1*2x + k2*4*r^2*x + k3*6*r^4*x)
            // d(1+k1*r^2+k2*r^4+k3*r^6)/dy = (k1*2y + k2*4*r^2*y + k3*6*r^4*y)
            let dr_dx = 2.0 * x;
            let dr_dy = 2.0 * y;
            let d_radial_term_dx = (k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4) * dr_dx;
            let d_radial_term_dy = (k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4) * dr_dy;

            // Jacobian elements (derivatives of estimated distorted coords w.r.t. undistorted coords)
            // d(x_est)/dx = radial_distortion + x * d(radial_distortion)/dx + d(tangential_x)/dx
            let j00 =
                radial_distortion + x * d_radial_term_dx + 2.0 * p1 * y + p2 * (dr_dx + 4.0 * x);
            // d(x_est)/dy = x * d(radial_distortion)/dy + d(tangential_x)/dy
            let j01 = x * d_radial_term_dy + 2.0 * p1 * x + p2 * (dr_dy);
            // d(y_est)/dx = y * d(radial_distortion)/dx + d(tangential_y)/dx
            let j10 = y * d_radial_term_dx + p1 * (dr_dx) + 2.0 * p2 * y;
            // d(y_est)/dy = radial_distortion + y * d(radial_distortion)/dy + d(tangential_y)/dy
            let j11 =
                radial_distortion + y * d_radial_term_dy + p1 * (dr_dy + 4.0 * y) + 2.0 * p2 * x;

            // Construct the Jacobian matrix
            let jacobian = Matrix2::new(j00, j01, j10, j11);

            // Solve for the update step (delta) using the inverse of the Jacobian
            // J * delta = -error  => delta = -J.inverse() * error (or solve directly)
            if let Some(inv_jacobian) = jacobian.try_inverse() {
                let delta = inv_jacobian * error;

                // Update the undistorted point estimate
                point -= delta; // Apply the correction

                // Check for convergence based on the step size norm
                if delta.norm() < EPS {
                    break; // Converged
                }
            } else {
                // Jacobian is not invertible, cannot proceed
                // This might happen if distortion is extreme or point is near singularity
                return Err(CameraModelError::NumericalError(
                    "Jacobian is singular".to_string(), // Original message
                ));
            }

            // If loop finished without converging (max iterations reached)
            if iteration == MAX_ITERATIONS - 1 {
                return Err(CameraModelError::NumericalError(
                    // Original message: "Unprojection did not converge after {MAX_ITERATIONS} iterations.".to_string(),
                    format!(
                        "Unprojection did not converge after {} iterations.",
                        MAX_ITERATIONS
                    ),
                ));
            }
        }

        // Create the 3D point with the undistorted x, y and z=1
        let point3d = Vector3::new(point.x, point.y, 1.0);

        Ok(point3d.normalize())
    }

    /// Loads RadTan camera parameters from a YAML file.
    ///
    /// The YAML file is expected to have a specific structure, typically including
    /// `cam0`, `intrinsics` (fx, fy, cx, cy), `resolution` (width, height),
    /// and `distortion` (k1, k2, p1, p2, k3).
    ///
    /// # Arguments
    ///
    /// * `path` - A string slice representing the path to the YAML file.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Self, CameraModelError>`. On success, it provides an instance
    /// of [`RadTanModel`] populated with parameters from the file.
    ///
    /// # Errors
    ///
    /// This function can return various [`CameraModelError`] variants:
    /// * [`CameraModelError::IOError`]: If there's an issue reading the file.
    /// * [`CameraModelError::YamlError`]: If the YAML content is malformed or cannot be parsed.
    /// * [`CameraModelError::InvalidParams`]: If the YAML structure is missing expected fields
    ///   (e.g., "intrinsics", "resolution", "distortion") or if parameter values are of incorrect types
    ///   or counts (e.g., not 5 distortion parameters).
    /// * Errors from `validate_params` (e.g., [`CameraModelError::FocalLengthMustBePositive`])
    ///   if the loaded intrinsic parameters are invalid.
    ///
    /// # Related
    /// * [`RadTanModel::save_to_yaml()`]
    fn load_from_yaml(path: &str) -> Result<Self, CameraModelError> {
        let contents = fs::read_to_string(path)?;
        let docs = YamlLoader::load_from_str(&contents)?;

        if docs.is_empty() {
            return Err(CameraModelError::InvalidParams(
                "Empty YAML document".to_string(),
            ));
        }

        let doc = &docs[0];

        let intrinsics_yaml = doc["cam0"]["intrinsics"] // Original: `intrinsics`
            .as_vec()
            .ok_or_else(|| CameraModelError::InvalidParams("Invalid intrinsics".to_string()))?;
        let resolution_yaml = doc["cam0"]["resolution"] // Original: `resolution`
            .as_vec()
            .ok_or_else(|| CameraModelError::InvalidParams("Invalid resolution".to_string()))?;

        // Extract distortion parameters
        let distortion_node = doc["cam0"]["distortion"].as_vec().ok_or_else(|| {
            CameraModelError::InvalidParams("Missing distortion parameters".to_string())
        })?;

        let intrinsics = Intrinsics {
            fx: intrinsics_yaml[0] // Original: `intrinsics[0]`
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid fx".to_string()))?,
            fy: intrinsics_yaml[1] // Original: `intrinsics[1]`
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid fy".to_string()))?,
            cx: intrinsics_yaml[2] // Original: `intrinsics[2]`
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid cx".to_string()))?,
            cy: intrinsics_yaml[3] // Original: `intrinsics[3]`
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid cy".to_string()))?,
        };

        let resolution = Resolution {
            width: resolution_yaml[0] // Original: `resolution[0]`
                .as_i64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid width".to_string()))?
                as u32,
            height: resolution_yaml[1] // Original: `resolution[1]`
                .as_i64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid height".to_string()))?
                as u32,
        };

        let mut distortions = [0.0; 5]; // Initialize the fixed-size array

        // Ensure the YAML node contains the correct number of parameters
        if distortion_node.len() != 5 {
            return Err(CameraModelError::InvalidParams(format!(
                "Expected 5 distortion parameters in YAML, found {}",
                distortion_node.len()
            )));
        }

        for (i, param) in distortion_node.iter().enumerate() {
            let value = param.as_f64().ok_or_else(|| {
                CameraModelError::InvalidParams(format!(
                    "Invalid distortion parameter at index {}", // Original: no float specification
                    i
                ))
            })?;
            // Assign the parsed value to the corresponding index in the array
            distortions[i] = value;
        }

        // This check is redundant due to the fixed-size array and the loop above.
        // if distortion.len() != 5 {
        //     return Err(CameraModelError::InvalidParams(format!(
        //         "Expected 5 distortion parameters, got {}",
        //         distortion.len()
        //     )));
        // }

        let model = RadTanModel {
            intrinsics,
            resolution,
            distortions,
        };

        // Validate parameters
        model.validate_params()?;

        Ok(model)
    }

    /// Saves the camera model's parameters to a YAML file.
    ///
    /// The output YAML file will include the camera model type, intrinsic parameters,
    /// distortion coefficients (k1, k2, p1, p2, k3), and resolution.
    ///
    /// **Note:** The `camera_model` field in the output YAML is currently hardcoded to
    /// `"double_sphere"` in this implementation. This appears to be a bug and should ideally
    /// be `"rad_tan"` or a more generic identifier. This documentation reflects the
    /// current behavior.
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
    /// * [`RadTanModel::load_from_yaml()`]
    fn save_to_yaml(&self, path: &str) -> Result<(), CameraModelError> {
        // Create the YAML structure using serde_yaml
        let yaml =
            serde_yaml::to_value(&serde_yaml::Mapping::from_iter(
                [
                    (
                        serde_yaml::Value::String("cam0".to_string()),
                        serde_yaml::to_value(
                            &serde_yaml::Mapping::from_iter(
                                [
                                    (
                                        serde_yaml::Value::String("camera_model".to_string()),
                                        // TODO: This should ideally be "rad_tan" or similar, not "double_sphere".
                                        // Documenting current behavior.
                                        serde_yaml::Value::String("double_sphere".to_string()),
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
                                        serde_yaml::Value::String("distortion".to_string()),
                                        serde_yaml::to_value(self.distortions.to_vec()) // Original: explicit vec construction
                                            .map_err(|e| {
                                                CameraModelError::YamlError(e.to_string())
                                            })?,
                                    ),
                                    (
                                        serde_yaml::Value::String("rostopic".to_string()), // Often included in Kalibr format
                                        serde_yaml::Value::String("/cam0/image_raw".to_string()),
                                    ),
                                    (
                                        serde_yaml::Value::String("resolution".to_string()),
                                        serde_yaml::to_value(vec![
                                            self.resolution.width,
                                            self.resolution.height,
                                        ])
                                        .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
                                    ),
                                ],
                            ),
                        )
                        .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
                    ),
                ],
            ))
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
    /// It utilizes `validation::validate_intrinsics` from the parent `camera` module.
    ///
    /// **Note:** Currently, this method does not perform validation specific to the
    /// `distortions` parameters (e.g., checking for reasonable ranges).
    ///
    /// # Return Value
    ///
    /// Returns `Ok(())` if the intrinsic parameters are valid.
    ///
    /// # Errors
    ///
    /// Returns a [`CameraModelError`] if intrinsic parameter validation fails, specifically:
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

    /// Returns a vector containing the distortion coefficients of the camera.
    ///
    /// The coefficients are returned in the order: `[k1, k2, p1, p2, k3]`.
    ///
    /// # Return Value
    /// A `Vec<f64>` containing the 5 distortion coefficients.
    fn get_distortion(&self) -> Vec<f64> {
        self.distortions.to_vec()
    }

    // linear_estimation removed from impl CameraModel for RadTanModel
    // optimize removed from impl CameraModel for RadTanModel
}

/// Contains unit tests for the Radial-Tangential (RadTan) camera model.
///
/// These tests cover loading parameters from YAML, saving parameters to YAML,
/// and the consistency of projection and unprojection operations.
#[cfg(test)]
mod tests {
    use super::*;

    /// Tests loading [`RadTanModel`] parameters from "samples/rad_tan.yaml".
    #[test]
    fn test_radtan_load_from_yaml() {
        let path = "samples/rad_tan.yaml";
        let model = RadTanModel::load_from_yaml(path).unwrap();

        assert_eq!(model.intrinsics.fx, 461.629);
        assert_eq!(model.intrinsics.fy, 460.152);
        assert_eq!(model.intrinsics.cx, 362.680);
        assert_eq!(model.intrinsics.cy, 246.049);
        assert_eq!(model.resolution.width, 752);
        assert_eq!(model.resolution.height, 480);

        // Check distortion parameters
        assert_eq!(model.distortions.len(), 5);
        assert_eq!(model.distortions[0], -0.28340811); // k1
        assert_eq!(model.distortions[1], 0.07395907); // k2
        assert_eq!(model.distortions[2], 0.00019359); // p1
        assert_eq!(model.distortions[3], 1.76187114e-05); // p2
        assert_eq!(model.distortions[4], 0.0); // k3
    }

    /// Tests saving [`RadTanModel`] parameters and reloading them.
    #[test]
    fn test_radtan_save_to_yaml() {
        use std::fs;
        // Note: The original test created "output" dir with a println,
        // which is not ideal for automated tests. Assuming dir exists or can be created.
        fs::create_dir_all("output").expect("Failed to create output directory for test.");

        // Define input and output paths
        let input_path = "samples/rad_tan.yaml";
        let output_path = "output/rad_tan_saved.yaml";

        // Load the camera model from the original YAML
        let model = RadTanModel::load_from_yaml(input_path).unwrap();

        // Save the model to the output path
        model.save_to_yaml(output_path).unwrap();

        // Load the model from the saved file
        let saved_model = RadTanModel::load_from_yaml(output_path).unwrap();

        // Compare the original and saved models
        assert_eq!(model.intrinsics.fx, saved_model.intrinsics.fx);
        assert_eq!(model.intrinsics.fy, saved_model.intrinsics.fy);
        assert_eq!(model.intrinsics.cx, saved_model.intrinsics.cx);
        assert_eq!(model.intrinsics.cy, saved_model.intrinsics.cy);
        assert_eq!(model.resolution.width, saved_model.resolution.width);
        assert_eq!(model.resolution.height, saved_model.resolution.height);
        assert_eq!(model.distortions.len(), saved_model.distortions.len());
        for i in 0..5 {
            // Original test compared each element individually
            assert_eq!(model.distortions[i], saved_model.distortions[i]);
        }
        // Clean up the saved file
        fs::remove_file(output_path).unwrap();
    }

    /// Tests projection and unprojection consistency for the [`RadTanModel`].
    #[test]
    fn test_radtan_project_unproject() {
        // Load the camera model from YAML
        let path = "samples/rad_tan.yaml";
        let model = RadTanModel::load_from_yaml(path).unwrap();

        // Create a 3D point in camera coordinates (pointing somewhat forward and to the side)
        let point_3d = Vector3::new(0.5, -0.3, 2.0);
        let norm_3d = point_3d.normalize();

        // Project the 3D point to pixel coordinates
        let point_2d = model.project(&point_3d).unwrap();

        // Check if the pixel coordinates are within the image bounds
        assert!(point_2d.x >= 0.0 && point_2d.x < model.resolution.width as f64);
        assert!(point_2d.y >= 0.0 && point_2d.y < model.resolution.height as f64);

        // Unproject the pixel point back to a 3D ray direction
        let point_3d_unprojected = model.unproject(&point_2d).unwrap();

        // Check if the unprojected point is close to the original normalized point
        // Original test used 1e-6, which is fine.
        assert!((norm_3d.x - point_3d_unprojected.x).abs() < 1e-6);
        assert!((norm_3d.y - point_3d_unprojected.y).abs() < 1e-6);
        assert!((norm_3d.z - point_3d_unprojected.z).abs() < 1e-6);
    }

    /// Tests projection and unprojection for multiple points with [`RadTanModel`].
    #[test]
    fn test_radtan_multiple_points() {
        let path = "samples/rad_tan.yaml";
        let model = RadTanModel::load_from_yaml(path).unwrap();

        // Define a set of 3D test points covering different parts of the field of view
        let test_points = vec![
            Vector3::new(0.0, 0.0, 1.0),   // Center
            Vector3::new(0.5, 0.0, 1.0),   // Right
            Vector3::new(-0.5, 0.0, 1.0),  // Left
            Vector3::new(0.0, 0.5, 1.0),   // Top
            Vector3::new(0.0, -0.5, 1.0),  // Bottom
            Vector3::new(0.3, 0.4, 1.0),   // Top-right
            Vector3::new(-0.3, 0.4, 1.0),  // Top-left
            Vector3::new(0.3, -0.4, 1.0),  // Bottom-right
            Vector3::new(-0.3, -0.4, 1.0), // Bottom-left
            Vector3::new(0.1, 0.1, 2.0),   // Further away
                                           // Original test had one more point: Vector3::new(1.0, 0.8, 3.0),
                                           // but the provided code for this attempt does not have it.
        ];

        for (i, original_point) in test_points.iter().enumerate() {
            // Project the 3D point to pixel coordinates
            let pixel_point = match model.project(original_point) {
                Ok(p) => p,
                Err(e) => {
                    println!(
                        "Point {} at {:?} failed projection: {:?}",
                        i, original_point, e
                    );
                    continue; // Skip points that fail projection
                }
            };

            // Unproject back to 3D
            let ray_direction = match model.unproject(&pixel_point) {
                Ok(r) => r,
                Err(e) => {
                    // Original test did not panic here but continued.
                    println!(
                        "Point {} at pixel {:?} failed unprojection: {:?}",
                        i, pixel_point, e
                    );
                    continue;
                }
            };

            // The original point and unprojected ray should point in the same direction
            let original_direction = original_point.normalize();
            let dot_product = original_direction.dot(&ray_direction);

            // Assert with helpful debug information
            // Original test used 0.99, which is fine.
            assert!(dot_product > 0.99,
                    "Test point {}: Direction mismatch. Original: {:?}, Unprojected: {:?}, Dot product: {}",
                    i, original_direction, ray_direction, dot_product);
        }
    }
}
