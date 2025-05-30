//! Implements the Kannala-Brandt camera model, suitable for fisheye or wide-angle lenses.
//!
//! This module provides the [`KannalaBrandtModel`] struct and its associated methods
//! for representing and working with a camera that follows the Kannala-Brandt
//! distortion model. This model is often used for cameras with significant radial
//! distortion, such as fisheye lenses. It adheres to the [`CameraModel`] trait
//! defined in the parent `camera` module ([`crate::camera`]).
//!
//! # References
//!
//! The Kannala-Brandt model is based on the paper:
//! *   Kannala, J., & Brandt, S. S. (2006). A generic camera model and calibration
//!     method for conventional, wide-angle, and fish-eye lenses.
//!     *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *28*(8), 1335-1340.

use nalgebra::{DMatrix, DVector, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use std::{f64::consts::PI, fs, fmt, io::Write};
use yaml_rust::YamlLoader;

use crate::camera::{validation, CameraModel, CameraModelError, Intrinsics, Resolution};

/// Implements the Kannala-Brandt camera model for fisheye/wide-angle lenses.
///
/// This struct holds the intrinsic parameters ([`Intrinsics`]: fx, fy, cx, cy),
/// image [`Resolution`], and a set of 4 distortion coefficients (`k1, k2, k3, k4`)
/// that define the symmetric radial distortion characteristics of the lens according
/// to the Kannala-Brandt model. The projection function is a polynomial function of theta,
/// the angle between the 3D point and the optical axis.
///
/// # Fields
///
/// *   `intrinsics`: [`Intrinsics`] - Holds the focal lengths (fx, fy) and principal point (cx, cy).
/// *   `resolution`: [`Resolution`] - The width and height of the camera image in pixels.
/// *   `distortions`: `[f64; 4]` - The 4 distortion coefficients: `[k1, k2, k3, k4]`.
///     These coefficients model the radial distortion as a polynomial function of the angle theta.
///
/// # Examples
///
/// ```rust
/// use nalgebra::DVector;
/// use vision_toolkit_rs::camera::kannala_brandt::KannalaBrandtModel;
/// use vision_toolkit_rs::camera::{Intrinsics, Resolution, CameraModel, CameraModelError};
///
/// // Parameters: fx, fy, cx, cy, k1, k2, k3, k4
/// let params = DVector::from_vec(vec![
///     460.0, 460.0, 320.0, 240.0,    // Intrinsics
///     -0.01, 0.05, -0.08, 0.04     // Distortion (k1, k2, k3, k4)
/// ]);
/// let mut kb_model = KannalaBrandtModel::new(&params).unwrap();
/// kb_model.resolution = Resolution { width: 640, height: 480 };
///
/// println!("Created Kannala-Brandt model: {:?}", kb_model);
/// assert_eq!(kb_model.intrinsics.fx, 460.0);
/// assert_eq!(kb_model.distortions[0], -0.01); // k1
///
/// // Example of loading from a (hypothetical) YAML
/// // let model_from_yaml = KannalaBrandtModel::load_from_yaml("path/to/your_kb_camera.yaml");
/// // if let Ok(model) = model_from_yaml {
/// //     println!("Loaded model intrinsics: {:?}", model.get_intrinsics());
/// //     println!("Loaded model distortions: {:?}", model.get_distortion());
/// // }
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct KannalaBrandtModel {
    /// Camera intrinsic parameters: `fx`, `fy`, `cx`, `cy`.
    pub intrinsics: Intrinsics,
    /// Image resolution as width and height in pixels.
    pub resolution: Resolution,
    /// The 4 distortion coefficients: `[k1, k2, k3, k4]`.
    pub distortions: [f64; 4],
}

// Removed KannalaBrandtOptimizationCost struct and its trait implementations

impl KannalaBrandtModel {
    /// Creates a new [`KannalaBrandtModel`] from a DVector of parameters.
    ///
    /// This constructor initializes the model with the provided parameters.
    /// The image resolution is initialized to 0x0 and should be set explicitly
    /// or by loading from a configuration file like YAML.
    /// The current implementation of `new` does not call `validate_params`.
    ///
    /// # Arguments
    ///
    /// * `parameters`: A `&DVector<f64>` containing the camera parameters in the following order:
    ///   1.  `fx`: Focal length along the x-axis.
    ///   2.  `fy`: Focal length along the y-axis.
    ///   3.  `cx`: Principal point x-coordinate.
    ///   4.  `cy`: Principal point y-coordinate.
    ///   5.  `k1`: First distortion coefficient.
    ///   6.  `k2`: Second distortion coefficient.
    ///   7.  `k3`: Third distortion coefficient.
    ///   8.  `k4`: Fourth distortion coefficient.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Self, CameraModelError>`.
    ///
    /// # Errors
    ///
    /// * [`CameraModelError::InvalidParams`]: If the `parameters` vector does not contain exactly 8 elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::DVector;
    /// use vision_toolkit_rs::camera::kannala_brandt::KannalaBrandtModel;
    /// use vision_toolkit_rs::camera::Resolution;
    ///
    /// let params_vec = DVector::from_vec(vec![
    ///     461.58, 460.28, 366.28, 249.08, // fx, fy, cx, cy
    ///     -0.012, 0.057, -0.084, 0.043    // k1, k2, k3, k4
    /// ]);
    /// let mut model = KannalaBrandtModel::new(&params_vec).unwrap();
    /// model.resolution = Resolution { width: 752, height: 480 }; // Set resolution manually
    ///
    /// assert_eq!(model.intrinsics.fx, 461.58);
    /// assert_eq!(model.distortions[0], -0.012); // k1
    /// assert_eq!(model.resolution.width, 752);
    /// ```
    pub fn new(parameters: &DVector<f64>) -> Result<Self, CameraModelError> {
        if parameters.len() != 8 {
            return Err(CameraModelError::InvalidParams(format!(
                "Expected 8 parameters, got {}",
                parameters.len()
            )));
        }
        let model = KannalaBrandtModel {
            intrinsics: Intrinsics {
                fx: parameters[0],
                fy: parameters[1],
                cx: parameters[2],
                cy: parameters[3],
            },
            resolution: Resolution {
                // Resolution is typically set from YAML or context, not parameters vector
                width: 0,
                height: 0,
            },
            distortions: [parameters[4], parameters[5], parameters[6], parameters[7]],
        };

        // Basic validation, full validation might depend on resolution being set.
        // model.validate_params()?; // Cannot fully validate without resolution
        // Documenting current behavior: validate_params is not called here.
        Ok(model)
    }
}

/// Provides a debug string representation for [`KannalaBrandtModel`].
impl fmt::Debug for KannalaBrandtModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DoubleSphere [fx: {} fy: {} cx: {} cy: {} distortions: {:?}]",
            self.intrinsics.fx,
            self.intrinsics.fy,
            self.intrinsics.cx,
            self.intrinsics.cy,
            self.distortions,
        )
    }
}

impl CameraModel for KannalaBrandtModel {
    /// Projects a 3D point from camera coordinates to 2D image coordinates.
    ///
    /// The projection uses the Kannala-Brandt model:
    /// 1.  Calculates `r = sqrt(x^2 + y^2)`.
    /// 2.  Calculates `theta = atan2(r, z)`.
    /// 3.  Applies the polynomial distortion model: `theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)`.
    /// 4.  Projects to image plane: `u = fx * theta_d * (x/r) + cx`, `v = fy * theta_d * (y/r) + cy`.
    /// The Jacobian can optionally be computed with respect to the 8 model parameters
    /// (fx, fy, cx, cy, k1, k2, k3, k4).
    ///
    /// # Arguments
    ///
    /// * `point_3d`: A `&Vector3<f64>` representing the 3D point (X, Y, Z) in camera coordinates.
    /// * `compute_jacobian`: A boolean flag. If true, the Jacobian of the projection
    ///   with respect to the camera parameters is computed.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<(Vector2<f64>, Option<DMatrix<f64>>), CameraModelError>`.
    /// On success, it provides a tuple containing:
    /// *   The projected 2D point (`Vector2<f64>`) in pixel coordinates (u, v).
    /// *   An `Option<DMatrix<f64>>` which is `Some(jacobian)` if `compute_jacobian` was true
    ///     (2x8 matrix), or `None` otherwise.
    ///
    /// # Errors
    ///
    /// * [`CameraModelError::PointIsOutSideImage`]: If the 3D point's Z-coordinate is negative.
    /// * [`CameraModelError::PointAtCameraCenter`]: If the 3D point's Z-coordinate is very close to zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::{DVector, Vector3};
    /// use vision_toolkit_rs::camera::kannala_brandt::KannalaBrandtModel;
    /// use vision_toolkit_rs::camera::{CameraModel, Resolution};
    ///
    /// let params = DVector::from_vec(vec![460.0,460.0,320.0,240.0, -0.01,0.05,-0.08,0.04]);
    /// let mut model = KannalaBrandtModel::new(&params).unwrap();
    /// model.resolution = Resolution { width: 640, height: 480 };
    ///
    /// let point_3d = Vector3::new(0.1, 0.2, 1.0);
    /// match model.project(&point_3d, false) {
    ///     Ok((point_2d, jac_opt)) => {
    ///         println!("Projected: ({}, {})", point_2d.x, point_2d.y);
    ///         assert!(jac_opt.is_none());
    ///         assert!(point_2d.x > 0.0 && point_2d.y > 0.0); // Basic check
    ///     },
    ///     Err(e) => panic!("Projection failed: {:?}", e),
    /// }
    /// ```
    fn project(
        &self,
        point_3d: &Vector3<f64>,
        compute_jacobian: bool,
    ) -> Result<(Vector2<f64>, Option<DMatrix<f64>>), CameraModelError> {
        let x = point_3d.x;
        let y = point_3d.y;
        let z = point_3d.z;

        if z < 0.0 {
            // Point is behind the image plane (or camera)
            return Err(CameraModelError::PointIsOutSideImage);
        } else if z < f64::EPSILON {
            // Point is at or extremely close to the camera center (z >= 0 but very small)
            return Err(CameraModelError::PointAtCameraCenter);
        }

        let k1 = self.distortions[0];
        let k2 = self.distortions[1];
        let k3 = self.distortions[2];
        let k4 = self.distortions[3];

        let fx = self.intrinsics.fx;
        let fy = self.intrinsics.fy;
        let cx = self.intrinsics.cx;
        let cy = self.intrinsics.cy;

        let r_sq = x * x + y * y;
        let r = r_sq.sqrt();
        let theta = r.atan2(z); // atan2(y,x) in nalgebra is atan2(self, other) -> atan2(r,z)

        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        let theta_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9;

        let (x_r, y_r) = if r < f64::EPSILON {
            // Use a small epsilon for r
            // If r is very small, point is close to optical axis.
            // x_r and y_r would be ill-defined or lead to instability.
            // For many fisheye models, the projection at r=0 is (cx, cy).
            // Here, theta_d * x_r and theta_d * y_r would be 0.
            // Let's assume for r=0, x_r and y_r are effectively 0 for the multiplication.
            // Or, if theta_d is also 0 (theta=0), then it's 0.
            // If theta=0 (on optical axis), theta_d = 0. So proj_x = cx, proj_y = cy.
            // This seems consistent.
            (0.0, 0.0)
        } else {
            (x / r, y / r)
        };

        let proj_x = fx * theta_d * x_r + cx;
        let proj_y = fy * theta_d * y_r + cy;
        let point_2d = Vector2::new(proj_x, proj_y);

        let mut jacobian_option: Option<DMatrix<f64>> = None;

        if compute_jacobian {
            let mut jacobian = DMatrix::zeros(2, 8); // fx, fy, cx, cy, k1, k2, k3, k4

            // Jacobian calculation based on C++ snippet logic
            // (x_r, y_r) are already computed, using the same epsilon logic as projection

            // Column 0: dfx
            jacobian[(0, 0)] = theta_d * x_r;
            jacobian[(1, 0)] = 0.0;

            // Column 1: dfy
            jacobian[(0, 1)] = 0.0;
            jacobian[(1, 1)] = theta_d * y_r;

            // Column 2: dcx
            jacobian[(0, 2)] = 1.0;
            jacobian[(1, 2)] = 0.0;

            // Column 3: dcy
            jacobian[(0, 3)] = 0.0;
            jacobian[(1, 3)] = 1.0;

            // Columns 4-7: dk1, dk2, dk3, dk4
            // This part matches the C++ logic: de_dp * dp_dd_theta * dd_theta_dks
            // where de_dp = [[fx, 0], [0, fy]]
            //       dp_dd_theta = [x_r, y_r]^T
            //       dd_theta_dks = [theta3, theta5, theta7, theta9]
            // Resulting in:
            // jacobian.rightCols<4>() = [fx * x_r; fy * y_r] * [theta3, theta5, theta7, theta9]

            let fx_x_r_term = fx * x_r;
            let fy_y_r_term = fy * y_r;

            jacobian[(0, 4)] = fx_x_r_term * theta3; // d(proj_x)/dk1
            jacobian[(1, 4)] = fy_y_r_term * theta3; // d(proj_y)/dk1

            jacobian[(0, 5)] = fx_x_r_term * theta5; // d(proj_x)/dk2
            jacobian[(1, 5)] = fy_y_r_term * theta5; // d(proj_y)/dk2

            jacobian[(0, 6)] = fx_x_r_term * theta7; // d(proj_x)/dk3
            jacobian[(1, 6)] = fy_y_r_term * theta7; // d(proj_y)/dk3

            jacobian[(0, 7)] = fx_x_r_term * theta9; // d(proj_x)/dk4
            jacobian[(1, 7)] = fy_y_r_term * theta9; // d(proj_y)/dk4

            jacobian_option = Some(jacobian);
        }

        Ok((point_2d, jacobian_option))
    }

    /// Unprojects a 2D image point to a 3D ray in camera coordinates.
    ///
    /// This method applies the inverse Kannala-Brandt model. It first converts
    /// the pixel coordinates `(u,v)` to normalized, distorted coordinates `(mx, my)`.
    /// The radial distance `ru = sqrt(mx^2 + my^2)` corresponds to `theta_d`.
    /// The core of the unprojection is to find `theta` (the angle of the 3D ray with the
    /// optical axis) by iteratively solving the polynomial equation
    /// `theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)`
    /// for `theta`, typically using Newton-Raphson method.
    /// Once `theta` is found, the 3D ray `(x, y, z)` is constructed as
    /// `x = sin(theta) * (mx/ru)`, `y = sin(theta) * (my/ru)`, `z = cos(theta)`,
    /// and then normalized.
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
    /// * [`CameraModelError::PointIsOutSideImage`]: If the input 2D point is outside the camera's
    ///   resolution (if resolution is set and > 0).
    /// * [`CameraModelError::NumericalError`]: If the iterative Newton-Raphson method fails
    ///   to converge when solving for `theta`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::{DVector, Vector2};
    /// use vision_toolkit_rs::camera::kannala_brandt::KannalaBrandtModel;
    /// use vision_toolkit_rs::camera::{CameraModel, Resolution};
    ///
    /// let params = DVector::from_vec(vec![460.0,460.0,320.0,240.0, -0.01,0.05,-0.08,0.04]);
    /// let mut model = KannalaBrandtModel::new(&params).unwrap();
    /// model.resolution = Resolution { width: 640, height: 480 };
    ///
    /// let point_2d = Vector2::new(330.0, 250.0); // A point near the center
    /// match model.unproject(&point_2d) {
    ///     Ok(ray_3d) => {
    ///         println!("Unprojected ray: ({}, {}, {})", ray_3d.x, ray_3d.y, ray_3d.z);
    ///         assert!((ray_3d.norm() - 1.0).abs() < 1e-6); // Should be a unit vector
    ///     },
    ///     Err(e) => panic!("Unprojection failed: {:?}", e),
    /// }
    /// ```
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        // Check if point is outside resolution, if resolution is set
        if self.resolution.width > 0 && self.resolution.height > 0 {
            if point_2d.x < 0.0
                || point_2d.x >= self.resolution.width as f64
                || point_2d.y < 0.0
                || point_2d.y >= self.resolution.height as f64
            {
                return Err(CameraModelError::PointIsOutSideImage);
            }
        }

        let u = point_2d.x;
        let v = point_2d.y;

        let mx = (u - self.intrinsics.cx) / self.intrinsics.fx;
        let my = (v - self.intrinsics.cy) / self.intrinsics.fy;

        let mut ru = (mx * mx + my * my).sqrt();

        // Clamp ru (distorted theta, theta_d) as per C++: min(max(-PI/2, ru), PI/2)
        // Since ru is sqrt, it's >= 0. So effectively min(ru, PI/2).
        ru = ru.min(PI / 2.0);

        let mut theta = ru; // Initial guess for undistorted theta
        const PRECISION: f64 = 1e-6; // Adjusted precision from C++ 1e-3
        const MAX_ITERATION: usize = 10;
        let mut converged = true;

        if ru > PRECISION {
            // Only run Newton-Raphson if ru is significantly large
            let k1 = self.distortions[0];
            let k2 = self.distortions[1];
            let k3 = self.distortions[2];
            let k4 = self.distortions[3];

            for _i in 0..MAX_ITERATION {
                let theta2 = theta * theta;
                let theta4 = theta2 * theta2;
                let theta6 = theta4 * theta2;
                let theta8 = theta4 * theta4;

                let k1_theta2 = k1 * theta2;
                let k2_theta4 = k2 * theta4;
                let k3_theta6 = k3 * theta6;
                let k4_theta8 = k4 * theta8;

                // f(theta) = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8) - ru
                let f = theta * (1.0 + k1_theta2 + k2_theta4 + k3_theta6 + k4_theta8) - ru;
                // f'(theta) = (1 + k1*theta^2 + ...) + theta * (2*k1*theta + 4*k2*theta^3 + ...)
                // f'(theta) = 1 + 3*k1*theta^2 + 5*k2*theta^4 + 7*k3*theta^6 + 9*k4*theta^8
                let f_prime = 1.0
                    + (3.0 * k1_theta2)
                    + (5.0 * k2_theta4)
                    + (7.0 * k3_theta6)
                    + (9.0 * k4_theta8);

                if f_prime.abs() < f64::EPSILON {
                    // Avoid division by zero
                    converged = false;
                    break;
                }
                let delta = f / f_prime;
                theta -= delta;

                if delta.abs() < PRECISION {
                    break;
                }
                if _i == MAX_ITERATION - 1 {
                    // Check if max iterations reached
                    converged = false;
                }
            }
        } else {
            // If ru is very small, theta is also small.
            // If ru <= PRECISION, C++ code sets converged = false.
            // However, if ru is small, theta is likely also small (close to ru).
            // For ru = 0, theta = 0.
            // Let's follow C++: if ru <= PRECISION, treat as not converged for safety,
            // or handle theta = ru directly if that's more appropriate.
            // C++ sets converged = false if ru <= PRECISION.
            if ru > 0.0 {
                // if ru is > 0 but <= PRECISION, C++ implies it's not converged.
                converged = false;
            } else {
                // ru is 0.0 or very close, theta should be 0.0
                theta = 0.0;
                converged = true; // For ru=0, theta=0 is a valid solution.
            }
        }

        if !converged {
            return Err(CameraModelError::NumericalError(
                "Unprojection failed to converge".to_string(),
            ));
        }

        // Check for ru being zero before division, if theta is non-zero.
        // If theta is zero (on optical axis), sin(theta) is zero, so x and y are zero.
        // If ru is zero, mx and my must be zero.
        let (x_comp, y_comp) = if ru.abs() < f64::EPSILON {
            // If ru is zero, mx and my must be zero. Point is at principal point.
            // sin(theta)/ru is like sin(x)/x -> 1 as x->0.
            // So x_comp = cos(theta), y_comp = cos(theta) if theta is also 0.
            // More simply, if ru=0, then mx=0, my=0.
            // Then point3d.x = sin(theta)*0 = 0, point3d.y = sin(theta)*0 = 0.
            // This is correct for a point at the principal point.
            (0.0, 0.0)
        } else {
            (mx / ru, my / ru)
        };

        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        let point3d = Vector3::new(sin_theta * x_comp, sin_theta * y_comp, cos_theta);
        Ok(point3d.normalize()) // Ensure unit vector
    }

    /// Loads [`KannalaBrandtModel`] parameters from a YAML file.
    ///
    /// The YAML file is expected to follow a structure where camera parameters are nested
    /// under `cam0`. The intrinsic parameters (`fx`, `fy`, `cx`, `cy`) are read from
    /// `cam0.intrinsics`, and the Kannala-Brandt distortion coefficients (`k1, k2, k3, k4`)
    /// are read from `cam0.distortion`. The `resolution` (width, height) is also
    /// expected under `cam0.resolution`.
    ///
    /// # Arguments
    ///
    /// * `path`: A string slice representing the path to the YAML file.
    ///
    /// # Return Value
    ///
    /// Returns a `Result<Self, CameraModelError>`. On success, it provides an instance
    /// of [`KannalaBrandtModel`] populated with parameters from the file.
    ///
    /// # Errors
    ///
    /// This function can return:
    /// * [`CameraModelError::IOError`]: If there's an issue reading the file.
    /// * [`CameraModelError::YamlError`]: If the YAML content is malformed or cannot be parsed.
    /// * [`CameraModelError::InvalidParams`]: If the YAML structure is missing expected fields
    ///   (e.g., "cam0", "intrinsics", "resolution", "distortion") or if parameter values
    ///   are of incorrect types or counts.
    /// * Errors from [`KannalaBrandtModel::validate_params()`] if the loaded intrinsic parameters are invalid.
    ///
    /// # Related
    /// * [`KannalaBrandtModel::save_to_yaml()`]
    fn load_from_yaml(path: &str) -> Result<Self, CameraModelError> {
        let contents = fs::read_to_string(path)?;
        let docs = YamlLoader::load_from_str(&contents)?;
        if docs.is_empty() {
            return Err(CameraModelError::InvalidParams(
                "Empty YAML document".to_string(),
            ));
        }
        let doc = &docs[0];

        let cam_node = &doc["cam0"];
        if cam_node.is_badvalue() {
            return Err(CameraModelError::InvalidParams(
                "Missing 'cam0' node in YAML".to_string(),
            ));
        }

        let intrinsics_yaml = cam_node["intrinsics"].as_vec().ok_or_else(|| {
            CameraModelError::InvalidParams("Invalid intrinsics format in YAML: not an array".to_string())
        })?;
        if intrinsics_yaml.len() < 4 {
            return Err(CameraModelError::InvalidParams(
                "Intrinsics array in YAML must have at least 4 elements (fx, fy, cx, cy)".to_string(),
            ));
        }

        let resolution_yaml = cam_node["resolution"].as_vec().ok_or_else(|| {
            CameraModelError::InvalidParams("Invalid resolution format in YAML: not an array".to_string())
        })?;
        if resolution_yaml.len() < 2 {
            return Err(CameraModelError::InvalidParams(
                "Resolution array in YAML must have at least 2 elements (width, height)".to_string(),
            ));
        }

        // Note: YAML field name for distortion in load is "distortion"
        let distortion_coeffs_yaml = cam_node["distortion"].as_vec().ok_or_else(|| {
            CameraModelError::InvalidParams(
                "Missing or invalid 'distortion' array in YAML".to_string(),
            )
        })?;
        if distortion_coeffs_yaml.len() < 4 {
            return Err(CameraModelError::InvalidParams(
                "'distortion' array in YAML must have at least 4 elements (k1, k2, k3, k4)".to_string(),
            ));
        }

        let intrinsics = Intrinsics {
            fx: intrinsics_yaml[0]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid fx in YAML: not a float".to_string()))?,
            fy: intrinsics_yaml[1]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid fy in YAML: not a float".to_string()))?,
            cx: intrinsics_yaml[2]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid cx in YAML: not a float".to_string()))?,
            cy: intrinsics_yaml[3]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid cy in YAML: not a float".to_string()))?,
        };

        let resolution = Resolution {
            width: resolution_yaml[0]
                .as_i64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid width in YAML: not an integer".to_string()))?
                as u32,
            height: resolution_yaml[1]
                .as_i64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid height in YAML: not an integer".to_string()))?
                as u32,
        };

        let distortions = [
            distortion_coeffs_yaml[0]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid k1 in YAML: not a float".to_string()))?,
            distortion_coeffs_yaml[1]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid k2 in YAML: not a float".to_string()))?,
            distortion_coeffs_yaml[2]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid k3 in YAML: not a float".to_string()))?,
            distortion_coeffs_yaml[3]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid k4 in YAML: not a float".to_string()))?,
        ];

        let model = KannalaBrandtModel {
            intrinsics,
            resolution,
            distortions,
        };

        model.validate_params()?;
        Ok(model)
    }

    /// Saves the [`KannalaBrandtModel`] parameters to a YAML file.
    ///
    /// The parameters are saved under the `cam0` key. The intrinsic parameters
    /// (`fx, fy, cx, cy`) are saved in the `intrinsics` array. The distortion
    /// coefficients (`k1, k2, k3, k4`) are saved in the `distortion_coeffs` array.
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
    /// * [`KannalaBrandtModel::load_from_yaml()`]
    fn save_to_yaml(&self, path: &str) -> Result<(), CameraModelError> {
        let mut cam0_map = serde_yaml::Mapping::new();
        cam0_map.insert(
            serde_yaml::Value::String("camera_model".to_string()),
            serde_yaml::Value::String("kannala_brandt".to_string()),
        );
        cam0_map.insert(
            serde_yaml::Value::String("intrinsics".to_string()),
            serde_yaml::to_value(vec![
                self.intrinsics.fx,
                self.intrinsics.fy,
                self.intrinsics.cx,
                self.intrinsics.cy,
            ])
            .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
        );
        // Note: YAML field name for distortion in save is "distortion_coeffs"
        cam0_map.insert(
            serde_yaml::Value::String("distortion_coeffs".to_string()),
            serde_yaml::to_value(self.distortions.to_vec())
                .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
        );
        cam0_map.insert(
            serde_yaml::Value::String("resolution".to_string()),
            serde_yaml::to_value(vec![self.resolution.width, self.resolution.height])
                .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
        );
        // Optional: Add rostopic if common for your use case
        // cam0_map.insert(
        //     serde_yaml::Value::String("rostopic".to_string()),
        //     serde_yaml::Value::String("/cam0/image_raw".to_string()),
        // );

        let mut root_map = serde_yaml::Mapping::new();
        root_map.insert(
            serde_yaml::Value::String("cam0".to_string()),
            serde_yaml::Value::Mapping(cam0_map),
        );

        let yaml_string = serde_yaml::to_string(&root_map)
            .map_err(|e| CameraModelError::YamlError(e.to_string()))?;

        let mut file =
            fs::File::create(path).map_err(|e| CameraModelError::IOError(e.to_string()))?;
        file.write_all(yaml_string.as_bytes())
            .map_err(|e| CameraModelError::IOError(e.to_string()))?;
        Ok(())
    }

    /// Validates the intrinsic parameters of the [`KannalaBrandtModel`].
    ///
    /// This method primarily checks the validity of the core intrinsic parameters
    /// (focal lengths, principal point) using [`validation::validate_intrinsics`].
    /// Currently, there are no specific validation rules implemented for the
    /// Kannala-Brandt distortion coefficients (`k1` through `k4`) themselves, as they
    /// can typically be positive or negative.
    ///
    /// # Return Value
    ///
    /// Returns `Ok(())` if the intrinsic parameters are valid.
    ///
    /// # Errors
    ///
    /// Returns a [`CameraModelError`] if intrinsic parameter validation fails, as propagated
    /// from [`validation::validate_intrinsics`] (e.g.,
    /// [`CameraModelError::FocalLengthMustBePositive`],
    /// [`CameraModelError::PrincipalPointMustBeFinite`]).
    fn validate_params(&self) -> Result<(), CameraModelError> {
        validation::validate_intrinsics(&self.intrinsics)?;
        // No specific validation for Kannala-Brandt distortions k1-k4 mentioned in common practice,
        // as they can be positive or negative. Their impact is on the projection function.
        Ok(())
    }

    /// Returns a clone of the camera's image resolution.
    ///
    /// # Return Value
    /// A [`Resolution`] struct containing the width and height of the camera image.
    fn get_resolution(&self) -> Resolution {
        self.resolution.clone()
    }

    /// Returns a clone of the camera's intrinsic parameters.
    ///
    /// # Return Value
    /// An [`Intrinsics`] struct containing `fx`, `fy`, `cx`, and `cy`.
    fn get_intrinsics(&self) -> Intrinsics {
        self.intrinsics.clone()
    }

    /// Returns a vector containing the distortion coefficients of the camera.
    ///
    /// The coefficients are returned in the order: `[k1, k2, k3, k4]`.
    ///
    /// # Return Value
    /// A `Vec<f64>` containing the 4 distortion coefficients.
    fn get_distortion(&self) -> Vec<f64> {
        self.distortions.to_vec()
    }

    // linear_estimation removed from impl CameraModel for KannalaBrandtModel
    // optimize removed from impl CameraModel for KannalaBrandtModel
}

/// Unit tests for the [`KannalaBrandtModel`].
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Helper function to create a sample [`KannalaBrandtModel`] instance for testing.
    /// Parameters are taken from "samples/kannala_brandt.yaml".
    fn get_sample_kb_model() -> KannalaBrandtModel {
        // Parameters from samples/kannala_brandt.yaml
        // intrinsics: [461.58688085556616, 460.2811732644195, 366.28603126815506, 249.08026891791644]
        // distortion: [-0.012523386218579752, 0.057836801948828065, -0.08495347810986263, 0.04362766880887814]
        // resolution: [752, 480]
        let params = DVector::from_vec(vec![
            461.58688085556616,    // fx
            460.2811732644195,     // fy
            366.28603126815506,    // cx
            249.08026891791644,    // cy
            -0.012523386218579752, // k1
            0.057836801948828065,  // k2
            -0.08495347810986263,  // k3
            0.04362766880887814,   // k4
        ]);
        let mut model = KannalaBrandtModel::new(&params).unwrap();
        model.resolution = Resolution {
            width: 752,
            height: 480,
        };
        model
    }

    /// Tests loading [`KannalaBrandtModel`] parameters from "samples/kannala_brandt.yaml".
    #[test]
    fn test_load_from_yaml_ok() {
        // Load the camera model from YAML
        let path = "samples/kannala_brandt.yaml";
        let model = KannalaBrandtModel::load_from_yaml(path).unwrap();

        assert_relative_eq!(model.intrinsics.fx, 461.58688085556616, epsilon = 1e-9);
        assert_relative_eq!(model.intrinsics.fy, 460.2811732644195, epsilon = 1e-9);
        assert_relative_eq!(model.intrinsics.cx, 366.28603126815506, epsilon = 1e-9);
        assert_relative_eq!(model.intrinsics.cy, 249.08026891791644, epsilon = 1e-9);

        assert_eq!(model.resolution.width, 752);
        assert_eq!(model.resolution.height, 480);

        assert_relative_eq!(model.distortions[0], -0.012523386218579752, epsilon = 1e-9); // k1
        assert_relative_eq!(model.distortions[1], 0.057836801948828065, epsilon = 1e-9); // k2
        assert_relative_eq!(model.distortions[2], -0.08495347810986263, epsilon = 1e-9); // k3
        assert_relative_eq!(model.distortions[3], 0.04362766880887814, epsilon = 1e-9);  // k4
    }

    /// Tests loading from a non-existent YAML file, expecting an I/O error.
    #[test]
    fn test_load_from_yaml_file_not_found() {
        let model_result = KannalaBrandtModel::load_from_yaml("samples/non_existent_file.yaml");
        assert!(model_result.is_err());
        match model_result.err().unwrap() {
            CameraModelError::IOError(_) => {} // Expected
            other_error => panic!("Expected IOError, got {:?}", other_error),
        }
    }

    /// Tests the consistency of projection and unprojection for the [`KannalaBrandtModel`].
    #[test]
    fn test_project_unproject_identity() {
        let model = get_sample_kb_model();
        let point_3d = Vector3::new(0.1, 0.2, 1.0); // A sample point in front of the camera

        match model.project(&point_3d, false) {
            Ok((point_2d, _)) => {
                // Ensure the projected point is within image bounds (or check if it should be)
                // This depends on the specific point and camera params
                assert!(
                    point_2d.x >= 0.0 && point_2d.x < model.resolution.width as f64,
                    "x out of bounds: {}",
                    point_2d.x
                );
                assert!(
                    point_2d.y >= 0.0 && point_2d.y < model.resolution.height as f64,
                    "y out of bounds: {}",
                    point_2d.y
                );

                match model.unproject(&point_2d) {
                    Ok(unprojected_point_3d) => {
                        // Unprojection usually returns a unit vector
                        // Compare with the original point's direction
                        let point_3d_normalized = point_3d.normalize();
                        assert_relative_eq!(
                            unprojected_point_3d.x,
                            point_3d_normalized.x,
                            epsilon = 1e-5 // Using a slightly larger epsilon due to iterative nature
                        );
                        assert_relative_eq!(
                            unprojected_point_3d.y,
                            point_3d_normalized.y,
                            epsilon = 1e-5
                        );
                        assert_relative_eq!(
                            unprojected_point_3d.z,
                            point_3d_normalized.z,
                            epsilon = 1e-5
                        );
                    }
                    Err(e) => panic!("Unprojection failed: {:?}", e),
                }
            }
            Err(e) => panic!("Projection failed: {:?}", e),
        }
    }

    /// Tests projection of a point exactly at the camera's optical center (0,0,0).
    #[test]
    fn test_project_point_at_center() {
        let model = get_sample_kb_model();
        let point_3d = Vector3::new(0.0, 0.0, 0.0); // Point at camera optical center
        let result = model.project(&point_3d, false);
        assert!(matches!(result, Err(CameraModelError::PointAtCameraCenter)));
    }

    /// Tests projection of a point located behind the camera.
    #[test]
    fn test_project_point_behind_camera() {
        let model = get_sample_kb_model();
        let point_3d = Vector3::new(0.1, 0.2, -1.0); // Point behind camera
        let result = model.project(&point_3d, false);
        assert!(matches!(result, Err(CameraModelError::PointIsOutSideImage)));
    }

    /// Tests projection with Jacobian computation.
    #[test]
    fn test_project_with_jacobian() {
        let model = get_sample_kb_model();
        let point_3d = Vector3::new(0.1, 0.2, 1.0);

        match model.project(&point_3d, true) {
            Ok((_point_2d, jacobian_option)) => {
                assert!(jacobian_option.is_some(), "Jacobian should be Some when requested");
                let jacobian = jacobian_option.unwrap();
                assert_eq!(jacobian.nrows(), 2, "Jacobian should have 2 rows (for u, v)");
                assert_eq!(jacobian.ncols(), 8, "Jacobian should have 8 columns (for fx,fy,cx,cy,k1,k2,k3,k4)");
                // Further checks on Jacobian values would require numerical differentiation
                // or known analytical values for specific points.
            }
            Err(e) => panic!("Projection failed: {:?}", e),
        }
    }

    /// Tests unprojection of a point outside the image resolution bounds.
    #[test]
    fn test_unproject_out_of_bounds() {
        let model = get_sample_kb_model();
        let point_2d_outside = Vector2::new(
            model.resolution.width as f64 + 10.0, // 10 pixels outside width
            model.resolution.height as f64 + 10.0, // 10 pixels outside height
        );
        let result = model.unproject(&point_2d_outside);
        assert!(matches!(result, Err(CameraModelError::PointIsOutSideImage)));
    }

    /// Tests the getter methods: `get_intrinsics`, `get_resolution`, and `get_distortion`.
    #[test]
    fn test_getters() {
        let model = get_sample_kb_model();

        let intrinsics = model.get_intrinsics();
        assert_relative_eq!(intrinsics.fx, 461.58688085556616);
        assert_relative_eq!(intrinsics.fy, 460.2811732644195);
        assert_relative_eq!(intrinsics.cx, 366.28603126815506);
        assert_relative_eq!(intrinsics.cy, 249.08026891791644);

        let resolution = model.get_resolution();
        assert_eq!(resolution.width, 752);
        assert_eq!(resolution.height, 480);

        let distortion_coeffs = model.get_distortion();
        assert_eq!(distortion_coeffs.len(), 4);
        assert_relative_eq!(distortion_coeffs[0], -0.012523386218579752); // k1
        assert_relative_eq!(distortion_coeffs[1], 0.057836801948828065); // k2
        assert_relative_eq!(distortion_coeffs[2], -0.08495347810986263); // k3
        assert_relative_eq!(distortion_coeffs[3], 0.04362766880887814);   // k4
    }

    // --- Tests for linear_estimation and optimize ---
    // These tests are more complex and depend on the full implementation
    // of these methods and potentially argmin cost functions.
    // Below are sketches of what these tests might look like.

    // fn generate_synthetic_data(
    //     model: &KannalaBrandtModel,
    //     num_points: usize,
    // ) -> (Matrix3xX<f64>, Matrix2xX<f64>) {
    //     let mut points_3d_vec = Vec::new();
    //     let mut points_2d_vec = Vec::new();

    //     // Generate some 3D points in a reasonable FOV
    //     for i in 0..num_points {
    //         let x = (i as f64 * 0.1) - (num_points as f64 * 0.05); // Spread points
    //         let y = (i as f64 * 0.05) - (num_points as f64 * 0.025);
    //         let z = 1.0 + (i as f64 * 0.01); // Vary depth
    //         let p3d = Vector3::new(x, y, z);

    //         if let Ok((p2d, _)) = model.project(&p3d, false) {
    //             // Ensure point is within image (simplified check)
    //             if p2d.x > 0.0
    //                 && p2d.x < model.resolution.width as f64
    //                 && p2d.y > 0.0
    //                 && p2d.y < model.resolution.height as f64
    //             {
    //                 points_3d_vec.push(p3d);
    //                 points_2d_vec.push(p2d);
    //             }
    //         }
    //     }

    //     let points_3d = Matrix3xX::from_columns(&points_3d_vec);
    //     let points_2d = Matrix2xX::from_columns(&points_2d_vec);
    //     (points_3d, points_2d)
    // }

    // #[test]
    // fn test_linear_estimation() {
    //     let ground_truth_model = get_sample_kb_model();
    //     let (points_3d, points_2d) = generate_synthetic_data(&ground_truth_model, 20);

    //     // Assume intrinsics (fx, fy, cx, cy) are known for linear estimation of distortion
    //     let estimated_model_result = KannalaBrandtModel::linear_estimation(
    //         &ground_truth_model.intrinsics,
    //         &ground_truth_model.resolution,
    //         &points_2d,
    //         &points_3d,
    //     );

    //     assert!(
    //         estimated_model_result.is_ok(),
    //         "Linear estimation failed: {:?}",
    //         estimated_model_result.err()
    //     );
    //     let estimated_model = estimated_model_result.unwrap();

    //     // Compare estimated distortion with ground truth
    //     // Linear estimation might not be perfectly accurate, so use a larger epsilon
    //     for i in 0..4 {
    //         assert_relative_eq!(
    //             estimated_model.distortions[i],
    //             ground_truth_model.distortions[i],
    //             epsilon = 1e-1 // Example epsilon, adjust based on expected accuracy
    //         );
    //     }
    // }

    // For optimize test, you would need:
    // 1. KannalaBrandtOptimizationCost struct (similar to DoubleSphereOptimizationCost)
    //    - Implementing argmin traits: Operator, Jacobian, CostFunction, Gradient, Hessian
    // 2. The optimize method in KannalaBrandtModel using argmin (e.g., GaussNewton solver)

    // #[test]
    // fn test_optimize_trait_method() { // Renamed to avoid conflict if original test_optimize is kept
    //     let ground_truth_model = get_sample_kb_model();
    //     let (points_3d, points_2d) = generate_synthetic_data(&ground_truth_model, 50);

    //     // Create an initial model with slightly perturbed parameters
    //     let mut initial_model = ground_truth_model.clone();
    //     initial_model.intrinsics.fx *= 1.05;
    //     initial_model.intrinsics.cy *= 0.95;
    //     initial_model.distortions[0] *= 1.2;
    //     initial_model.distortions[2] *= 0.8;

    //     // Use the Optimizer trait's optimize method
    //     let optimize_result = Optimizer::optimize(&mut initial_model, &points_3d, &points_2d, false);
    //     assert!(optimize_result.is_ok(), "Optimization failed: {:?}", optimize_result.err());

    //     let optimized_model = initial_model; // optimize should modify in place

    //     // Compare optimized parameters with ground truth
    //     assert_relative_eq!(optimized_model.intrinsics.fx, ground_truth_model.intrinsics.fx, epsilon = 1e-3);
    //     assert_relative_eq!(optimized_model.intrinsics.fy, ground_truth_model.intrinsics.fy, epsilon = 1e-3);
    //     assert_relative_eq!(optimized_model.intrinsics.cx, ground_truth_model.intrinsics.cx, epsilon = 1e-3);
    //     assert_relative_eq!(optimized_model.intrinsics.cy, ground_truth_model.intrinsics.cy, epsilon = 1e-3);

    //     for i in 0..4 {
    //         assert_relative_eq!(optimized_model.distortions[i], ground_truth_model.distortions[i], epsilon = 1e-3);
    //     }
    // }
}
