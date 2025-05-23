//! Double Sphere Camera Model Implementation
//!
//! This module implements the Double Sphere camera model, which is particularly useful
//! for wide-angle and fisheye cameras. The model uses two sphere projections to handle
//! the distortion characteristics of such cameras.
//!
//! # References
//!
//! The Double Sphere model is based on:
//! "The Double Sphere Camera Model" by Vladyslav Usenko and Nikolaus Demmel

use crate::camera::{validation, CameraModel, CameraModelError, Intrinsics, Resolution};
use argmin::{
    core::{
        observers::ObserverMode, CostFunction, Error, Executor, Gradient, Hessian, Jacobian,
        Operator, State,
    },
    solver::gaussnewton::GaussNewton,
};
use argmin_observer_slog::SlogLogger;
use log::info;
use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3xX, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use std::{fmt, fs, io::Write};
use yaml_rust::YamlLoader;

/// Cost function for Double Sphere camera model optimization.
///
/// This structure holds the 3D-2D point correspondences used during
/// camera calibration optimization. It implements the necessary traits
/// for use with the argmin optimization library.
#[derive(Clone)]
struct DoubleSphereOptimizationCost {
    /// 3D points in camera coordinate system (3×N matrix)
    points3d: Matrix3xX<f64>,
    /// Corresponding 2D points in image coordinates (2×N matrix)
    points2d: Matrix2xX<f64>,
}

impl DoubleSphereOptimizationCost {
    /// Creates a new cost function for Double Sphere camera optimization.
    ///
    /// # Arguments
    ///
    /// * `points3d` - 3D points in camera coordinate system (3×N matrix)
    /// * `points2d` - Corresponding 2D points in image coordinates (2×N matrix)
    ///
    /// # Panics
    ///
    /// Panics if the number of 3D and 2D points don't match.
    pub fn new(points3d: Matrix3xX<f64>, points2d: Matrix2xX<f64>) -> Self {
        assert_eq!(points3d.ncols(), points2d.ncols());
        DoubleSphereOptimizationCost { points3d, points2d }
    }
}

/// Implementation of the Operator trait for Gauss-Newton optimization.
impl Operator for DoubleSphereOptimizationCost {
    /// Parameter vector: [fx, fy, cx, cy, alpha, xi]
    type Param = DVector<f64>;
    /// Output residuals vector (2×N elements for N points)
    type Output = DVector<f64>;

    /// Applies the camera model to compute projection residuals.
    ///
    /// # Arguments
    ///
    /// * `p` - Parameter vector containing camera intrinsics and distortion parameters
    ///
    /// # Returns
    ///
    /// Vector of residuals between projected and observed 2D points.
    ///
    /// # Errors
    ///
    /// Returns an error if the camera model parameters are invalid or
    /// if projection fails for the given points.
    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let num_points = self.points3d.ncols();
        let mut residuals = DVector::zeros(num_points * 2);
        let model = DoubleSphereModel::new(&p)?;
        let mut counter = 0;

        for i in 0..num_points {
            let p3d = &self.points3d.column(i).into_owned();
            let p2d_gt = &self.points2d.column(i).into_owned();
            let project_result = model.project(p3d, false);

            if let Ok((p2d_projected, _)) = project_result {
                // Each point contributes 2 residuals (x and y dimensions)
                residuals[counter * 2] = p2d_projected.x - p2d_gt.x;
                residuals[counter * 2 + 1] = p2d_projected.y - p2d_gt.y;
                counter += 1;
            }
        }
        // Only return the rows with actual residuals
        residuals = residuals.rows(0, counter * 2).into_owned();
        info!("Size residuals: {}", residuals.len());
        Ok(residuals)
    }
}

/// Implementation of the Jacobian trait for Gauss-Newton optimization.
impl Jacobian for DoubleSphereOptimizationCost {
    type Param = DVector<f64>;
    type Jacobian = DMatrix<f64>;

    /// Computes the Jacobian matrix of residuals with respect to camera parameters.
    ///
    /// # Arguments
    ///
    /// * `p` - Parameter vector containing camera intrinsics and distortion parameters
    ///
    /// # Returns
    ///
    /// Jacobian matrix (2N×6) where N is the number of points and 6 is the number of parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the camera model parameters are invalid or
    /// if projection fails for the given points.
    fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, Error> {
        let num_points = self.points3d.ncols();
        let mut jacobian = DMatrix::zeros(num_points * 2, 6); // 2 residuals per point, 6 parameters
        let model = DoubleSphereModel::new(&p)?;
        let mut counter = 0;

        for i in 0..num_points {
            let p3d = &self.points3d.column(i).into_owned();

            // Get Jacobian for this point (2x6 matrix)
            let (_, jacobian_point_2x6) = model.project(p3d, true).unwrap();

            if let Some(jac) = jacobian_point_2x6 {
                // Copy the 2x6 Jacobian for this point into the overall Jacobian matrix
                jacobian.view_mut((counter * 2, 0), (2, 6)).copy_from(&jac);
                counter += 1;
            }
        }
        jacobian = jacobian.rows(0, counter * 2).into_owned();
        info!("Size residuals: {}", jacobian.nrows());
        Ok(jacobian)
    }
}

/// Implementation of the CostFunction trait for optimization.
impl CostFunction for DoubleSphereOptimizationCost {
    /// Parameter vector: [fx, fy, cx, cy, alpha, xi]
    type Param = DVector<f64>;
    /// Sum of squared errors
    type Output = f64;

    /// Computes the cost function as the sum of squared projection errors.
    ///
    /// # Arguments
    ///
    /// * `p` - Parameter vector containing camera intrinsics and distortion parameters
    ///
    /// # Returns
    ///
    /// Total cost as the sum of squared residuals.
    ///
    /// # Errors
    ///
    /// Returns an error if the camera model parameters are invalid or
    /// if projection fails for the given points.
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let mut total_error_sq = 0.0;
        let model = DoubleSphereModel::new(&p)?;

        for i in 0..self.points3d.ncols() {
            let p3d = &self.points3d.column(i).into_owned();
            let p2d_gt = &self.points2d.column(i).into_owned();

            // The project function now returns a tuple with the projection and optional Jacobian
            let (p2d_projected, _) = model.project(p3d, false).unwrap();

            total_error_sq += (p2d_projected - p2d_gt).norm();
        }

        info!("total_error_sq: {total_error_sq}");
        Ok(total_error_sq)
    }
}

/// Implementation of the Gradient trait for optimization.
impl Gradient for DoubleSphereOptimizationCost {
    type Param = DVector<f64>;
    /// Gradient of the cost function (J^T * r)
    type Gradient = DVector<f64>;

    /// Computes the gradient of the cost function.
    ///
    /// The gradient is computed as J^T * r, where J is the Jacobian matrix
    /// and r is the residual vector.
    ///
    /// # Arguments
    ///
    /// * `p` - Parameter vector containing camera intrinsics and distortion parameters
    ///
    /// # Returns
    ///
    /// Gradient vector with respect to the camera parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the camera model parameters are invalid or
    /// if projection fails for the given points.
    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        let mut grad = DVector::zeros(6);
        let model = DoubleSphereModel::new(&p)?;

        for i in 0..self.points3d.ncols() {
            let p3d = &self.points3d.column(i).into_owned();
            let p2d_gt = &self.points2d.column(i).into_owned();

            let (p2d_projected, jacobian_point_2x6) = model.project(p3d, true).unwrap();

            if let Some(jacobian) = jacobian_point_2x6 {
                let residual_2x1 = p2d_projected - p2d_gt;

                // grad += J_i^T * r_i
                grad += jacobian.transpose() * residual_2x1;
            }
        }
        info!("Gradient: {}", grad);
        Ok(grad)
    }
}

/// Implementation of the Hessian trait for optimization.
impl Hessian for DoubleSphereOptimizationCost {
    type Param = DVector<f64>;
    /// Hessian matrix approximation (J^T * J)
    type Hessian = DMatrix<f64>;

    /// Computes the Hessian matrix using the Gauss-Newton approximation.
    ///
    /// The Hessian is approximated as J^T * J, where J is the Jacobian matrix.
    /// This is a common approximation used in non-linear least squares optimization.
    ///
    /// # Arguments
    ///
    /// * `p` - Parameter vector containing camera intrinsics and distortion parameters
    ///
    /// # Returns
    ///
    /// Approximate Hessian matrix (6×6).
    ///
    /// # Errors
    ///
    /// Returns an error if the camera model parameters are invalid or
    /// if projection fails for the given points.
    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        let mut jtj = DMatrix::zeros(6, 6);
        let model = DoubleSphereModel::new(&p)?;

        for i in 0..self.points3d.ncols() {
            let p3d = &self.points3d.column(i).into_owned();
            // We only need the Jacobian for J^T J
            let (_, jacobian_point_2x6) = model.project(p3d, true).unwrap();

            // Check if jacobian_point_2x6 is Some before using it
            if let Some(jacobian) = jacobian_point_2x6 {
                jtj += jacobian.transpose() * jacobian;
            }
        }

        info!("Hessian: {}", jtj);
        Ok(jtj)
    }
}

/// Double Sphere camera model implementation.
///
/// The Double Sphere model is designed for wide-angle and fisheye cameras.
/// It uses two sphere projections to model the distortion characteristics
/// of such cameras more accurately than traditional models like polynomial distortion.
/// The model is defined by intrinsic parameters (fx, fy, cx, cy) and two
/// distortion parameters: `alpha` and `xi`.
///
/// # Parameters
///
/// * `intrinsics`: [`Intrinsics`] - Camera intrinsic parameters (fx, fy, cx, cy).
/// * `resolution`: [`Resolution`] - Image resolution (width, height).
/// * `alpha`: `f64` - The first distortion parameter, controlling the transition between
///   the two spheres. It must be in the range (0, 1].
/// * `xi`: `f64` - The second distortion parameter, representing the displacement
///   between the centers of the two spheres. It must be a finite number.
///
/// # References
///
/// * Usenko, V., Demmel, N., & Cremers, D. (2018). The Double Sphere Camera Model.
///   In 2018 International Conference on 3D Vision (3DV).
#[derive(Clone, Serialize, Deserialize)]
pub struct DoubleSphereModel {
    /// Camera intrinsic parameters
    pub intrinsics: Intrinsics,
    /// Image resolution
    pub resolution: Resolution,
    /// First distortion parameter (0 < alpha <= 1)
    pub alpha: f64,
    /// Second distortion parameter
    pub xi: f64,
}

impl DoubleSphereModel {
    /// Creates a new Double Sphere model from a parameter vector.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Parameter vector [fx, fy, cx, cy, alpha, xi]
    ///
    /// # Returns
    ///
    /// A new DoubleSphereModel instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the parameter vector doesn't have exactly 6 elements.
    fn new(parameters: &DVector<f64>) -> Result<Self, CameraModelError> {
        let model = DoubleSphereModel {
            intrinsics: Intrinsics {
                fx: parameters[0],
                fy: parameters[1],
                cx: parameters[2],
                cy: parameters[3],
            },
            resolution: Resolution {
                width: 0,
                height: 0,
            },
            alpha: parameters[4],
            xi: parameters[5],
        };

        // model.validate_params()?;
        info!("new model is: {:?}", model);
        Ok(model)
    }

    /// Checks if a 3D point can be projected using the Double Sphere model.
    ///
    /// # Arguments
    ///
    /// * `z` - Z-coordinate of the 3D point
    /// * `d1` - Distance from origin to the point
    ///
    /// # Returns
    ///
    /// `true` if the point can be projected, `false` otherwise.
    fn check_projection_condition(&self, z: f64, d1: f64) -> bool {
        let w1 = match self.alpha <= 0.5 {
            true => self.alpha / (1.0 - self.alpha),
            false => (1.0 - self.alpha) / self.alpha,
        };
        let w2 = (w1 + self.xi) / (2.0 * w1 * self.xi + self.xi * self.xi + 1.0).sqrt();
        z > -w2 * d1
    }

    /// Checks if a 2D point can be unprojected using the Double Sphere model.
    ///
    /// # Arguments
    ///
    /// * `r_squared` - Squared radial distance from the principal point
    ///
    /// # Returns
    ///
    /// `true` if the point can be unprojected, `false` otherwise.
    fn check_unprojection_condition(&self, r_squared: f64) -> bool {
        let mut condition = true;
        if self.alpha > 0.5 {
            if r_squared > 1.0 / (2.0 * self.alpha - 1.0) {
                condition = false;
            }
        }
        condition
    }
}

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
    /// Projects a 3D point to 2D image coordinates using the Double Sphere model.
    ///
    /// # Arguments
    ///
    /// * `point_3d` - 3D point in camera coordinate system
    /// * `compute_jacobian` - Whether to compute the Jacobian matrix
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - The projected 2D point
    /// - Optional Jacobian matrix (2×6) if `compute_jacobian` is true
    ///
    /// # Errors
    ///
    /// Returns `CameraModelError::PointIsOutSideImage` if the point cannot be projected.
    fn project(
        &self,
        point_3d: &Vector3<f64>,
        compute_jacobian: bool,
    ) -> Result<(Vector2<f64>, Option<DMatrix<f64>>), CameraModelError> {
        const PRECISION: f64 = 1e-3;

        let x = point_3d.x;
        let y = point_3d.y;
        let z = point_3d.z;

        let r_squared = (x * x) + (y * y);
        let d1 = (r_squared + (z * z)).sqrt();
        let gamma = self.xi * d1 + z;
        let d2 = (r_squared + gamma * gamma).sqrt();

        let denom = self.alpha * d2 + (1.0 - self.alpha) * gamma;

        // Check if the projection is valid
        if denom < PRECISION || !self.check_projection_condition(z, d1) {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let mx = x / denom;
        let my = y / denom;

        // Project the point
        let projected_x = self.intrinsics.fx * (mx) + self.intrinsics.cx;
        let projected_y = self.intrinsics.fy * (my) + self.intrinsics.cy;

        let jacobian = if compute_jacobian {
            let mut d_proj_d_param = DMatrix::<f64>::zeros(2, 6);

            let u_cx = projected_x - self.intrinsics.cx;
            let v_cy = projected_y - self.intrinsics.cy;
            let m_alpha = 1.0 - self.alpha;

            // Set Jacobian entries for intrinsics
            d_proj_d_param[(0, 0)] = x; // ∂residual_x / ∂fx
            d_proj_d_param[(0, 1)] = 0.0; // ∂residual_y / ∂fx
            d_proj_d_param[(1, 0)] = 0.0; // ∂residual_x / ∂fy
            d_proj_d_param[(1, 1)] = y; // ∂residual_y / ∂fy

            d_proj_d_param[(0, 2)] = denom; // ∂residual_x / ∂cx
            d_proj_d_param[(0, 3)] = 0.0; // ∂residual_y / ∂cx
            d_proj_d_param[(1, 2)] = 0.0; // ∂residual_x / ∂cy
            d_proj_d_param[(1, 3)] = denom; // ∂residual_y / ∂cy

            d_proj_d_param[(0, 4)] = (gamma - d2) * u_cx; // ∂residual_x / ∂alpha
            d_proj_d_param[(1, 4)] = (gamma - d2) * v_cy; // ∂residual_y / ∂alpha

            let coeff = (self.alpha * d1 * gamma) / d2 + (m_alpha * d1);
            d_proj_d_param[(0, 5)] = -u_cx * coeff; // ∂residual_x / ∂xi
            d_proj_d_param[(1, 5)] = -v_cy * coeff; // ∂residual_y / ∂xi

            Some(d_proj_d_param)
        } else {
            None
        };

        Ok((Vector2::new(projected_x, projected_y), jacobian))
    }

    /// Unprojects a 2D image point to a 3D ray direction using the Double Sphere model.
    ///
    /// # Arguments
    ///
    /// * `point_2d` - 2D point in image coordinates
    ///
    /// # Returns
    ///
    /// A normalized 3D vector representing the ray direction.
    ///
    /// # Errors
    ///
    /// Returns `CameraModelError::PointIsOutSideImage` if the point cannot be unprojected.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        const PRECISION: f64 = 1e-3;

        let fx = self.intrinsics.fx;
        let fy = self.intrinsics.fy;
        let cx = self.intrinsics.cx;
        let cy = self.intrinsics.cy;
        let alpha = self.alpha;
        let xi = self.xi;

        let u = point_2d.x;
        let v = point_2d.y;
        let gamma = 1.0 - alpha;
        let mx = (u - cx) / fx;
        let my = (v - cy) / fy;
        let r_squared = (mx * mx) + (my * my);

        // Check if we can unproject this point
        if alpha != 0.0 && !self.check_unprojection_condition(r_squared) {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let mz = (1.0 - alpha * alpha * r_squared)
            / (alpha * (1.0 - (2.0 * alpha - 1.0) * r_squared).sqrt() + gamma);
        let mz_squared = mz * mz;

        let num = mz * xi + (mz_squared + (1.0 - xi * xi) * r_squared).sqrt();
        let denom = mz_squared + r_squared;

        // Check if denominator is too small
        if denom < PRECISION {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let coeff = num / denom;

        // Calculate the unprojected 3D point
        let point3d = Vector3::new(coeff * mx, coeff * my, coeff * mz - xi);

        Ok(point3d.normalize())
    }

    /// Loads a Double Sphere camera model from a YAML file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the YAML file
    ///
    /// # Returns
    ///
    /// A new DoubleSphereModel instance loaded from the file.
    ///
    /// # Errors
    ///
    /// Returns various `CameraModelError` variants if the file cannot be read,
    /// parsed, or contains invalid parameters.
    fn load_from_yaml(path: &str) -> Result<Self, CameraModelError> {
        let contents = fs::read_to_string(path)?;
        let docs = YamlLoader::load_from_str(&contents)?;

        if docs.is_empty() {
            return Err(CameraModelError::InvalidParams(
                "Empty YAML document".to_string(),
            ));
        }

        let doc = &docs[0];

        let intrinsics = doc["cam0"]["intrinsics"]
            .as_vec()
            .ok_or_else(|| CameraModelError::InvalidParams("Invalid intrinsics".to_string()))?;
        let resolution = doc["cam0"]["resolution"]
            .as_vec()
            .ok_or_else(|| CameraModelError::InvalidParams("Invalid resolution".to_string()))?;

        let alpha = intrinsics[4]
            .as_f64()
            .ok_or_else(|| CameraModelError::InvalidParams("Invalid alpha".to_string()))?;

        let xi = intrinsics[5]
            .as_f64()
            .ok_or_else(|| CameraModelError::InvalidParams("Invalid xi".to_string()))?;

        let intrinsics = Intrinsics {
            fx: intrinsics[0]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid fx".to_string()))?,
            fy: intrinsics[1]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid fy".to_string()))?,
            cx: intrinsics[2]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid cx".to_string()))?,
            cy: intrinsics[3]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid cy".to_string()))?,
        };

        let resolution = Resolution {
            width: resolution[0]
                .as_i64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid width".to_string()))?
                as u32,
            height: resolution[1]
                .as_i64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid height".to_string()))?
                as u32,
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

    /// Saves the Double Sphere camera model to a YAML file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where to save the YAML file
    ///
    /// # Errors
    ///
    /// Returns `CameraModelError::IOError` if the file cannot be written,
    /// or `CameraModelError::YamlError` if serialization fails.
    fn save_to_yaml(&self, path: &str) -> Result<(), CameraModelError> {
        // Create the YAML structure using serde_yaml
        let yaml = serde_yaml::to_value(&serde_yaml::Mapping::from_iter([(
            serde_yaml::Value::String("cam0".to_string()),
            serde_yaml::to_value(&serde_yaml::Mapping::from_iter([
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
                        self.alpha,
                        self.xi,
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

    /// Validates the camera model parameters.
    ///
    /// # Returns
    ///
    /// `Ok(())` if all parameters are valid.
    ///
    /// # Errors
    ///
    /// Returns `CameraModelError::InvalidParams` if any parameter is invalid:
    /// - `alpha` must be in the range (0, 1]
    /// - `xi` must be finite
    /// - Intrinsic parameters must be valid (checked by `validation::validate_intrinsics`)
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

    /// Returns the image resolution.
    ///
    /// # Returns
    ///
    /// A copy of the Resolution struct containing width and height.
    fn get_resolution(&self) -> Resolution {
        self.resolution.clone()
    }

    /// Returns the camera intrinsic parameters.
    ///
    /// # Returns
    ///
    /// A copy of the Intrinsics struct containing fx, fy, cx, cy.
    fn get_intrinsics(&self) -> Intrinsics {
        self.intrinsics.clone()
    }

    /// Returns the distortion parameters.
    ///
    /// # Returns
    ///
    /// A vector containing [xi, alpha] distortion parameters.
    fn get_distortion(&self) -> Vec<f64> {
        vec![self.xi, self.alpha]
    }

    /// Performs linear estimation of the Double Sphere model parameters.
    ///
    /// This method provides an initial estimate of the `alpha` parameter
    /// while setting `xi` to 0.0. It's typically used as initialization
    /// for non-linear optimization.
    ///
    /// # Arguments
    ///
    /// * `intrinsics` - Known camera intrinsic parameters
    /// * `resolution` - Image resolution
    /// * `points_2d` - 2D points in image coordinates (2×N matrix)
    /// * `points_3d` - Corresponding 3D points in camera coordinates (3×N matrix)
    ///
    /// # Returns
    ///
    /// A new DoubleSphereModel with estimated parameters.
    ///
    /// # Errors
    ///
    /// Returns `CameraModelError::InvalidParams` if the number of 2D and 3D points don't match,
    /// or `CameraModelError::NumericalError` if the linear system cannot be solved.
    fn linear_estimation(
        intrinsics: &Intrinsics,
        resolution: &Resolution,
        points_2d: &Matrix2xX<f64>,
        points_3d: &Matrix3xX<f64>,
    ) -> Result<Self, CameraModelError> {
        // Check if the number of 2D and 3D points match
        if points_2d.ncols() != points_3d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }

        // Initialize with xi = 0.0
        let xi = 0.0;

        // Set up the linear system to solve for alpha
        let num_points = points_2d.ncols();
        let mut a = nalgebra::DMatrix::zeros(num_points * 2, 1);
        let mut b = nalgebra::DVector::zeros(num_points * 2);

        for i in 0..num_points {
            let x = points_3d[(0, i)];
            let y = points_3d[(1, i)];
            let z = points_3d[(2, i)];
            let u = points_2d[(0, i)];
            let v = points_2d[(1, i)];

            let d = (x * x + y * y + z * z).sqrt();
            let u_cx = u - intrinsics.cx;
            let v_cy = v - intrinsics.cy;

            a[(i * 2, 0)] = u_cx * (d - z);
            a[(i * 2 + 1, 0)] = v_cy * (d - z);

            b[i * 2] = (intrinsics.fx * x) - (u_cx * z);
            b[i * 2 + 1] = (intrinsics.fy * y) - (v_cy * z);
        }

        // Solve the linear system using SVD
        let svd = a.svd(true, true);
        let alpha = match svd.solve(&b, 1e-10) {
            Ok(sol) => sol[0], // Handle the successful case
            Err(err_msg) => {
                return Err(CameraModelError::NumericalError(err_msg.to_string()));
            }
        };

        // Clamp alpha to valid range (0, 1]
        let alpha = alpha;

        let model = DoubleSphereModel {
            intrinsics: intrinsics.clone(),
            resolution: resolution.clone(),
            alpha,
            xi,
        };

        // Validate parameters
        model.validate_params()?;

        Ok(model)
    }

    /// Optimize the camera model parameters using the provided 2D-3D point correspondences
    fn optimize(
        &mut self,
        points_3d: &Matrix3xX<f64>,
        points_2d: &Matrix2xX<f64>,
        verbose: bool,
    ) -> Result<(), CameraModelError> {
        if points_3d.ncols() != points_2d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }

        if points_3d.ncols() == 0 {
            return Err(CameraModelError::InvalidParams(
                "Points arrays cannot be empty".to_string(),
            ));
        }

        let cost_function = DoubleSphereOptimizationCost::new(points_3d.clone(), points_2d.clone());

        // Initial parameters
        let param = vec![
            self.intrinsics.fx,
            self.intrinsics.fy,
            self.intrinsics.cx,
            self.intrinsics.cy,
            self.alpha,
            self.xi,
        ];

        let init_param: DVector<f64> = DVector::from_vec(param);

        // Configure the Gauss-Newton solver
        let solver = GaussNewton::new();

        // Setup executor with the solver and cost function
        let executor_builder = Executor::new(cost_function, solver)
            .configure(|state| state.param(init_param).max_iters(100))
            .add_observer(SlogLogger::term(), ObserverMode::NewBest);

        if verbose {
            info!("Starting optimization with Gauss-Newton...");
        }

        let res = executor_builder
            .run()
            .map_err(|e| CameraModelError::NumericalError(e.to_string()))?;

        if verbose {
            info!("Optimization finished: \n{}", res);
            info!("Termination status: {:?}", res.state().termination_status);
        }

        let best_params_dvec = res.state().get_best_param().unwrap().clone();

        // Update model parameters from the final parameters
        self.intrinsics.fx = best_params_dvec[0];
        self.intrinsics.fy = best_params_dvec[1];
        self.intrinsics.cx = best_params_dvec[2];
        self.intrinsics.cy = best_params_dvec[3];
        self.alpha = best_params_dvec[4];
        self.xi = best_params_dvec[5];

        // Validate the optimized parameters
        self.validate_params()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq; // For floating point comparisons

    // Helper to get a default model, similar to the one in samples/double_sphere.yaml
    fn get_sample_model() -> DoubleSphereModel {
        DoubleSphereModel {
            intrinsics: Intrinsics {
                fx: 348.112754378549,
                fy: 347.1109973814674,
                cx: 365.8121721753254,
                cy: 249.3555778487899,
            },
            resolution: Resolution { width: 752, height: 480 },
            alpha: 0.5657413673629862,
            xi: -0.24425190195168348,
        }
    }

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

    #[test]
    fn test_double_sphere_save_to_yaml() {
        use std::fs;

        // Create output directory if it doesn't exist
        fs::create_dir_all("output").unwrap_or_else(|_| {
            info!("Output directory already exists or couldn't be created");
        });

        // Define input and output paths
        let input_path = "samples/double_sphere.yaml";
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

        // Clean up the saved file (optional)
        // fs::remove_file(output_path).unwrap();
    }

    #[test]
    fn test_double_sphere_project_unproject() {
        // Load the camera model from YAML
        let path = "samples/double_sphere.yaml";
        let model = DoubleSphereModel::load_from_yaml(path).unwrap();

        // Create a 3D point in camera coordinates (pointing somewhat forward and to the side)
        let point_3d = Vector3::new(0.5, -0.3, 2.0);
        let norm_3d = point_3d.normalize();

        // Project the 3D point to pixel coordinates
        let (point_2d, _) = model.project(&point_3d, false).unwrap();

        // Check if the pixel coordinates are within the image bounds
        assert!(point_2d.x >= 0.0 && point_2d.x < model.resolution.width as f64);
        assert!(point_2d.y >= 0.0 && point_2d.y < model.resolution.height as f64);

        // Unproject the pixel point back to a 3D ray direction
        let point_3d_unprojected = model.unproject(&point_2d).unwrap();

        // Check if the unprojected point is close to the original point
        assert!((norm_3d.x - point_3d_unprojected.x).abs() < 1e-6);
        assert!((norm_3d.y - point_3d_unprojected.y).abs() < 1e-6);
        assert!((norm_3d.z - point_3d_unprojected.z).abs() < 1e-6);
    }

    #[test]
    fn test_project_point_at_center() {
        let model = get_sample_model();
        // For Double Sphere, projecting from the origin (0,0,0) might not be well-defined
        // depending on how d1 and denom are handled. Let's use a point very close to origin on Z axis.
        let point_3d_on_z = Vector3::new(0.0, 0.0, 1e-9);
        let result_origin = model.project(&point_3d_on_z, false);
        // Depending on precision, this might project to cx,cy or be an error.
        // If it projects, it should be very close to cx, cy.
        if let Ok((p, _)) = result_origin {
            assert_relative_eq!(p.x, model.intrinsics.cx, epsilon = 1e-3);
            assert_relative_eq!(p.y, model.intrinsics.cy, epsilon = 1e-3);
        } else {
            // Or it could be an error if denom becomes too small or condition fails
            assert!(matches!(result_origin, Err(CameraModelError::PointIsOutSideImage)));
        }

        // A point exactly at (0,0,0) for some models might lead to d1 = 0, denom = 0.
        // The current implementation's check `denom < PRECISION` should catch this.
        let result_exact_origin = model.project(&Vector3::new(0.0, 0.0, 0.0), false);
        assert!(matches!(result_exact_origin, Err(CameraModelError::PointIsOutSideImage)), "Projecting (0,0,0) should ideally be an error or handled gracefully");
    }

    #[test]
    fn test_project_point_behind_camera() {
        let model = get_sample_model();
        let point_3d = Vector3::new(0.1, 0.2, -1.0); // Point behind camera
        let result = model.project(&point_3d, false);
        assert!(matches!(result, Err(CameraModelError::PointIsOutSideImage)));
    }

    #[test]
    fn test_validate_params_valid() {
        let model = get_sample_model();
        assert!(model.validate_params().is_ok());
    }

    #[test]
    fn test_validate_params_invalid_alpha() {
        let mut model = get_sample_model();
        model.alpha = 0.0; // Invalid: alpha must be > 0
        assert!(matches!(model.validate_params(), Err(CameraModelError::InvalidParams(_))));

        model.alpha = 1.1; // Invalid: alpha must be <= 1
        assert!(matches!(model.validate_params(), Err(CameraModelError::InvalidParams(_))));
    }

    #[test]
    fn test_validate_params_invalid_xi() {
        let mut model = get_sample_model();
        model.xi = f64::NAN;
        assert!(matches!(model.validate_params(), Err(CameraModelError::InvalidParams(_))));

        model.xi = f64::INFINITY;
        assert!(matches!(model.validate_params(), Err(CameraModelError::InvalidParams(_))));
    }

    #[test]
    fn test_validate_params_invalid_intrinsics() {
        let mut model = get_sample_model();
        model.intrinsics.fx = 0.0; // Invalid: fx must be > 0 (checked by validation::validate_intrinsics)
        assert!(matches!(model.validate_params(), Err(CameraModelError::FocalLengthMustBePositive)));
    }

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
        assert_relative_eq!(distortion[0], model.xi);    // xi is first
        assert_relative_eq!(distortion[1], model.alpha); // alpha is second
    }

    #[test]
    fn test_double_sphere_optimize() {
        // Load a reference model from YAML file
        let input_path = "samples/double_sphere.yaml";
        let reference_model = DoubleSphereModel::load_from_yaml(input_path).unwrap();

        // Use geometry::sample_points to generate a set of 2D-3D point correspondences
        let n = 500;
        let (points_2d, points_3d) =
            crate::geometry::sample_points(Some(&reference_model), n).unwrap();

        // Create a model with added noise to the parameters
        let mut noisy_model = DoubleSphereModel {
            intrinsics: Intrinsics {
                // Add some noise to the intrinsic parameters (±5-10%)
                fx: reference_model.intrinsics.fx * 0.95,
                fy: reference_model.intrinsics.fy * 1.02,
                cx: reference_model.intrinsics.cx + 0.5,
                cy: reference_model.intrinsics.cy - 0.8,
            },
            resolution: reference_model.resolution.clone(),
            // Add noise to distortion parameters
            xi: -0.1,
            alpha: (reference_model.alpha * 0.95).max(0.1).min(0.99),
        };

        info!("Reference model: {:?}", reference_model);
        info!("Noisy model: {:?}", noisy_model);

        // Optimize the model with noise
        noisy_model.optimize(&points_3d, &points_2d, false).unwrap();

        info!("Optimized model: {:?}", noisy_model);

        // Check that parameters have been optimized close to reference values
        assert!(
            (noisy_model.intrinsics.fx - reference_model.intrinsics.fx).abs() < 1.0,
            "fx parameter didn't converge to expected value"
        );
        assert!(
            (noisy_model.intrinsics.fy - reference_model.intrinsics.fy).abs() < 1.0,
            "fy parameter didn't converge to expected value"
        );
        assert!(
            (noisy_model.intrinsics.cx - reference_model.intrinsics.cx).abs() < 1.0,
            "cx parameter didn't converge to expected value"
        );
        assert!(
            (noisy_model.intrinsics.cy - reference_model.intrinsics.cy).abs() < 1.0,
            "cy parameter didn't converge to expected value"
        );
        assert!(
            (noisy_model.alpha - reference_model.alpha).abs() < 0.05,
            "alpha parameter didn't converge to expected value"
        );
        assert!(
            (noisy_model.xi - reference_model.xi).abs() < 0.05,
            "xi parameter didn't converge to expected value"
        );
        // Verify that alpha is within bounds (0, 1]
        assert!(
            noisy_model.alpha > 0.0 && noisy_model.alpha <= 1.0,
            "Alpha parameter out of valid range (0, 1]"
        );

        // Verify that focal lengths are positive
        assert!(noisy_model.intrinsics.fx > 0.0, "fx must be positive");
        assert!(noisy_model.intrinsics.fy > 0.0, "fy must be positive");
    }

    #[test]
    fn test_double_sphere_optimization_cost_basic() {
        // Load camera model from YAML
        let model = DoubleSphereModel::load_from_yaml("samples/double_sphere.yaml").unwrap();

        // 5 sample 2D points as a flat vector
        let points_2d_vec = vec![0.0, 0.0, 2.0, 2.0, -2.0, 2.0, 2.0, -2.0, -2.0, -2.0];
        let mat_2d = Matrix2xX::from_vec(points_2d_vec);
        let num_points = mat_2d.ncols();
        info!("Size 2d points: {}", num_points);

        // Create a Matrix3xX for 3D points, fill with zeros
        let mut mat_3d = Matrix3xX::zeros(num_points);

        // Fill the unprojected 3D points into mat_3d, only if unprojection succeeds
        let mut valid_count = 0;
        for i in 0..num_points {
            let p2d = mat_2d.column(i).into_owned();
            if let Ok(p3d) = model.unproject(&p2d) {
                mat_3d.set_column(valid_count, &p3d);
                // Overwrite the corresponding 2D point in mat_2d if needed (optional, but keeps correspondence)
                // mat_2d.set_column(valid_count, &p2d);
                valid_count += 1;
            }
        }

        assert!(
            valid_count > 0,
            "At least one 2D point should be unprojectable"
        );

        // Construct cost struct
        let cost = DoubleSphereOptimizationCost::new(mat_3d.clone(), mat_2d.clone());

        // Prepare parameter vector from model
        let p = nalgebra::DVector::from_vec(vec![
            model.intrinsics.fx,
            model.intrinsics.fy,
            model.intrinsics.cx,
            model.intrinsics.cy,
            model.alpha,
            model.xi,
        ]);

        // Test operator (apply)
        let residuals = cost.apply(&p).unwrap();
        assert_eq!(residuals.len(), 2 * mat_3d.ncols());
        assert!(
            residuals.iter().all(|v| v.abs() < 1e-6),
            "Residuals should be near zero for perfect model"
        );

        // Test cost
        let c = cost.cost(&p).unwrap();
        assert!(c >= 0.0, "Cost should be non-negative");
        assert!(c < 1e-5, "Cost should be near zero for perfect model");

        // Test jacobian
        let jac = cost.jacobian(&p).unwrap();
        assert_eq!(jac.nrows(), 2 * mat_3d.ncols());
        assert_eq!(jac.ncols(), 6);

        // Test gradient
        let grad = cost.gradient(&p).unwrap();
        assert_eq!(grad.len(), 6);

        // Test hessian
        let hess = cost.hessian(&p).unwrap();
        assert_eq!(hess.nrows(), 6);
        assert_eq!(hess.ncols(), 6);
    }
}
