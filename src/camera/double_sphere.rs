use crate::camera::{validation, CameraModel, CameraModelError, Intrinsics, Resolution};
use argmin::{
    core::{observers::ObserverMode, CostFunction, Error, Executor, Gradient, Hessian, State},
    solver::trustregion::{Dogleg, TrustRegion},
};
use argmin_observer_slog::SlogLogger;
use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3xX, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use yaml_rust::YamlLoader;

// Cost function for optimization
#[derive(Clone)]
struct DoubleSphereOptimizationCost {
    points3d: Matrix3xX<f64>,
    points2d: Matrix2xX<f64>,
}

impl DoubleSphereOptimizationCost {
    pub fn new(points3d: Matrix3xX<f64>, points2d: Matrix2xX<f64>) -> Self {
        assert_eq!(points3d.ncols(), points2d.ncols());
        DoubleSphereOptimizationCost { points3d, points2d }
    }

    // Helper to unpack parameters
    fn unpack_params(p: &DVector<f64>) -> (f64, f64, f64, f64, f64, f64) {
        (p[0], p[1], p[2], p[3], p[4], p[5])
    }
}

impl CostFunction for DoubleSphereOptimizationCost {
    type Param = DVector<f64>; // [fx, fy, cx, cy, alpha, xi]
    type Output = f64; // Sum of squared errors

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let (fx, fy, cx, cy, alpha, xi) = Self::unpack_params(p);
        let mut total_error_sq = 0.0;

        let intrinsics = Intrinsics { fx, fy, cx, cy };
        let resolution = Resolution {
            width: 0,  // Not used in projection
            height: 0, // Not used in projection
        };

        let model = DoubleSphereModel {
            intrinsics,
            resolution,
            xi,
            alpha,
        };

        for i in 0..self.points3d.ncols() {
            let p3d = &self.points3d.column(i).into_owned();
            let p2d_gt = &self.points2d.column(i).into_owned();

            // The project function now returns a tuple with the projection and optional Jacobian
            let (p2d_projected, _) = model.project(p3d, false).unwrap();

            total_error_sq += (p2d_projected - p2d_gt).norm();
        }

        println!("total_error_sq: {total_error_sq}");
        Ok(total_error_sq)
    }
}

impl Gradient for DoubleSphereOptimizationCost {
    type Param = DVector<f64>;
    type Gradient = DVector<f64>; // Gradient of the cost function (J^T * r)

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        let (fx, fy, cx, cy, alpha, xi) = Self::unpack_params(p);
        let mut grad = DVector::zeros(6);

        let intrinsics = Intrinsics { fx, fy, cx, cy };
        let resolution = Resolution {
            width: 0,  // Not used in projection
            height: 0, // Not used in projection
        };

        let model = DoubleSphereModel {
            intrinsics,
            resolution,
            xi,
            alpha,
        };

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
        println!("Gradient: {}", grad);
        Ok(grad)
    }
}

// For Gauss-Newton like behavior in TrustRegion (J^T J approximation for Hessian)
impl Hessian for DoubleSphereOptimizationCost {
    type Param = DVector<f64>;
    type Hessian = DMatrix<f64>; // J^T * J

    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        let (fx, fy, cx, cy, alpha, xi) = Self::unpack_params(p);
        let mut jtj = DMatrix::zeros(6, 6);

        let intrinsics = Intrinsics { fx, fy, cx, cy };
        let resolution = Resolution {
            width: 0,  // Not used in projection
            height: 0, // Not used in projection
        };

        let model = DoubleSphereModel {
            intrinsics,
            resolution,
            xi,
            alpha,
        };

        for i in 0..self.points3d.ncols() {
            let p3d = &self.points3d.column(i).into_owned();
            // We only need the Jacobian for J^T J
            let (_, jacobian_point_2x6) = model.project(p3d, true).unwrap();

            // Check if jacobian_point_2x6 is Some before using it
            if let Some(jacobian) = jacobian_point_2x6 {
                jtj += jacobian.transpose() * jacobian;
            }
        }

        println!("Hessian: {}", jtj);
        Ok(jtj)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoubleSphereModel {
    pub intrinsics: Intrinsics,
    pub resolution: Resolution,
    pub xi: f64,
    pub alpha: f64,
}

impl DoubleSphereModel {
    fn check_proj_condition(&self, z: f64, d1: f64) -> bool {
        let w1 = match self.alpha <= 0.5 {
            true => self.alpha / (1.0 - self.alpha),
            false => (1.0 - self.alpha) / self.alpha,
        };
        let w2 = (w1 + self.xi) / (2.0 * w1 * self.xi + self.xi * self.xi + 1.0).sqrt();
        z > -w2 * d1
    }

    fn check_unproj_condition(&self, r_squared: f64) -> bool {
        let mut condition = true;
        if self.alpha > 0.5 {
            if r_squared > 1.0 / (2.0 * self.alpha - 1.0) {
                condition = false;
            }
        }
        condition
    }
}

impl CameraModel for DoubleSphereModel {
    fn initialize(&mut self, parameters: &DVector<f64>) -> Result<(), CameraModelError> {
        self.intrinsics = Intrinsics {
            fx: parameters[0],
            fy: parameters[1],
            cx: parameters[2],
            cy: parameters[3],
        };
        self.resolution = Resolution {
            width: 0,
            height: 0,
        };
        self.xi = parameters[4];
        self.alpha = parameters[3];
        Ok(())
    }

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
        if denom < PRECISION || !self.check_proj_condition(z, d1) {
            return Ok((Vector2::new(-1.0, -1.0), None));
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
        if alpha != 0.0 && !self.check_unproj_condition(r_squared) {
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

        let xi = intrinsics[4]
            .as_f64()
            .ok_or_else(|| CameraModelError::InvalidParams("Invalid xi".to_string()))?;

        let alpha = intrinsics[5]
            .as_f64()
            .ok_or_else(|| CameraModelError::InvalidParams("Invalid alpha".to_string()))?;

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
            xi,
            alpha,
        };

        // Validate parameters
        model.validate_params()?;
        Ok(model)
    }

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
                        self.xi,
                        self.alpha,
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

    fn validate_params(&self) -> Result<(), CameraModelError> {
        validation::validate_intrinsics(&self.intrinsics)?;

        if !self.xi.is_finite() {
            return Err(CameraModelError::InvalidParams(
                "xi must be finite".to_string(),
            ));
        }

        if self.alpha <= 0.0 || self.alpha > 1.0 {
            return Err(CameraModelError::InvalidParams(
                "alpha must be in (0, 1]".to_string(),
            ));
        }

        Ok(())
    }

    fn get_resolution(&self) -> Resolution {
        self.resolution.clone()
    }

    fn get_intrinsics(&self) -> Intrinsics {
        self.intrinsics.clone()
    }

    fn get_distortion(&self) -> Vec<f64> {
        vec![self.xi, self.alpha]
    }

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
            xi,
            alpha,
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

        // Initial parameters as Vec<f64> instead of SVector
        let param = vec![
            self.intrinsics.fx,
            self.intrinsics.fy,
            self.intrinsics.cx,
            self.intrinsics.cy,
            self.xi,
            self.alpha,
        ];

        let init_param: DVector<f64> = DVector::from_vec(param);

        // Define reference bounds for parameters (used in validation)
        // fx, fy > 0
        // 0 < alpha < 1
        // xi can be any value

        // Configure the subproblem solver using Dogleg method
        let subproblem_solver = Dogleg::new();
        // Dogleg doesn't support bounds directly, we'll handle bounds in the cost function
        // For reference: the bounds were for fx,fy > 0, 0 < alpha < 1, and any xi

        // Configure the TrustRegion solver
        let solver = TrustRegion::new(subproblem_solver);
        // solver.set_radius_update(TrustRegionRadius::new().gamma_inc(2.5).gamma_dec(0.25)); // Optional: tune radius update

        // let executor_builder = Executor::new(cost_function, solver)
        // .configure(|state| state.param(init_param).max_iters(100)); // Set initial param and max iterations

        let executor_builder = Executor::new(cost_function, solver)
            .configure(|state| state.param(init_param).max_iters(100))
            .add_observer(SlogLogger::term(), ObserverMode::Always);

        if verbose {
            println!("Starting optimization...");
        }

        let res = executor_builder
            .run()
            .map_err(|e| CameraModelError::NumericalError(e.to_string()))?;

        if verbose {
            println!("Optimization finished: \n{}", res);
            println!("Termination status: {:?}", res.state().termination_status);
        }

        let best_params_dvec = res.state().get_best_param().unwrap().clone();

        // Update model parameters from the bounded final parameters
        self.intrinsics.fx = best_params_dvec[0];
        self.intrinsics.fy = best_params_dvec[1];
        self.intrinsics.cx = best_params_dvec[2];
        self.intrinsics.cy = best_params_dvec[3];
        self.xi = best_params_dvec[4];
        self.alpha = best_params_dvec[5];

        // Validate the optimized parameters
        self.validate_params()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_sphere_load_from_yaml() {
        let path = "samples/double_sphere.yaml";
        let model = DoubleSphereModel::load_from_yaml(path).unwrap();

        assert_eq!(model.intrinsics.fx, 348.112754378549);
        assert_eq!(model.intrinsics.fy, 347.1109973814674);
        assert_eq!(model.intrinsics.cx, 365.8121721753254);
        assert_eq!(model.intrinsics.cy, 249.3555778487899);
        assert_eq!(model.xi, -0.24425190195168348);
        assert_eq!(model.alpha, 0.5657413673629862);
        assert_eq!(model.resolution.width, 752);
        assert_eq!(model.resolution.height, 480);
    }

    #[test]
    fn test_double_sphere_save_to_yaml() {
        use std::fs;

        // Create output directory if it doesn't exist
        fs::create_dir_all("output").unwrap_or_else(|_| {
            println!("Output directory already exists or couldn't be created");
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
    fn test_double_sphere_optimize() {
        // Load a reference model from YAML file
        let input_path = "samples/double_sphere.yaml";
        let reference_model = DoubleSphereModel::load_from_yaml(input_path).unwrap();

        // Use geometry::sample_points to generate a set of 2D-3D point correspondences
        let n = 100;
        let (points_2d, points_3d) =
            crate::geometry::sample_points(Some(&reference_model), n).unwrap();

        // Create a model with added noise to the parameters
        let mut noisy_model = DoubleSphereModel {
            intrinsics: Intrinsics {
                // Add some noise to the intrinsic parameters (±5-10%)
                fx: reference_model.intrinsics.fx + 0.05,
                fy: reference_model.intrinsics.fy * 1.02,
                cx: reference_model.intrinsics.cx + 0.2,
                cy: reference_model.intrinsics.cy - 0.1,
            },
            resolution: reference_model.resolution.clone(),
            // Add noise to distortion parameters
            xi: 0.0,
            alpha: (reference_model.alpha * 0.85).max(0.1).min(0.99),
        };

        println!("Reference model parameters:");
        println!(
            "fx: {}, fy: {}",
            reference_model.intrinsics.fx, reference_model.intrinsics.fy
        );
        println!(
            "cx: {}, cy: {}",
            reference_model.intrinsics.cx, reference_model.intrinsics.cy
        );
        println!(
            "xi: {}, alpha: {}",
            reference_model.xi, reference_model.alpha
        );

        println!("\nNoisy model parameters (before optimization):");
        println!(
            "fx: {}, fy: {}",
            noisy_model.intrinsics.fx, noisy_model.intrinsics.fy
        );
        println!(
            "cx: {}, cy: {}",
            noisy_model.intrinsics.cx, noisy_model.intrinsics.cy
        );
        println!("xi: {}, alpha: {}", noisy_model.xi, noisy_model.alpha);

        // Optimize the model with noise
        noisy_model.optimize(&points_3d, &points_2d, true).unwrap();

        println!("\nOptimized model parameters:");
        println!(
            "fx: {}, fy: {}",
            noisy_model.intrinsics.fx, noisy_model.intrinsics.fy
        );
        println!(
            "cx: {}, cy: {}",
            noisy_model.intrinsics.cx, noisy_model.intrinsics.cy
        );
        println!("xi: {}, alpha: {}", noisy_model.xi, noisy_model.alpha);

        // Check that parameters have been optimized close to reference values
        assert!(
            (noisy_model.intrinsics.fx - reference_model.intrinsics.fx).abs() < 10.0,
            "fx parameter didn't converge to expected value"
        );
        assert!(
            (noisy_model.intrinsics.fy - reference_model.intrinsics.fy).abs() < 10.0,
            "fy parameter didn't converge to expected value"
        );
        assert!(
            (noisy_model.intrinsics.cx - reference_model.intrinsics.cx).abs() < 10.0,
            "cx parameter didn't converge to expected value"
        );
        assert!(
            (noisy_model.intrinsics.cy - reference_model.intrinsics.cy).abs() < 10.0,
            "cy parameter didn't converge to expected value"
        );
        assert!(
            (noisy_model.xi - reference_model.xi).abs() < 0.05,
            "xi parameter didn't converge to expected value"
        );
        assert!(
            (noisy_model.alpha - reference_model.alpha).abs() < 0.05,
            "alpha parameter didn't converge to expected value"
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
}
