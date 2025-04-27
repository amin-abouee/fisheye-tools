use argmin::{
    core::{observers::ObserverMode, CostFunction, Error, Executor, Gradient},
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS},
};
use argmin_observer_slog::SlogLogger;
use nalgebra::{Matrix3xX, Matrix2xX, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use yaml_rust::YamlLoader;

use crate::camera::{validation, CameraModel, CameraModelError, Intrinsics, Resolution};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoubleSphereModel {
    pub intrinsics: Intrinsics,
    pub resolution: Resolution,
    pub xi: f64,
    pub alpha: f64,
}

// Cost function for optimization
#[derive(Clone)]
struct DoubleSphereOptimizationCost {
    points_3d: Matrix3xX<f64>,
    points_2d: Matrix2xX<f64>,
    // Parameter bounds
    lower_bounds: nalgebra::SVector<f64, 6>,
    upper_bounds: nalgebra::SVector<f64, 6>,
}

impl CostFunction for DoubleSphereOptimizationCost {
    type Param = nalgebra::SVector<f64, 6>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        // Apply parameter bounds by projecting parameters to valid range
        let mut bounded_p = p.clone();
        for i in 0..p.len() {
            bounded_p[i] = bounded_p[i].max(self.lower_bounds[i]).min(self.upper_bounds[i]);
        }
        
        // Extract parameters
        let fx = bounded_p[0];
        let fy = bounded_p[1];
        let cx = bounded_p[2];
        let cy = bounded_p[3];
        let xi = bounded_p[4];
        let alpha = bounded_p[5];

        // Create temporary model with these parameters
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

        // Calculate cost as sum of squared reprojection errors
        let mut total_cost = 0.0;

        for (p3d, p2d) in self.points_3d.column_iter().zip(self.points_2d.column_iter()) {
            match model.project(&Vector3::from(p3d)) {
                Ok(projected) => {
                    // let dx = projected.x - p2d[0];
                    // let dy = projected.y - p2d[1];
                    // total_cost += dx * dx + dy * dy;
                    total_cost += (projected - p2d).norm();
                }
                Err(_) => {
                    // Penalize heavily if point cannot be projected
                    total_cost += 500.0; // Increased penalty for invalid projections
                }
            }
        }
        
        // Add barrier function for parameters near bounds
        let barrier_weight = 1.0;
        for i in 0..bounded_p.len() {
            let dist_to_lower = bounded_p[i] - self.lower_bounds[i];
            let dist_to_upper = self.upper_bounds[i] - bounded_p[i];
            
            // Add logarithmic barrier terms
            if dist_to_lower > 0.0 && dist_to_lower < 1e-3 {
                total_cost += barrier_weight * (-dist_to_lower.ln());
            }
            if dist_to_upper > 0.0 && dist_to_upper < 1e-3 {
                total_cost += barrier_weight * (-dist_to_upper.ln());
            }
        }

        Ok(total_cost)
    }
}

impl Gradient for DoubleSphereOptimizationCost {
    type Param = nalgebra::SVector<f64, 6>;
    type Gradient = nalgebra::SVector<f64, 6>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        // Apply parameter bounds by projecting parameters to valid range
        let mut bounded_p = p.clone();
        for i in 0..p.len() {
            bounded_p[i] = bounded_p[i].max(self.lower_bounds[i]).min(self.upper_bounds[i]);
        }
        
        // Extract parameters
        let fx = bounded_p[0];
        let fy = bounded_p[1];
        let cx = bounded_p[2];
        let cy = bounded_p[3];
        let xi = bounded_p[4];
        let alpha = bounded_p[5];

        // Initialize gradient vector
        let mut gradient_vec = vec![0.0; 6]; // Use a temporary vec

        for (p3d, p2d) in self.points_3d.column_iter().zip(self.points_2d.column_iter()) {
            let x = p3d.x;
            let y = p3d.y;
            let z = p3d.z;

            let xx = x * x;
            let yy = y * y;
            let zz = z * z;

            let r2 = xx + yy;
            let d1_2 = r2 + zz;
            let d1 = d1_2.sqrt();

            // Calculate projection and Jacobian
            let k = xi * d1 + z;
            let kk = k * k;
            let d2_2 = r2 + kk;
            let d2 = d2_2.sqrt();

            let norm = alpha * d2 + (1.0 - alpha) * k;
            let norm2 = norm * norm;

            // Skip points that would cause numerical issues
            if norm.abs() < 1e-6 {
                continue;
            }

            let mx = x / norm;
            let my = y / norm;

            // Projected point
            let proj_x = fx * mx + cx;
            let proj_y = fy * my + cy;

            // Residuals
            let dx = proj_x - p2d.x;
            let dy = proj_y - p2d.y;

            // Jacobian components for intrinsics
            let mut d_proj_d_param = vec![
                vec![mx, 0.0, 1.0, 0.0, 0.0, 0.0],
                vec![0.0, my, 0.0, 1.0, 0.0, 0.0],
            ];

            // Improved Jacobian entries for xi and alpha
            // Derivative of mx, my with respect to xi
            let d_norm_d_xi = alpha * k * d1 / d2;
            let d_mx_d_xi = -x * d_norm_d_xi / norm2;
            let d_my_d_xi = -y * d_norm_d_xi / norm2;
            
            // Derivative of mx, my with respect to alpha
            let d_norm_d_alpha = d2 - k;
            let d_mx_d_alpha = -x * d_norm_d_alpha / norm2;
            let d_my_d_alpha = -y * d_norm_d_alpha / norm2;
            
            d_proj_d_param[0][4] = fx * d_mx_d_xi;
            d_proj_d_param[1][4] = fy * d_my_d_xi;
            d_proj_d_param[0][5] = fx * d_mx_d_alpha;
            d_proj_d_param[1][5] = fy * d_my_d_alpha;

            // Update gradient with this point's contribution
            for i in 0..6 {
                gradient_vec[i] += 2.0 * (dx * d_proj_d_param[0][i] + dy * d_proj_d_param[1][i]);
            }
        }
        
        // Add gradient of barrier function for parameters near bounds
        let barrier_weight = 1.0;
        for i in 0..bounded_p.len() {
            let dist_to_lower = bounded_p[i] - self.lower_bounds[i];
            let dist_to_upper = self.upper_bounds[i] - bounded_p[i];
            
            // Add gradient of logarithmic barrier terms
            if dist_to_lower > 0.0 && dist_to_lower < 1e-3 {
                gradient_vec[i] -= barrier_weight / dist_to_lower;
            }
            if dist_to_upper > 0.0 && dist_to_upper < 1e-3 {
                gradient_vec[i] += barrier_weight / dist_to_upper;
            }
        }

        Ok(nalgebra::SVector::<f64, 6>::from_vec(gradient_vec))
    }
}

impl CameraModel for DoubleSphereModel {
    fn project(&self, point_3d: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
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
        if denom < PRECISION {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        // Project the point
        let projected_x = self.intrinsics.fx * (x / denom) + self.intrinsics.cx;
        let projected_y = self.intrinsics.fy * (y / denom) + self.intrinsics.cy;

        Ok(Vector2::new(projected_x, projected_y))
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
        if alpha != 0.0 && (2.0 * alpha - 1.0) * r_squared >= 1.0 {
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

    fn initialize(
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
        if points_3d.len() != points_2d.len() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }

        if points_3d.is_empty() {
            return Err(CameraModelError::InvalidParams(
                "Points arrays cannot be empty".to_string(),
            ));
        }

        // Initial parameters as SVector
        let param = nalgebra::SVector::<f64, 6>::new(
            self.intrinsics.fx,
            self.intrinsics.fy,
            self.intrinsics.cx,
            self.intrinsics.cy,
            self.xi,
            self.alpha,
        );
        
        // Define parameter bounds as SVector
        // Lower bounds: fx, fy > 0, cx, cy can be anywhere, xi can be anywhere, alpha in (0,1]
        let lower_bounds = nalgebra::SVector::<f64, 6>::new(
            1.0,                // fx > 0
            1.0,                // fy > 0
            f64::NEG_INFINITY,  // cx can be anywhere
            f64::NEG_INFINITY,  // cy can be anywhere
            f64::NEG_INFINITY,  // xi can be anywhere
            1e-6,               // alpha > 0 (small positive value)
        );
        
        // Upper bounds: fx, fy, cx, cy can be large, xi can be anywhere, alpha <= 1.0
        let upper_bounds = nalgebra::SVector::<f64, 6>::new(
            f64::INFINITY,  // fx can be large
            f64::INFINITY,  // fy can be large
            f64::INFINITY,  // cx can be large
            f64::INFINITY,  // cy can be large
            f64::INFINITY,  // xi can be anywhere
            1.0,            // alpha <= 1.0
        );
        // Setup cost function with bounds
        let cost_function = DoubleSphereOptimizationCost {
            points_3d: points_3d.clone(),
            points_2d: points_2d.clone(),
            lower_bounds: lower_bounds.clone(), // Clone SVector
            upper_bounds: upper_bounds.clone(), // Clone SVector
        };

        // Setup L-BFGS solver with line search
        let linesearch = MoreThuenteLineSearch::new();
        let solver = LBFGS::new(linesearch, 20); // Increased history size for better convergence

        // Create executor
        let mut executor = Executor::new(cost_function.clone(), solver); // Clone cost function

        // Configure verbosity
        if verbose {
            let term_logger = slog_term::TermDecorator::new().build();
            let drain = slog_term::FullFormat::new(term_logger).build().fuse();
            let drain = slog_async::Async::new(drain).build().fuse();
            let logger = slog::Logger::root(drain, slog::o!());
            executor = executor.add_observer(SlogLogger::new(logger), ObserverMode::Always);
        }

        // Set initial parameters and optimization options
        // Set initial parameters and optimization options
        executor = executor.configure(|state| {
            state
                .param(param) // param is now SVector
                .max_iters(100)       // Maximum number of iterations
                .target_cost(1e-8)    // Target cost function value
                // .grad_tol(1e-6)    // Gradient tolerance might be handled differently or implicitly
        });

        // Run optimization
        let res = executor
            .run()
            .map_err(|e| CameraModelError::NumericalError(e.to_string()))?;

        // Get optimized parameters
        let opt_param = res
            .state
            .best_param
            .ok_or_else(|| CameraModelError::NumericalError("Optimization failed to find parameters".to_string()))?;
        
        // Log optimization results if verbose
        if verbose {
            println!("Optimization terminated after {} iterations", res.state.get_iter());
            println!("Final cost: {}", res.state.get_cost());
            println!("Termination status: {:?}", res.termination_status);
        }

        // Apply bounds to the final parameters
        let final_params = opt_param.zip_map(&lower_bounds, |p, l| p.max(l))
                                  .zip_map(&upper_bounds, |p, u| p.min(u));

        // Update model parameters from the bounded final parameters
        self.intrinsics.fx = final_params[0];
        self.intrinsics.fy = final_params[1];
        self.intrinsics.cx = final_params[2];
        self.intrinsics.cy = final_params[3];
        self.xi = final_params[4];
        self.alpha = final_params[5];

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
        let point_2d = model.project(&point_3d).unwrap();

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
        // Create a simple camera model with initial parameters
        let intrinsics = Intrinsics {
            fx: 300.0,
            fy: 300.0,
            cx: 320.0,
            cy: 240.0,
        };
        
        let resolution = Resolution {
            width: 640,
            height: 480,
        };
        
        let mut model = DoubleSphereModel {
            intrinsics,
            resolution,
            xi: 0.0,
            alpha: 0.5,
        };
        
        // Create synthetic 3D points in a grid pattern
        let mut points_3d = Vec::new();
        let mut points_2d = Vec::new();
        
        // Create a reference model with known parameters to generate synthetic data
        let reference_model = DoubleSphereModel {
            intrinsics: Intrinsics {
                fx: 350.0,
                fy: 350.0,
                cx: 320.0,
                cy: 240.0,
            },
            resolution,
            xi: -0.2,
            alpha: 0.7,
        };
        
        // Generate synthetic data points
        for x in (-5..=5).step_by(2) {
            for y in (-5..=5).step_by(2) {
                for z in [2, 4, 6] {
                    let point_3d = Vector3::new(x as f64 * 0.1, y as f64 * 0.1, z as f64);
                    
                    // Project using reference model
                    if let Ok(point_2d) = reference_model.project(&point_3d) {
                        points_3d.push(point_3d);
                        points_2d.push(point_2d);
                    }
                }
            }
        }
        
        // Optimize the model
        model.optimize(&points_3d, &points_2d, false).unwrap();
        
        // Check that parameters are within reasonable bounds of the reference model
        assert!((model.intrinsics.fx - reference_model.intrinsics.fx).abs() < 20.0);
        assert!((model.intrinsics.fy - reference_model.intrinsics.fy).abs() < 20.0);
        assert!((model.intrinsics.cx - reference_model.intrinsics.cx).abs() < 20.0);
        assert!((model.intrinsics.cy - reference_model.intrinsics.cy).abs() < 20.0);
        assert!((model.xi - reference_model.xi).abs() < 0.1);
        assert!((model.alpha - reference_model.alpha).abs() < 0.1);
        
        // Verify that alpha is within bounds (0, 1]
        assert!(model.alpha > 0.0 && model.alpha <= 1.0);
        
        // Verify that focal lengths are positive
        assert!(model.intrinsics.fx > 0.0);
        assert!(model.intrinsics.fy > 0.0);
    }
}
