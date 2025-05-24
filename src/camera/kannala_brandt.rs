use log::info;
use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3xX, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use std::{f64::consts::PI, fs, io::Write};
use yaml_rust::YamlLoader;

use crate::camera::{validation, CameraModel, CameraModelError, Intrinsics, Resolution};
use crate::optimization::kannala_brandt::KannalaBrandtOptimizationCost; // Added
use crate::optimization::Optimizer; // Added

use argmin::core::{observers::ObserverMode, Error as ArgminError, Executor, State}; // Modified
use argmin::solver::gaussnewton::GaussNewton; // Added
use argmin_observer_slog::SlogLogger; // Added

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KannalaBrandtModel {
    pub intrinsics: Intrinsics,
    pub resolution: Resolution,
    pub distortions: [f64; 4], // k1, k2, k3, k4
}

// Removed KannalaBrandtOptimizationCost struct and its trait implementations

impl KannalaBrandtModel {
    // Constructor used by optimization cost function
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
        Ok(model)
    }
}

impl CameraModel for KannalaBrandtModel {
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
            CameraModelError::InvalidParams("Invalid intrinsics format".to_string())
        })?;
        if intrinsics_yaml.len() < 4 {
            return Err(CameraModelError::InvalidParams(
                "Intrinsics vector too short".to_string(),
            ));
        }

        let resolution_yaml = cam_node["resolution"].as_vec().ok_or_else(|| {
            CameraModelError::InvalidParams("Invalid resolution format".to_string())
        })?;
        if resolution_yaml.len() < 2 {
            return Err(CameraModelError::InvalidParams(
                "Resolution vector too short".to_string(),
            ));
        }

        let distortion_coeffs_yaml = cam_node["distortion"].as_vec().ok_or_else(|| {
            CameraModelError::InvalidParams(
                "Missing or invalid distortion coefficients".to_string(),
            )
        })?;
        if distortion_coeffs_yaml.len() < 4 {
            return Err(CameraModelError::InvalidParams(
                "'distortion_coeffs' must have at least 4 elements".to_string(),
            ));
        }

        let intrinsics = Intrinsics {
            fx: intrinsics_yaml[0]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid fx".to_string()))?,
            fy: intrinsics_yaml[1]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid fy".to_string()))?,
            cx: intrinsics_yaml[2]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid cx".to_string()))?,
            cy: intrinsics_yaml[3]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid cy".to_string()))?,
        };

        let resolution = Resolution {
            width: resolution_yaml[0]
                .as_i64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid width".to_string()))?
                as u32,
            height: resolution_yaml[1]
                .as_i64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid height".to_string()))?
                as u32,
        };

        let distortions = [
            distortion_coeffs_yaml[0]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid k1".to_string()))?,
            distortion_coeffs_yaml[1]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid k2".to_string()))?,
            distortion_coeffs_yaml[2]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid k3".to_string()))?,
            distortion_coeffs_yaml[3]
                .as_f64()
                .ok_or_else(|| CameraModelError::InvalidParams("Invalid k4".to_string()))?,
        ];

        let model = KannalaBrandtModel {
            intrinsics,
            resolution,
            distortions,
        };

        model.validate_params()?;
        Ok(model)
    }

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

    fn validate_params(&self) -> Result<(), CameraModelError> {
        validation::validate_intrinsics(&self.intrinsics)?;
        // No specific validation for Kannala-Brandt distortions k1-k4 mentioned,
        // they can be positive or negative.
        Ok(())
    }

    fn get_resolution(&self) -> Resolution {
        self.resolution.clone()
    }

    fn get_intrinsics(&self) -> Intrinsics {
        self.intrinsics.clone()
    }

    fn get_distortion(&self) -> Vec<f64> {
        self.distortions.to_vec()
    }

    // linear_estimation removed from impl CameraModel for KannalaBrandtModel
    // optimize removed from impl CameraModel for KannalaBrandtModel
}

impl Optimizer for KannalaBrandtModel {
    fn optimize(
        &mut self,
        points_3d: &Matrix3xX<f64>,
        points_2d: &Matrix2xX<f64>,
        verbose: bool,
    ) -> Result<(), CameraModelError> {
        // This is the same implementation as the original optimize method
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

        let cost_function =
            KannalaBrandtOptimizationCost::new(points_3d.clone(), points_2d.clone());

        let initial_params_vec = vec![
            self.intrinsics.fx,
            self.intrinsics.fy,
            self.intrinsics.cx,
            self.intrinsics.cy,
            self.distortions[0],
            self.distortions[1],
            self.distortions[2],
            self.distortions[3],
        ];
        let init_param: DVector<f64> = DVector::from_vec(initial_params_vec);

        let solver = GaussNewton::new();

        let mut executor_builder = Executor::new(cost_function, solver)
            .configure(|state| state.param(init_param).max_iters(100));

        if verbose {
            executor_builder =
                executor_builder.add_observer(SlogLogger::term(), ObserverMode::Always);
            info!("Starting Kannala-Brandt optimization with Gauss-Newton...");
        } else {
            executor_builder =
                executor_builder.add_observer(SlogLogger::term(), ObserverMode::NewBest);
        }

        let res = executor_builder
            .run()
            .map_err(|e| CameraModelError::NumericalError(e.to_string()))?;

        if verbose {
            info!("Optimization finished: \n{}", res);
            info!("Termination status: {:?}", res.state().termination_status);
        }

        if let Some(best_params_dvec) = res.state().get_best_param() {
            self.intrinsics.fx = best_params_dvec[0];
            self.intrinsics.fy = best_params_dvec[1];
            self.intrinsics.cx = best_params_dvec[2];
            self.intrinsics.cy = best_params_dvec[3];
            self.distortions[0] = best_params_dvec[4];
            self.distortions[1] = best_params_dvec[5];
            self.distortions[2] = best_params_dvec[6];
            self.distortions[3] = best_params_dvec[7];
            self.validate_params()?;
        } else {
            return Err(CameraModelError::NumericalError(
                "Optimization failed to find best parameters".to_string(),
            ));
        }
        Ok(())
    }

    fn linear_estimation(
        intrinsics: &Intrinsics,
        resolution: &Resolution,
        points_2d: &Matrix2xX<f64>,
        points_3d: &Matrix3xX<f64>,
    ) -> Result<Self, CameraModelError>
    where
        Self: Sized,
    {
        // Duplicating the implementation from CameraModel trait for now
        if points_3d.ncols() != points_2d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }
        if points_3d.ncols() < 4 {
            return Err(CameraModelError::InvalidParams(
                "Not enough points for linear estimation (need at least 4)".to_string(),
            ));
        }

        let num_points = points_3d.ncols();
        let mut a_mat = DMatrix::zeros(num_points * 2, 4);
        let mut b_vec = DVector::zeros(num_points * 2);

        for i in 0..num_points {
            let p3d = points_3d.column(i);
            let p2d = points_2d.column(i);

            let x_world = p3d.x;
            let y_world = p3d.y;
            let z_world = p3d.z;

            let u_img = p2d.x;
            let v_img = p2d.y;

            if z_world <= f64::EPSILON {
                continue;
            }

            let r_world = (x_world * x_world + y_world * y_world).sqrt();
            let theta = r_world.atan2(z_world);

            let theta2 = theta * theta;
            let theta3 = theta2 * theta;
            let theta5 = theta3 * theta2;
            let theta7 = theta5 * theta2;
            let theta9 = theta7 * theta2;

            a_mat[(i * 2, 0)] = theta3;
            a_mat[(i * 2, 1)] = theta5;
            a_mat[(i * 2, 2)] = theta7;
            a_mat[(i * 2, 3)] = theta9;

            a_mat[(i * 2 + 1, 0)] = theta3;
            a_mat[(i * 2 + 1, 1)] = theta5;
            a_mat[(i * 2 + 1, 2)] = theta7;
            a_mat[(i * 2 + 1, 3)] = theta9;

            let x_r = if r_world < f64::EPSILON { 0.0 } else { x_world / r_world };
            let y_r = if r_world < f64::EPSILON { 0.0 } else { y_world / r_world };

            if (intrinsics.fx * x_r).abs() < f64::EPSILON && x_r.abs() > f64::EPSILON {
                return Err(CameraModelError::NumericalError(
                    "fx * x_r is zero in linear estimation".to_string(),
                ));
            }
            if (intrinsics.fy * y_r).abs() < f64::EPSILON && y_r.abs() > f64::EPSILON {
                return Err(CameraModelError::NumericalError(
                    "fy * y_r is zero in linear estimation".to_string(),
                ));
            }

            if x_r.abs() > f64::EPSILON {
                b_vec[i * 2] = (u_img - intrinsics.cx) / (intrinsics.fx * x_r) - theta;
            } else {
                b_vec[i * 2] = if (u_img - intrinsics.cx).abs() < f64::EPSILON { -theta } else { 0.0 };
            }

            if y_r.abs() > f64::EPSILON {
                b_vec[i * 2 + 1] = (v_img - intrinsics.cy) / (intrinsics.fy * y_r) - theta;
            } else {
                b_vec[i * 2 + 1] = if (v_img - intrinsics.cy).abs() < f64::EPSILON { -theta } else { 0.0 };
            }
        }

        let svd = a_mat.svd(true, true);
        let x_coeffs = svd.solve(&b_vec, f64::EPSILON).map_err(|e_str| {
            CameraModelError::NumericalError(format!(
                "SVD solve failed in linear estimation: {}",
                e_str
            ))
        })?;

        let model = KannalaBrandtModel {
            intrinsics: intrinsics.clone(),
            resolution: resolution.clone(),
            distortions: [x_coeffs[0], x_coeffs[1], x_coeffs[2], x_coeffs[3]],
        };

        model.validate_params()?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq; 
    use crate::optimization::Optimizer as _; // Import for Optimizer trait methods in tests


    // Helper function to get a sample KannalaBrandtModel instance
    // This could load from YAML or use the new() function with sample parameters
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
        assert_relative_eq!(model.distortions[3], 0.04362766880887814, epsilon = 1e-9);
        // k4
    }

    #[test]
    fn test_load_from_yaml_file_not_found() {
        let model_result = KannalaBrandtModel::load_from_yaml("samples/non_existent_file.yaml");
        assert!(model_result.is_err());
        match model_result.err().unwrap() {
            CameraModelError::IOError(_) => {} // Expected
            _ => panic!("Expected IOError"),
        }
    }

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
                            epsilon = 1e-5
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

    #[test]
    fn test_project_point_at_center() {
        let model = get_sample_kb_model();
        let point_3d = Vector3::new(0.0, 0.0, 0.0); // Point at camera optical center
        let result = model.project(&point_3d, false);
        assert!(matches!(result, Err(CameraModelError::PointAtCameraCenter)));
    }

    #[test]
    fn test_project_point_behind_camera() {
        let model = get_sample_kb_model();
        let point_3d = Vector3::new(0.1, 0.2, -1.0); // Point behind camera
        let result = model.project(&point_3d, false);
        assert!(matches!(result, Err(CameraModelError::PointIsOutSideImage)));
    }

    #[test]
    fn test_project_with_jacobian() {
        let model = get_sample_kb_model();
        let point_3d = Vector3::new(0.1, 0.2, 1.0);

        match model.project(&point_3d, true) {
            Ok((_point_2d, jacobian_option)) => {
                assert!(jacobian_option.is_some());
                let jacobian = jacobian_option.unwrap();
                assert_eq!(jacobian.nrows(), 2); // 2D point (x, y)
                assert_eq!(jacobian.ncols(), 8); // 8 parameters (fx, fy, cx, cy, k1, k2, k3, k4)
                                                 // Further checks on Jacobian values would require numerical differentiation
                                                 // or known analytical values for specific points.
            }
            Err(e) => panic!("Projection failed: {:?}", e),
        }
    }

    #[test]
    fn test_unproject_out_of_bounds() {
        let model = get_sample_kb_model();
        let point_2d_outside = Vector2::new(
            model.resolution.width as f64 + 10.0,
            model.resolution.height as f64 + 10.0,
        );
        let result = model.unproject(&point_2d_outside);
        assert!(matches!(result, Err(CameraModelError::PointIsOutSideImage)));
    }

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
        assert_relative_eq!(distortion_coeffs[3], 0.04362766880887814); // k4
    }

    // --- Tests for linear_estimation and optimize ---
    // These tests are more complex and depend on the full implementation
    // of these methods and potentially argmin cost functions.
    // Below are sketches of what these tests might look like.

    fn generate_synthetic_data(
        model: &KannalaBrandtModel,
        num_points: usize,
    ) -> (Matrix3xX<f64>, Matrix2xX<f64>) {
        let mut points_3d_vec = Vec::new();
        let mut points_2d_vec = Vec::new();

        // Generate some 3D points in a reasonable FOV
        for i in 0..num_points {
            let x = (i as f64 * 0.1) - (num_points as f64 * 0.05); // Spread points
            let y = (i as f64 * 0.05) - (num_points as f64 * 0.025);
            let z = 1.0 + (i as f64 * 0.01); // Vary depth
            let p3d = Vector3::new(x, y, z);

            if let Ok((p2d, _)) = model.project(&p3d, false) {
                // Ensure point is within image (simplified check)
                if p2d.x > 0.0
                    && p2d.x < model.resolution.width as f64
                    && p2d.y > 0.0
                    && p2d.y < model.resolution.height as f64
                {
                    points_3d_vec.push(p3d);
                    points_2d_vec.push(p2d);
                }
            }
        }

        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);
        (points_3d, points_2d)
    }

    #[test]
    fn test_linear_estimation() {
        let ground_truth_model = get_sample_kb_model();
        let (points_3d, points_2d) = generate_synthetic_data(&ground_truth_model, 20);

        // Assume intrinsics (fx, fy, cx, cy) are known for linear estimation of distortion
        let estimated_model_result = KannalaBrandtModel::linear_estimation(
            &ground_truth_model.intrinsics,
            &ground_truth_model.resolution,
            &points_2d,
            &points_3d,
        );

        assert!(
            estimated_model_result.is_ok(),
            "Linear estimation failed: {:?}",
            estimated_model_result.err()
        );
        let estimated_model = estimated_model_result.unwrap();

        // Compare estimated distortion with ground truth
        // Linear estimation might not be perfectly accurate, so use a larger epsilon
        for i in 0..4 {
            assert_relative_eq!(
                estimated_model.distortions[i],
                ground_truth_model.distortions[i],
                epsilon = 1e-1
            );
        }
    }

    // For optimize test, you would need:
    // 1. KannalaBrandtOptimizationCost struct (similar to DoubleSphereOptimizationCost)
    //    - Implementing argmin traits: Operator, Jacobian, CostFunction, Gradient, Hessian
    // 2. The optimize method in KannalaBrandtModel using argmin (e.g., GaussNewton solver)

    
    #[test]
    fn test_optimize_trait_method() { // Renamed to avoid conflict if original test_optimize is kept
        let ground_truth_model = get_sample_kb_model();
        let (points_3d, points_2d) = generate_synthetic_data(&ground_truth_model, 50);

        // Create an initial model with slightly perturbed parameters
        let mut initial_model = ground_truth_model.clone();
        initial_model.intrinsics.fx *= 1.05;
        initial_model.intrinsics.cy *= 0.95;
        initial_model.distortions[0] *= 1.2;
        initial_model.distortions[2] *= 0.8;

        // Use the Optimizer trait's optimize method
        let optimize_result = Optimizer::optimize(&mut initial_model, &points_3d, &points_2d, false);
        assert!(optimize_result.is_ok(), "Optimization failed: {:?}", optimize_result.err());

        let optimized_model = initial_model; // optimize should modify in place

        // Compare optimized parameters with ground truth
        assert_relative_eq!(optimized_model.intrinsics.fx, ground_truth_model.intrinsics.fx, epsilon = 1e-3);
        assert_relative_eq!(optimized_model.intrinsics.fy, ground_truth_model.intrinsics.fy, epsilon = 1e-3);
        assert_relative_eq!(optimized_model.intrinsics.cx, ground_truth_model.intrinsics.cx, epsilon = 1e-3);
        assert_relative_eq!(optimized_model.intrinsics.cy, ground_truth_model.intrinsics.cy, epsilon = 1e-3);

        for i in 0..4 {
            assert_relative_eq!(optimized_model.distortions[i], ground_truth_model.distortions[i], epsilon = 1e-3);
        }
    }
    
}
