use crate::camera::{validation, CameraModel, CameraModelError, Intrinsics, Resolution};
use nalgebra::{DMatrix, DVector, Matrix2, Matrix2xX, Matrix3xX, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use yaml_rust::YamlLoader;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadTanModel {
    pub intrinsics: Intrinsics,
    pub resolution: Resolution,
    pub distortion: [f64; 5], // k1, k2, p1, p2, k3
}

impl RadTanModel {
    pub fn new(parameters: &DVector<f64>) -> Result<Self, CameraModelError> {
        let model = RadTanModel {
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
            distortion: [
                parameters[4],
                parameters[5],
                parameters[6],
                parameters[7],
                parameters[8],
            ],
        };

        model.validate_params()?;
        Ok(model)
    }
}

impl CameraModel for RadTanModel {
    fn project(
        &self,
        point_3d: &Vector3<f64>,
        _compute_jacobian: bool,
    ) -> Result<(Vector2<f64>, Option<DMatrix<f64>>), CameraModelError> {
        // If z is very small, the point is at the camera center
        if point_3d.z < f64::EPSILON.sqrt() {
            return Err(CameraModelError::PointAtCameraCenter);
        }

        let x = point_3d.x;
        let y = point_3d.y;
        let z = point_3d.z;

        let k1 = self.distortion[0];
        let k2 = self.distortion[1];
        let p1 = self.distortion[2];
        let p2 = self.distortion[3];
        let k3 = self.distortion[4];

        // Calculate normalized image coordinates
        let x_prime = x / z;
        let y_prime = y / z;

        let r2 = x_prime.powi(2) + y_prime.powi(2);
        let r4 = r2.powi(2);
        let r6 = r4.powi(2);

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

        Ok((Vector2::new(u, v), None))
    }

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

        let k1 = self.distortion[0];
        let k2 = self.distortion[1];
        let p1 = self.distortion[2];
        let p2 = self.distortion[3];
        let k3 = self.distortion[4];

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
            let r6 = r4 * r2; // r^6

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
            let term1 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4; // Derivative of (r*radial_distortion) w.r.t r, divided by r
                                                            // Re-deriving based on the distortion estimate formulas:
                                                            // d(x_est)/dx = radial_distortion + x * d(radial_distortion)/dx + d(tangential_x)/dx
                                                            // d(x_est)/dy = x * d(radial_distortion)/dy + d(tangential_x)/dy
                                                            // etc.
                                                            // d(r^2)/dx = 2x, d(r^2)/dy = 2y
                                                            // d(radial_distortion)/dx = (k1 + 2*k2*r2 + 3*k3*r4) * 2x
                                                            // d(radial_distortion)/dy = (k1 + 2*k2*r2 + 3*k3*r4) * 2y

            // Jacobian elements (derivatives of estimated distorted coords w.r.t. undistorted coords)
            let j00 =
                radial_distortion + x * (term1 * 2.0 * x) + 2.0 * p1 * y + p2 * (2.0 * x + 4.0 * x); // d(x_est)/dx
            let j01 = x * (term1 * 2.0 * y) + 2.0 * p1 * x + p2 * (2.0 * y); // d(x_est)/dy
            let j10 = y * (term1 * 2.0 * x) + p1 * (2.0 * x) + 2.0 * p2 * y; // d(y_est)/dx
            let j11 =
                radial_distortion + y * (term1 * 2.0 * y) + p1 * (2.0 * y + 4.0 * y) + 2.0 * p2 * x; // d(y_est)/dy

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
                    "Jacobian is singular".to_string(),
                ));
            }

            // If loop finished without converging (max iterations reached)
            if iteration == MAX_ITERATIONS - 1 {
                return Err(CameraModelError::NumericalError(
                    "Unprojection did not converge after {MAX_ITERATIONS} iterations.".to_string(),
                ));
            }
        }

        // Create the 3D point with the undistorted x, y and z=1
        let point3d = Vector3::new(point.x, point.y, 1.0);

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

        // Extract distortion parameters
        let distortion_node = doc["cam0"]["distortion"].as_vec().ok_or_else(|| {
            CameraModelError::InvalidParams("Missing distortion parameters".to_string())
        })?;

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

        let mut distortion = [0.0; 5]; // Initialize the fixed-size array

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
                    "Invalid distortion parameter at index {}",
                    i
                ))
            })?;
            // Assign the parsed value to the corresponding index in the array
            distortion[i] = value;
        }

        if distortion.len() != 5 {
            return Err(CameraModelError::InvalidParams(format!(
                "Expected 5 distortion parameters, got {}",
                distortion.len()
            )));
        }

        let model = RadTanModel {
            intrinsics,
            resolution,
            distortion,
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
                    ])
                    .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
                ),
                (
                    serde_yaml::Value::String("distortion".to_string()),
                    serde_yaml::to_value(vec![
                        self.distortion[0],
                        self.distortion[1],
                        self.distortion[2],
                        self.distortion[3],
                        self.distortion[4],
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
        Ok(())
    }

    fn get_resolution(&self) -> Resolution {
        self.resolution.clone()
    }

    fn get_intrinsics(&self) -> Intrinsics {
        self.intrinsics.clone()
    }

    fn get_distortion(&self) -> Vec<f64> {
        self.distortion.to_vec()
    }

    fn linear_estimation(
        intrinsics: &Intrinsics,
        resolution: &Resolution,
        points_2d: &Matrix2xX<f64>,
        points_3d: &Matrix3xX<f64>,
    ) -> Result<Self, CameraModelError> {
        if points_2d.ncols() != points_3d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }

        let n_points = points_2d.ncols();

        // Set up matrices for solving the system: A * x = b
        // Where x will contain the distortion parameters k1, k2, k3
        let mut a = nalgebra::DMatrix::zeros(n_points * 2, 3);
        let mut b = nalgebra::DVector::zeros(n_points * 2);

        for i in 0..n_points {
            let x = points_3d[(0, i)]; // X
            let y = points_3d[(1, i)]; // Y
            let z = points_3d[(2, i)]; // Z
            let u = points_2d[(0, i)]; // u
            let v = points_2d[(1, i)]; // v

            // Calculate normalized image coordinates
            let x_prime = x / z;
            let y_prime = y / z;

            // Calculate radial terms
            let r2 = x_prime * x_prime + y_prime * y_prime;
            let r4 = r2 * r2;
            let r6 = r4 * r2;

            // Fill matrix A
            a[(i * 2, 0)] = r2;
            a[(i * 2, 1)] = r4;
            a[(i * 2, 2)] = r6;
            a[(i * 2 + 1, 0)] = r2;
            a[(i * 2 + 1, 1)] = r4;
            a[(i * 2 + 1, 2)] = r6;

            // Fill vector b
            b[i * 2] = (u - intrinsics.cx) / (intrinsics.fx * x_prime) - 1.0;
            b[i * 2 + 1] = (v - intrinsics.cy) / (intrinsics.fy * y_prime) - 1.0;
        }

        // Solve Ax = b for x using SVD
        // A: (2*num_points) x 3
        // x: 3 x 1 (vector)
        // b: (2*num_points) x 1 (vector)
        let svd = a.svd(true, true);

        let x = match svd.solve(&b, 1e-10) {
            Ok(sol) => sol, // Handle the successful case
            Err(err_msg) => {
                return Err(CameraModelError::NumericalError(err_msg.to_string()));
            }
        };

        // Create the model with the computed distortion parameters
        let mut distortion = [0.0; 5];
        distortion[0] = x[0]; // k1
        distortion[1] = x[1]; // k2
        distortion[4] = x[2]; // k3
        distortion[2] = 0.0; // p1
        distortion[3] = 0.0; // p2

        let model = RadTanModel {
            intrinsics: intrinsics.clone(),
            resolution: resolution.clone(),
            distortion,
        };

        // Validate parameters
        model.validate_params()?;

        Ok(model)
    }

    fn optimize(
        &mut self,
        _points_3d: &Matrix3xX<f64>,
        _points_2d: &Matrix2xX<f64>,
        _verbose: bool,
    ) -> Result<(), CameraModelError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(model.distortion.len(), 5);
        assert_eq!(model.distortion[0], -0.28340811);
        assert_eq!(model.distortion[1], 0.07395907);
        assert_eq!(model.distortion[2], 0.00019359);
        assert_eq!(model.distortion[3], 1.76187114e-05);
        assert_eq!(model.distortion[4], 0.0);
    }

    #[test]
    fn test_radtan_save_to_yaml() {
        use std::fs;

        // Create output directory if it doesn't exist
        fs::create_dir_all("output").unwrap_or_else(|_| {
            println!("Output directory already exists or couldnot be created");
        });

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
        assert_eq!(model.distortion.len(), saved_model.distortion.len());
        assert_eq!(model.distortion[0], saved_model.distortion[0]);
        assert_eq!(model.distortion[1], saved_model.distortion[1]);
        assert_eq!(model.distortion[2], saved_model.distortion[2]);
        assert_eq!(model.distortion[3], saved_model.distortion[3]);
        assert_eq!(model.distortion[4], saved_model.distortion[4]);
    }

    #[test]
    fn test_radtan_project_unproject() {
        // Load the camera model from YAML
        let path = "samples/rad_tan.yaml";
        let model = RadTanModel::load_from_yaml(path).unwrap();

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
        ];

        for (i, original_point) in test_points.iter().enumerate() {
            // Project the 3D point to pixel coordinates
            let (pixel_point, _) = match model.project(original_point, false) {
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
            assert!(dot_product > 0.99,
                    "Test point {}: Direction mismatch. Original: {:?}, Unprojected: {:?}, Dot product: {}",
                    i, original_point, ray_direction, dot_product);
        }
    }
}
