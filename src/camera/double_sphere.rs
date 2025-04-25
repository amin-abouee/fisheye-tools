use nalgebra::{Matrix2xX, Matrix3xX, Vector2, Vector3};
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
}
