use nalgebra::{Matrix2xX, Matrix3xX, Point2, Point3};
use serde::{Deserialize, Serialize};
use std::fs;
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
    fn project(&self, point_3d: &Point3<f64>) -> Result<Point2<f64>, CameraModelError> {
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

        Ok(Point2::new(projected_x, projected_y))
    }

    fn unproject(&self, point_2d: &Point2<f64>) -> Result<Point3<f64>, CameraModelError> {
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
        let point3d = Point3::new(coeff * mx, coeff * my, coeff * mz - xi);

        // Normalize the point to get a unit vector
        let norm = point3d.coords.norm();

        Ok(Point3::new(
            point3d.x / norm,
            point3d.y / norm,
            point3d.z / norm,
        ))
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
        // Implementation for saving to YAML
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

    fn initialize(
        intrinsics: &Intrinsics,
        resolution: &Resolution,
        _points_2d: &Matrix2xX<f64>,
        _points_3d: &Matrix3xX<f64>,
    ) -> Result<Self, CameraModelError> {
        let model = DoubleSphereModel {
            intrinsics: intrinsics.clone(),
            resolution: resolution.clone(),
            xi: 0.0,
            alpha: 1.0,
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
    fn test_double_sphere_project_unproject() {
        // Load the camera model from YAML
        let path = "samples/double_sphere.yaml";
        let model = DoubleSphereModel::load_from_yaml(path).unwrap();

        // Create a 3D point in camera coordinates (pointing somewhat forward and to the side)
        let point_3d = Point3::new(0.5, -0.3, 2.0);
        let norm_3d = point_3d / point_3d.coords.norm();

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
