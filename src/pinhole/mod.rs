use nalgebra::{Point2, Point3};
use serde::{Deserialize, Serialize};
use std::fs;
use yaml_rust::YamlLoader;

use crate::camera::{validation, CameraModel, CameraModelError, Intrinsics, Resolution};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinholeModel {
    pub intrinsics: Intrinsics,
    pub resolution: Resolution,
}

impl CameraModel for PinholeModel {
    fn project(&self, point_3d: &Point3<f64>) -> Result<Point2<f64>, CameraModelError> {
        // If z is very small, the point is at the camera center
        if point_3d.z < f64::EPSILON.sqrt() {
            return Err(CameraModelError::PointAtCameraCenter);
        }
        let u: f64 = self.intrinsics.fx * point_3d.x / point_3d.z + self.intrinsics.cx;
        let v: f64 = self.intrinsics.fy * point_3d.y / point_3d.z + self.intrinsics.cy;

        if u < 0.0
            || u >= self.resolution.width as f64
            || v < 0.0
            || v >= self.resolution.height as f64
        {
            return Err(CameraModelError::ProjectionOutSideImage);
        }

        Ok(Point2::new(u, v))
    }

    fn unproject(&self, point_2d: &Point2<f64>) -> Result<Point3<f64>, CameraModelError> {
        if point_2d.x < 0.0
            || point_2d.x >= self.resolution.width as f64
            || point_2d.y < 0.0
            || point_2d.y >= self.resolution.height as f64
        {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let mx: f64 = (point_2d.x - self.intrinsics.cx) / self.intrinsics.fx;
        let my: f64 = (point_2d.y - self.intrinsics.cy) / self.intrinsics.fy;

        let r2: f64 = mx * mx + my * my;

        let norm: f64 = (1.0 as f64 + r2).sqrt();
        let norm_inv: f64 = 1.0 as f64 / norm;

        Ok(Point3::new(mx * norm_inv, my * norm_inv, norm_inv))
    }

    fn load_from_yaml(path: &str) -> Result<Self, CameraModelError> {
        let contents = fs::read_to_string(path)?;
        let docs = YamlLoader::load_from_str(&contents)?;
        let doc = &docs[0];

        let intrinsics = doc["cam0"]["intrinsics"]
            .as_vec()
            .ok_or_else(|| CameraModelError::InvalidParams("Invalid intrinsics".to_string()))?;
        let resolution = doc["cam0"]["resolution"]
            .as_vec()
            .ok_or_else(|| CameraModelError::InvalidParams("Invalid resolution".to_string()))?;

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

        let model = PinholeModel {
            intrinsics,
            resolution,
        };

        // Validate parameters
        model.validate_params()?;

        Ok(model)
    }

    fn validate_params(&self) -> Result<(), CameraModelError> {
        validation::validate_intrinsics(&self.intrinsics)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinhole_load_from_yaml() {
        let path = "src/pinhole/pinhole.yaml";
        let model = PinholeModel::load_from_yaml(path).unwrap();

        assert_eq!(model.intrinsics.fx, 461.629);
        assert_eq!(model.intrinsics.fy, 460.152);
        assert_eq!(model.intrinsics.cx, 362.680);
        assert_eq!(model.intrinsics.cy, 246.049);
        assert_eq!(model.resolution.width, 752);
        assert_eq!(model.resolution.height, 480);
    }

    #[test]
    fn test_pinhole_project_unproject() {
        let path = "src/pinhole/pinhole.yaml";
        let model = PinholeModel::load_from_yaml(path).unwrap();

        // 3D point in camera coordinates
        let point_3d = Point3::new(1.0, 1.0, 5.0);
        let norm_3d = point_3d / point_3d.coords.norm();

        // Project the 3D point to 2D
        let point_2d = model.project(&point_3d).unwrap();

        // Unproject the 2D point back to 3D
        let point_3d_unprojected = model.unproject(&point_2d).unwrap();

        // Check if the unprojected point is close to the original point
        assert!((norm_3d.x - point_3d_unprojected.x).abs() < 1e-6);
        assert!((norm_3d.y - point_3d_unprojected.y).abs() < 1e-6);
        assert!((norm_3d.z - point_3d_unprojected.z).abs() < 1e-6);
    }
}
