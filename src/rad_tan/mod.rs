use nalgebra::{Point2, Point3};
use serde::{Deserialize, Serialize};
use std::fs;
use yaml_rust::YamlLoader;

use crate::camera::{validation, CameraModel, CameraModelError, Intrinsics, Resolution};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadTanModel {
    pub intrinsics: Intrinsics,
    pub resolution: Resolution,
    pub distortion: Vec<f64>, // k1, k2, p1, p2, k3
}

impl CameraModel for RadTanModel {
    fn project(&self, point_3d: &Point3<f64>) -> Result<Point2<f64>, CameraModelError> {
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

        let mut distortion = Vec::with_capacity(5);

        for param in distortion_node {
            let value = param.as_f64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid distortion parameter".to_string())
            })?;
            distortion.push(value);
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

    fn validate_params(&self) -> Result<(), CameraModelError> {
        validation::validate_intrinsics(&self.intrinsics)?;
        if self.distortion.len() != 5 {
            return Err(CameraModelError::InvalidParams(
                "RadTan model requires 5 distortion parameters".to_string(),
            ));
        }
        Ok(())
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_load_from_yaml() {
        let path = "src/rad_tan/radtan.yaml";
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
}
