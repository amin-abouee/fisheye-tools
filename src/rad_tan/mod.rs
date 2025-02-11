use nalgebra::{Point2, Point3};
use serde::{Deserialize, Serialize};
use std::{fs};
use yaml_rust::YamlLoader;

use crate::camera::{CameraModel, Intrinsics, Resolution, CameraModelError, validation};

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
        let u: f64 = self.intrinsics.fx * point_3d.x / point_3d.z + self.intrinsics.cx;
        let v: f64 = self.intrinsics.fy * point_3d.y / point_3d.z + self.intrinsics.cy;

        if u < 0.0 || u >= self.resolution.width as f64 || v < 0.0 || v >= self.resolution.height as f64 {
            return Err(CameraModelError::ProjectionOutsideImage);
        }

        Ok(Point2::new(u, v))
    }

    fn unproject(&self, point_2d: &Point2<f64>) -> Result<Point3<f64>, CameraModelError> {
        
        if point_2d.x < 0.0 || point_2d.x >= self.resolution.width as f64 || point_2d.y < 0.0 || point_2d.y >= self.resolution.height as f64 {
            return Err(CameraModelError::PointIsOutsideImage);
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
        let _docs = YamlLoader::load_from_str(&contents)?;
        // TODO: Parse YAML and create RadTanModel
        unimplemented!()
    }

    fn validate_params(&self) -> Result<(), CameraModelError> {
        validation::validate_intrinsics(&self.intrinsics)?;
        if self.distortion.len() != 5 {
            return Err(CameraModelError::InvalidParams("RadTan model requires 5 distortion parameters".to_string()));
        }
        Ok(())
    }
}