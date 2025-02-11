use nalgebra::{Point2, Point3};
use serde::{Deserialize, Serialize};
use std::fs;
use yaml_rust::YamlLoader;

use crate::camera::{CameraModel, Intrinsics, Resolution, CameraModelError, validation};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoubleSphereModel {
    pub intrinsics: Intrinsics,
    pub resolution: Resolution,
    pub xi: f64,
    pub alpha: f64,
}

impl CameraModel for DoubleSphereModel {
    fn project(&self, point_3d: &Point3<f64>) -> Result<Point2<f64>, CameraModelError> {
        // TODO: Implement double sphere projection
        unimplemented!()
    }

    fn unproject(&self, point_2d: &Point2<f64>) -> Result<Point3<f64>, CameraModelError> {
        // TODO: Implement double sphere unprojection
        unimplemented!()
    }

    fn load_from_yaml(path: &str) -> Result<Self, CameraModelError> {
        let contents = fs::read_to_string(path)?;
        let docs = YamlLoader::load_from_str(&contents)?;
        // TODO: Parse YAML and create DoubleSphereModel
        unimplemented!()
    }

    fn validate_params(&self) -> Result<(), CameraModelError> {
        validation::validate_intrinsics(&self.intrinsics)?;
        if !self.xi.is_finite() || !self.alpha.is_finite() {
            return Err(CameraModelError::InvalidParams("xi and alpha must be finite".to_string()));
        }
        Ok(())
    }
} 