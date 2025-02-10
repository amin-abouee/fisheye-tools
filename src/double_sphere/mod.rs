use nalgebra::{Point2, Point3};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use yaml_rust::YamlLoader;

use super::{CameraModel, Intrinsics, validation};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoubleSphereModel {
    pub intrinsics: Intrinsics,
    pub xi: f64,
    pub alpha: f64,
}

impl CameraModel for DoubleSphereModel {
    fn project(&self, point_3d: &Point3<f64>) -> Result<Point2<f64>, Box<dyn Error>> {
        // TODO: Implement double sphere projection
        unimplemented!()
    }

    fn unproject(&self, point_2d: &Point2<f64>) -> Result<Point3<f64>, Box<dyn Error>> {
        // TODO: Implement double sphere unprojection
        unimplemented!()
    }

    fn load_from_yaml(path: &str) -> Result<Self, Box<dyn Error>> {
        let contents = fs::read_to_string(path)?;
        let docs = YamlLoader::load_from_str(&contents)?;
        // TODO: Parse YAML and create DoubleSphereModel
        unimplemented!()
    }

    fn validate_params(&self) -> Result<(), Box<dyn Error>> {
        validation::validate_intrinsics(&self.intrinsics)?;
        if !self.xi.is_finite() || !self.alpha.is_finite() {
            return Err("Double sphere parameters must be finite".into());
        }
        Ok(())
    }
} 