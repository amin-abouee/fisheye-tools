use nalgebra::{Point2, Point3};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use yaml_rust::YamlLoader;

use super::{CameraModel, Intrinsics, validation};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KannalaBrandtModel {
    pub intrinsics: Intrinsics,
    pub distortion: Vec<f64>, // k1, k2, k3, k4
}

impl CameraModel for KannalaBrandtModel {
    fn project(&self, point_3d: &Point3<f64>) -> Result<Point2<f64>, Box<dyn Error>> {
        // TODO: Implement Kannala-Brandt projection
        unimplemented!()
    }

    fn unproject(&self, point_2d: &Point2<f64>) -> Result<Point3<f64>, Box<dyn Error>> {
        // TODO: Implement Kannala-Brandt unprojection
        unimplemented!()
    }

    fn load_from_yaml(path: &str) -> Result<Self, Box<dyn Error>> {
        let contents = fs::read_to_string(path)?;
        let docs = YamlLoader::load_from_str(&contents)?;
        // TODO: Parse YAML and create KannalaBrandtModel
        unimplemented!()
    }

    fn validate_params(&self) -> Result<(), Box<dyn Error>> {
        validation::validate_intrinsics(&self.intrinsics)?;
        if self.distortion.len() != 4 {
            return Err("Kannala-Brandt model requires 4 distortion parameters".into());
        }
        Ok(())
    }
}