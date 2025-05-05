use nalgebra::{Matrix2xX, Matrix3xX, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use std::fs;
use yaml_rust::YamlLoader;

use crate::camera::{validation, CameraModel, CameraModelError, Intrinsics, Resolution};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KannalaBrandtModel {
    pub intrinsics: Intrinsics,
    pub resolution: Resolution,
    pub coefficients: [f64; 4], // k1, k2, k3, k4
}

impl CameraModel for KannalaBrandtModel {
    fn project(&self, point_3d: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        if point_3d.z < f64::EPSILON.sqrt() {
            return Err(CameraModelError::PointAtCameraCenter);
        }

        // To be implemented
        Err(CameraModelError::InvalidParams(
            "Not implemented yet".to_string(),
        ))
    }

    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        if point_2d.x < 0.0
            || point_2d.x >= self.resolution.width as f64
            || point_2d.y < 0.0
            || point_2d.y >= self.resolution.height as f64
        {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        // To be implemented
        Err(CameraModelError::InvalidParams(
            "Not implemented yet".to_string(),
        ))
    }

    fn load_from_yaml(path: &str) -> Result<Self, CameraModelError> {
        // Basic implementation that assumes a similar format to other camera models
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

        let model = KannalaBrandtModel {
            intrinsics,
            resolution,
            coefficients: [0.0, 0.0, 0.0, 0.0], // Default coefficients
        };

        // Validate parameters
        model.validate_params()?;

        Ok(model)
    }

    fn save_to_yaml(&self, _path: &str) -> Result<(), CameraModelError> {
        // Implementation for saving to YAML
        Ok(())
    }

    fn validate_params(&self) -> Result<(), CameraModelError> {
        validation::validate_intrinsics(&self.intrinsics)?;
        // Additional validations to be implemented
        Ok(())
    }

    fn get_resolution(&self) -> Resolution {
        self.resolution.clone()
    }

    fn get_intrinsics(&self) -> Intrinsics {
        self.intrinsics.clone()
    }

    fn get_distortion(&self) -> Vec<f64> {
        self.coefficients.to_vec()
    }

    fn initialize(
        intrinsics: &Intrinsics,
        resolution: &Resolution,
        _points_2d: &Matrix2xX<f64>,
        _points_3d: &Matrix3xX<f64>,
    ) -> Result<Self, CameraModelError> {
        let model = KannalaBrandtModel {
            intrinsics: intrinsics.clone(),
            resolution: resolution.clone(),
            coefficients: [0.0, 0.0, 0.0, 0.0],
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
