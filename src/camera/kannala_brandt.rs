use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3xX, Vector2, Vector3};
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

impl KannalaBrandtModel {
    #[allow(dead_code)]
    fn new(parameters: &DVector<f64>) -> Result<Self, CameraModelError> {
        let model = KannalaBrandtModel {
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
            coefficients: [parameters[4], parameters[5], parameters[6], parameters[7]],
        };

        model.validate_params()?;
        Ok(model)
    }

    fn check_projection_condition(&self, z: f64) -> bool {
        z > 0.0
    }
}

impl CameraModel for KannalaBrandtModel {
    fn project(
        &self,
        point_3d: &Vector3<f64>,
        compute_jacobian: bool,
    ) -> Result<(Vector2<f64>, Option<DMatrix<f64>>), CameraModelError> {
        if point_3d.z < f64::EPSILON.sqrt() {
            return Err(CameraModelError::PointAtCameraCenter);
        }

        let x = point_3d.x;
        let y = point_3d.y;
        let z = point_3d.z;

        if !self.check_projection_condition(z) {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let k1 = self.coefficients[0];
        let k2 = self.coefficients[1];
        let k3 = self.coefficients[2];
        let k4 = self.coefficients[3];

        let fx = self.intrinsics.fx;
        let fy = self.intrinsics.fy;
        let cx = self.intrinsics.cx;
        let cy = self.intrinsics.cy;

        let r = (x * x + y * y).sqrt();
        let theta = r.atan2(z);

        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        let theta_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9;

        let epsilon = 1e-6; // Epsilon for r in projection
        let (x_r, y_r) = if r < epsilon {
            (0.0, 0.0)
        } else {
            (x / r, y / r)
        };

        let proj_x = fx * theta_d * x_r + cx;
        let proj_y = fy * theta_d * y_r + cy;
        let point_2d = Vector2::new(proj_x, proj_y);

        let mut jacobian_option: Option<DMatrix<f64>> = None;

        if compute_jacobian {
            let mut jacobian = DMatrix::zeros(2, 8);

            let jac_epsilon = 1e-9; // Epsilon for r in Jacobian x_r, y_r calculation
            let (jac_x_r, jac_y_r) = if r < jac_epsilon {
                (0.0, 0.0)
            } else {
                (x / r, y / r)
            };

            // Column 0: dfx
            jacobian[(0, 0)] = theta_d * jac_x_r;
            jacobian[(1, 0)] = 0.0;

            // Column 1: dfy
            jacobian[(0, 1)] = 0.0;
            jacobian[(1, 1)] = theta_d * jac_y_r;

            // Column 2: dcx
            jacobian[(0, 2)] = 1.0;
            jacobian[(1, 2)] = 0.0;

            // Column 3: dcy
            jacobian[(0, 3)] = 0.0;
            jacobian[(1, 3)] = 1.0;

            // Columns 4-7: dk1, dk2, dk3, dk4
            // de_dp * dp_dd_theta = [fx * jac_x_r, fy * jac_y_r] (2x1 vector)
            // dd_theta_dks = [theta3, theta5, theta7, theta9] (1x4 row vector)
            // Resulting 2x4 matrix:
            // [fx * jac_x_r * theta3, fx * jac_x_r * theta5, fx * jac_x_r * theta7, fx * jac_x_r * theta9]
            // [fy * jac_y_r * theta3, fy * jac_y_r * theta5, fy * jac_y_r * theta7, fy * jac_y_r * theta9]

            let fx_x_r = fx * jac_x_r;
            let fy_y_r = fy * jac_y_r;

            jacobian[(0, 4)] = fx_x_r * theta3;
            jacobian[(1, 4)] = fy_y_r * theta3;

            jacobian[(0, 5)] = fx_x_r * theta5;
            jacobian[(1, 5)] = fy_y_r * theta5;

            jacobian[(0, 6)] = fx_x_r * theta7;
            jacobian[(1, 6)] = fy_y_r * theta7;

            jacobian[(0, 7)] = fx_x_r * theta9;
            jacobian[(1, 7)] = fy_y_r * theta9;

            jacobian_option = Some(jacobian);
        }

        Ok((point_2d, jacobian_option))
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

    fn linear_estimation(
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
