// src/optimization/rad_tan.rs

use crate::camera::rad_tan::RadTanModel;
use crate::camera::CameraModel; // For RadTanModel::project
use crate::camera::CameraModelError; // Error type for RadTanModel::new
use crate::optimization::OptimizationCost;

use argmin::core::{CostFunction, Error as ArgminError, Gradient, Hessian, Jacobian, Operator};
use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3xX};
use log::info; // If logging is needed

#[derive(Clone)]
pub struct RadTanOptimizationCost {
    points3d: Matrix3xX<f64>,
    points2d: Matrix2xX<f64>,
}

impl RadTanOptimizationCost {
    pub fn new(points3d: Matrix3xX<f64>, points2d: Matrix2xX<f64>) -> Self {
        assert_eq!(points3d.ncols(), points2d.ncols());
        RadTanOptimizationCost { points3d, points2d }
    }

    // Helper to create model and map error, similar to other models
    fn get_model_from_params(p: &DVector<f64>) -> Result<RadTanModel, ArgminError> {
        RadTanModel::new(p).map_err(|e: CameraModelError| {
            let boxed_err: Box<dyn std::error::Error + Send + Sync> = Box::new(e);
            ArgminError::Consolidation("Failed to create RadTanModel from params".to_string(), boxed_err)
        })
    }
}

impl OptimizationCost for RadTanOptimizationCost {
    type Param = DVector<f64>;    // fx, fy, cx, cy, k1, k2, p1, p2, k3 (9 params)
    type Output = DVector<f64>;   // Residuals vector for Operator
    type Jacobian = DMatrix<f64>; // Jacobian matrix
    type Hessian = DMatrix<f64>;  // Hessian matrix (J^T J)
}

impl Operator for RadTanOptimizationCost {
    type Param = DVector<f64>;
    type Output = DVector<f64>;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {
        if p.len() != 9 {
            return Err(ArgminError::InvalidParameter{
                text: format!("Parameter vector length must be 9, got {}", p.len())
            });
        }
        let num_points = self.points3d.ncols();
        let mut residuals = DVector::zeros(num_points * 2);
        let model = Self::get_model_from_params(p)?;
        let mut counter = 0;

        for i in 0..num_points {
            let p3d_col = self.points3d.column(i);
            let p2d_gt_col = self.points2d.column(i);
            
            // nalgebra columns are views, convert to owned Vector3/2 for project method if it expects owned.
            // Assuming project takes references to Vector3/2 that can be formed from columns directly.
            // If RadTanModel::project expects owned VectorN<f64, U3>, then .into_owned() is needed.
            // Based on other models, project takes &Vector3<f64>.
            let p3d = p3d_col.into_owned();

            match model.project(&p3d, false) { // Pass false for compute_jacobian
                Ok((p2d_projected, _)) => {
                    residuals[counter * 2] = p2d_projected.x - p2d_gt_col.x;
                    residuals[counter * 2 + 1] = p2d_projected.y - p2d_gt_col.y;
                    counter += 1;
                }
                Err(e) => {
                    // Optionally log or handle error for specific point projection
                    info!("Point projection failed for point {}: {:?}", i, e);
                }
            }
        }
        residuals = residuals.rows(0, counter * 2).into_owned();
        Ok(residuals)
    }
}

impl Jacobian for RadTanOptimizationCost {
    type Param = DVector<f64>;
    type Jacobian = DMatrix<f64>;

    fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, ArgminError> {
        if p.len() != 9 {
             return Err(ArgminError::InvalidParameter{
                text: format!("Parameter vector length must be 9, got {}", p.len())
            });
        }
        // Placeholder implementation
        // let num_residuals = self.points3d.ncols() * 2;
        // Ok(DMatrix::zeros(num_residuals, 9)) 
        Err(ArgminError::NotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraModel, Intrinsics, Resolution, RadTanModel as RTCameraModel};
    use crate::optimization::Optimizer;
    use nalgebra::{Matrix2xX, Matrix3xX, Vector3, DVector};
    use approx::assert_relative_eq;
    use log::info;

    fn get_sample_rt_camera_model() -> RTCameraModel {
        RTCameraModel {
            intrinsics: Intrinsics {
                fx: 461.629, fy: 460.152, cx: 362.680, cy: 246.049,
            },
            resolution: Resolution { width: 752, height: 480 },
            distortion: [-0.2834, 0.0739, 0.0001, 1.7618e-05, 0.0], // k1,k2,p1,p2,k3
        }
    }

    fn sample_points_for_rt_model(
        model: &RTCameraModel,
        num_points: usize,
    ) -> (Matrix2xX<f64>, Matrix3xX<f64>) {
        let mut points_2d_vec = Vec::new();
        let mut points_3d_vec = Vec::new();

        for i in 0..num_points {
            let x = (i as f64 * 0.1) - (num_points as f64 * 0.05);
            let y = (i as f64 * 0.05) - (num_points as f64 * 0.025);
            let z = 1.0 + (i as f64 * 0.01);
            let p3d = Vector3::new(x, y, z);

            if let Ok((p2d, _)) = model.project(&p3d, false) {
                 if p2d.x > 0.0 && p2d.x < model.resolution.width as f64 && 
                    p2d.y > 0.0 && p2d.y < model.resolution.height as f64 {
                    points_3d_vec.push(p3d);
                    points_2d_vec.push(p2d);
                }
            }
        }
         if points_2d_vec.is_empty() && num_points > 0 {
             info!("Warning: sample_points_for_rt_model generated no valid points. Using a dummy point.");
             points_3d_vec.push(Vector3::new(0.0, 0.0, 1.0));
             points_2d_vec.push(Vector2::new(model.intrinsics.cx, model.intrinsics.cy));
        }
        (Matrix2xX::from_columns(&points_2d_vec), Matrix3xX::from_columns(&points_3d_vec))
    }

    #[test]
    fn test_radtan_optimization_cost_apply_and_cost() {
        let model_camera = get_sample_rt_camera_model();
        let (points_2d, points_3d) = sample_points_for_rt_model(&model_camera, 5);
        assert!(points_3d.ncols() > 0, "Need at least one valid point for testing cost function.");

        let cost = RadTanOptimizationCost::new(points_3d.clone(), points_2d.clone());
        let params_vec = vec![
            model_camera.intrinsics.fx, model_camera.intrinsics.fy,
            model_camera.intrinsics.cx, model_camera.intrinsics.cy,
            model_camera.distortion[0], model_camera.distortion[1],
            model_camera.distortion[2], model_camera.distortion[3],
            model_camera.distortion[4],
        ];
        let p = DVector::from_vec(params_vec);

        // Test apply (residuals)
        let residuals_result = cost.apply(&p);
        assert!(residuals_result.is_ok(), "apply() failed: {:?}", residuals_result.err());
        let residuals = residuals_result.unwrap();
        assert_eq!(residuals.len(), 2 * points_3d.ncols());
        assert!(residuals.iter().all(|&v| v.abs() < 1e-6));

        // Test cost
        let cost_result = cost.cost(&p);
        assert!(cost_result.is_ok(), "cost() failed: {:?}", cost_result.err());
        let c = cost_result.unwrap();
        assert!(c >= 0.0 && c < 1e-5);
    }

    #[test]
    fn test_radtan_optimization_cost_placeholders() {
        let model_camera = get_sample_rt_camera_model();
        let (points_2d, points_3d) = sample_points_for_rt_model(&model_camera, 1);
        assert!(points_3d.ncols() > 0);
        
        let cost = RadTanOptimizationCost::new(points_3d.clone(), points_2d.clone());
        let params_vec = vec![0.0; 9]; // Dummy params
        let p = DVector::from_vec(params_vec);

        assert!(matches!(cost.jacobian(&p), Err(ArgminError::NotImplemented)));
        assert!(matches!(cost.gradient(&p), Err(ArgminError::NotImplemented)));
        assert!(matches!(cost.hessian(&p), Err(ArgminError::NotImplemented)));
    }

    #[test]
    fn test_radtan_optimize_trait_method_call() {
        let reference_model = get_sample_rt_camera_model();
        let (points_2d, points_3d) = sample_points_for_rt_model(&reference_model, 10);
        assert!(points_3d.ncols() > 0);

        let mut noisy_model = get_sample_rt_camera_model(); // Start with reference
        noisy_model.intrinsics.fx *= 1.05; // Add some noise

        // Expect optimize to return an error because Jacobian is not implemented
        let optimize_result = Optimizer::optimize(&mut noisy_model, &points_3d, &points_2d, false);
        
        // Argmin's GaussNewton solver will fail during `init` if gradient (which needs jacobian) is not implemented.
        // The error comes from the executor.run() call.
        assert!(optimize_result.is_err(), "Optimize should return an error due to unimplemented Jacobian/Gradient.");
        if let Err(CameraModelError::NumericalError(e)) = optimize_result {
            assert!(e.contains("Argmin optimization failed"), "Error message should indicate Argmin failure.");
            // Further check for "NotImplemented" if possible, though argmin might wrap it.
        } else {
            panic!("Expected CameraModelError::NumericalError containing ArgminError, got {:?}", optimize_result);
        }
    }
    
    #[test]
    fn test_radtan_linear_estimation_optimizer_trait() {
        let reference_model = get_sample_rt_camera_model();
        let (points_2d, points_3d) = sample_points_for_rt_model(&reference_model, 20);
        assert!(points_3d.ncols() > 2, "Need at least 3 points for RadTan linear estimation (for k1,k2,k3).");

        let estimated_model_result = RTCameraModel::linear_estimation(
            &reference_model.intrinsics,
            &reference_model.resolution,
            &points_2d,
            &points_3d,
        );
        
        assert!(estimated_model_result.is_ok(), "Linear estimation failed: {:?}", estimated_model_result.err());
        let estimated_model = estimated_model_result.unwrap();

        // Linear estimation for RadTan usually estimates k1, k2, k3. p1, p2 are often set to 0.
        // The current implementation estimates k1, k2, k3 and sets p1, p2 to 0.
        assert_relative_eq!(estimated_model.distortion[0], reference_model.distortion[0], epsilon = 0.1); // k1
        assert_relative_eq!(estimated_model.distortion[1], reference_model.distortion[1], epsilon = 0.1); // k2
        assert_relative_eq!(estimated_model.distortion[4], reference_model.distortion[4], epsilon = 0.1); // k3
        assert_relative_eq!(estimated_model.distortion[2], 0.0, epsilon = 1e-9); // p1
        assert_relative_eq!(estimated_model.distortion[3], 0.0, epsilon = 1e-9); // p2

        // Intrinsics should remain the same
        assert_relative_eq!(estimated_model.intrinsics.fx, reference_model.intrinsics.fx, epsilon = 1e-9);
    }
}

impl CostFunction for RadTanOptimizationCost {
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {
        if p.len() != 9 {
            return Err(ArgminError::InvalidParameter{
                text: format!("Parameter vector length must be 9, got {}", p.len())
            });
        }
        let residuals = self.apply(p)?;
        Ok(residuals.norm_squared() / 2.0) // Standard least squares cost
    }
}

impl Gradient for RadTanOptimizationCost {
    type Param = DVector<f64>;
    type Gradient = DVector<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        if p.len() != 9 {
            return Err(ArgminError::InvalidParameter{
                text: format!("Parameter vector length must be 9, got {}", p.len())
            });
        }
        // Placeholder implementation due to Jacobian placeholder
        // Ok(DVector::zeros(9))
        Err(ArgminError::NotImplemented)
    }
}

impl Hessian for RadTanOptimizationCost {
    type Param = DVector<f64>;
    type Hessian = DMatrix<f64>;

    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, ArgminError> {
         if p.len() != 9 {
            return Err(ArgminError::InvalidParameter{
                text: format!("Parameter vector length must be 9, got {}", p.len())
            });
        }
        // Placeholder implementation due to Jacobian placeholder
        // Ok(DMatrix::zeros(9,9))
        Err(ArgminError::NotImplemented)
    }
}
