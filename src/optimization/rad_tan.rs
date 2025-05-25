use crate::camera::{CameraModel, CameraModelError, RadTanModel};
use crate::optimization::Optimizer;
// use argmin::{
//     core::{
//         observers::ObserverMode, CostFunction, Error as ArgminError, Executor, Gradient, Hessian,
//         Jacobian, Operator, State,
//     },
//     solver::gaussnewton::GaussNewton,
// };
// use argmin_observer_slog::SlogLogger;
use log::info;
use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3xX, Vector2, Vector3}; // If logging is needed

#[derive(Clone)]
pub struct RadTanOptimizationCost {
    model: RadTanModel,
    points3d: Matrix3xX<f64>,
    points2d: Matrix2xX<f64>,
}

impl RadTanOptimizationCost {
    pub fn new(model: RadTanModel, points3d: Matrix3xX<f64>, points2d: Matrix2xX<f64>) -> Self {
        assert_eq!(points3d.ncols(), points2d.ncols());
        RadTanOptimizationCost {
            model,
            points3d,
            points2d,
        }
    }
}
impl Optimizer for RadTanOptimizationCost {
    fn optimize(&mut self, verbose: bool) -> Result<(), CameraModelError> {
        Ok(())
    }

    fn linear_estimation(&mut self) -> Result<(), CameraModelError>
    where
        Self: Sized,
    {
        // Duplicating implementation from CameraModel trait
        if self.points2d.ncols() != self.points3d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }

        let n_points = self.points2d.ncols();
        let mut a = nalgebra::DMatrix::zeros(n_points * 2, 3);
        let mut b = nalgebra::DVector::zeros(n_points * 2);

        for i in 0..n_points {
            let x = self.points3d[(0, i)];
            let y = self.points3d[(1, i)];
            let z = self.points3d[(2, i)];
            let u = self.points2d[(0, i)];
            let v = self.points2d[(1, i)];

            let x_prime = x / z;
            let y_prime = y / z;

            let r2 = x_prime * x_prime + y_prime * y_prime;
            let r4 = r2 * r2;
            let r6 = r4 * r2;

            a[(i * 2, 0)] = r2;
            a[(i * 2, 1)] = r4;
            a[(i * 2, 2)] = r6;
            a[(i * 2 + 1, 0)] = r2;
            a[(i * 2 + 1, 1)] = r4;
            a[(i * 2 + 1, 2)] = r6;

            b[i * 2] = (u - self.model.intrinsics.cx) / (self.model.intrinsics.fx * x_prime) - 1.0;
            b[i * 2 + 1] =
                (v - self.model.intrinsics.cy) / (self.model.intrinsics.fy * y_prime) - 1.0;
        }

        let svd = a.svd(true, true);
        let x_coeffs = match svd.solve(&b, 1e-10) {
            Ok(sol) => sol,
            Err(err_msg) => {
                return Err(CameraModelError::NumericalError(err_msg.to_string()));
            }
        };

        // distortion = [0.0; 5];
        // distortion[0] = x_coeffs[0];
        // distortion[1] = x_coeffs[1];
        // distortion[4] = x_coeffs[2];
        // distortion[2] = 0.0;
        // distortion[3] = 0.0;
        self.model.distortion = [x_coeffs[0], x_coeffs[0], x_coeffs[0], 0.0, 0.0]; // Update the model with the estimated distortion coefficients

        // let model = RadTanModel {
        //     intrinsics: self.model.intrinsics.clone(),
        //     resolution: self.model.resolution.clone(),
        //     distortion,
        // };
        self.model.validate_params()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraModel, Intrinsics, RadTanModel as RTCameraModel, Resolution};
    use crate::optimization::Optimizer;
    use approx::assert_relative_eq;
    use log::info;
    use nalgebra::{DVector, Matrix2xX, Matrix3xX, Vector3};

    fn get_sample_rt_camera_model() -> RTCameraModel {
        RTCameraModel {
            intrinsics: Intrinsics {
                fx: 461.629,
                fy: 460.152,
                cx: 362.680,
                cy: 246.049,
            },
            resolution: Resolution {
                width: 752,
                height: 480,
            },
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
                if p2d.x > 0.0
                    && p2d.x < model.resolution.width as f64
                    && p2d.y > 0.0
                    && p2d.y < model.resolution.height as f64
                {
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
        (
            Matrix2xX::from_columns(&points_2d_vec),
            Matrix3xX::from_columns(&points_3d_vec),
        )
    }

    // #[test]
    // fn test_radtan_optimization_cost_apply_and_cost() {
    //     let model_camera = get_sample_rt_camera_model();
    //     let (points_2d, points_3d) = sample_points_for_rt_model(&model_camera, 5);
    //     assert!(points_3d.ncols() > 0, "Need at least one valid point for testing cost function.");

    //     let cost = RadTanOptimizationCost::new(points_3d.clone(), points_2d.clone());
    //     let params_vec = vec![
    //         model_camera.intrinsics.fx, model_camera.intrinsics.fy,
    //         model_camera.intrinsics.cx, model_camera.intrinsics.cy,
    //         model_camera.distortion[0], model_camera.distortion[1],
    //         model_camera.distortion[2], model_camera.distortion[3],
    //         model_camera.distortion[4],
    //     ];
    //     let p = DVector::from_vec(params_vec);

    //     // Test apply (residuals)
    //     let residuals_result = cost.apply(&p);
    //     assert!(residuals_result.is_ok(), "apply() failed: {:?}", residuals_result.err());
    //     let residuals = residuals_result.unwrap();
    //     assert_eq!(residuals.len(), 2 * points_3d.ncols());
    //     assert!(residuals.iter().all(|&v| v.abs() < 1e-6));

    //     // Test cost
    //     let cost_result = cost.cost(&p);
    //     assert!(cost_result.is_ok(), "cost() failed: {:?}", cost_result.err());
    //     let c = cost_result.unwrap();
    //     assert!(c >= 0.0 && c < 1e-5);
    // }

    // #[test]
    // fn test_radtan_optimization_cost_placeholders() {
    //     let model_camera = get_sample_rt_camera_model();
    //     let (points_2d, points_3d) = sample_points_for_rt_model(&model_camera, 1);
    //     assert!(points_3d.ncols() > 0);

    //     let cost = RadTanOptimizationCost::new(points_3d.clone(), points_2d.clone());
    //     let params_vec = vec![0.0; 9]; // Dummy params
    //     let p = DVector::from_vec(params_vec);

    //     assert!(matches!(cost.jacobian(&p), Err(ArgminError::NotImplemented)));
    //     assert!(matches!(cost.gradient(&p), Err(ArgminError::NotImplemented)));
    //     assert!(matches!(cost.hessian(&p), Err(ArgminError::NotImplemented)));
    // }

    // #[test]
    // fn test_radtan_optimize_trait_method_call() {
    //     let reference_model = get_sample_rt_camera_model();
    //     let (points_2d, points_3d) = sample_points_for_rt_model(&reference_model, 10);
    //     assert!(points_3d.ncols() > 0);

    //     let mut noisy_model = get_sample_rt_camera_model(); // Start with reference
    //     noisy_model.intrinsics.fx *= 1.05; // Add some noise

    //     // Expect optimize to return an error because Jacobian is not implemented
    //     let optimize_result = Optimizer::optimize(&mut noisy_model, &points_3d, &points_2d, false);

    //     // Argmin's GaussNewton solver will fail during `init` if gradient (which needs jacobian) is not implemented.
    //     // The error comes from the executor.run() call.
    //     assert!(optimize_result.is_err(), "Optimize should return an error due to unimplemented Jacobian/Gradient.");
    //     if let Err(CameraModelError::NumericalError(e)) = optimize_result {
    //         assert!(e.contains("Argmin optimization failed"), "Error message should indicate Argmin failure.");
    //         // Further check for "NotImplemented" if possible, though argmin might wrap it.
    //     } else {
    //         panic!("Expected CameraModelError::NumericalError containing ArgminError, got {:?}", optimize_result);
    //     }
    // }

    // #[test]
    // fn test_radtan_linear_estimation_optimizer_trait() {
    //     let reference_model = get_sample_rt_camera_model();
    //     let (points_2d, points_3d) = sample_points_for_rt_model(&reference_model, 20);
    //     assert!(points_3d.ncols() > 2, "Need at least 3 points for RadTan linear estimation (for k1,k2,k3).");

    //     let estimated_model_result = RTCameraModel::linear_estimation(
    //         &reference_model.intrinsics,
    //         &reference_model.resolution,
    //         &points_2d,
    //         &points_3d,
    //     );

    //     assert!(estimated_model_result.is_ok(), "Linear estimation failed: {:?}", estimated_model_result.err());
    //     let estimated_model = estimated_model_result.unwrap();

    //     // Linear estimation for RadTan usually estimates k1, k2, k3. p1, p2 are often set to 0.
    //     // The current implementation estimates k1, k2, k3 and sets p1, p2 to 0.
    //     assert_relative_eq!(estimated_model.distortion[0], reference_model.distortion[0], epsilon = 0.1); // k1
    //     assert_relative_eq!(estimated_model.distortion[1], reference_model.distortion[1], epsilon = 0.1); // k2
    //     assert_relative_eq!(estimated_model.distortion[4], reference_model.distortion[4], epsilon = 0.1); // k3
    //     assert_relative_eq!(estimated_model.distortion[2], 0.0, epsilon = 1e-9); // p1
    //     assert_relative_eq!(estimated_model.distortion[3], 0.0, epsilon = 1e-9); // p2

    //     // Intrinsics should remain the same
    //     assert_relative_eq!(estimated_model.intrinsics.fx, reference_model.intrinsics.fx, epsilon = 1e-9);
    // }
}
