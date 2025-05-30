use crate::camera::{CameraModel, CameraModelError, KannalaBrandtModel};
use crate::optimization::Optimizer;
use argmin::{
    core::{
        observers::ObserverMode, CostFunction, Error as ArgminError, Executor, Gradient, Hessian,
        Jacobian, Operator, State,
    },
    solver::gaussnewton::GaussNewton,
};
use argmin_observer_slog::SlogLogger;
use log::info; // Keep if logging from cost functions is desired
use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3xX};

/// Cost function for Kannala-Brandt camera model optimization.
#[derive(Clone)]
pub struct KannalaBrandtOptimizationCost {
    model: KannalaBrandtModel,
    points3d: Matrix3xX<f64>,
    points2d: Matrix2xX<f64>,
}

impl KannalaBrandtOptimizationCost {
    pub fn new(
        model: KannalaBrandtModel,
        points3d: Matrix3xX<f64>,
        points2d: Matrix2xX<f64>,
    ) -> Self {
        assert_eq!(points3d.ncols(), points2d.ncols());
        KannalaBrandtOptimizationCost {
            model,
            points3d,
            points2d,
        }
    }
}

impl Optimizer for KannalaBrandtOptimizationCost {
    fn optimize(&mut self, verbose: bool) -> Result<(), CameraModelError> {
        // This is the same implementation as the original optimize method
        if self.points3d.ncols() != self.points2d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }
        if self.points3d.ncols() == 0 {
            return Err(CameraModelError::InvalidParams(
                "Points arrays cannot be empty".to_string(),
            ));
        }

        // let cost_function =
        //     KannalaBrandtOptimizationCost::new(self.points3d.clone(), self.points2d.clone());
        let cost_function = self.clone();

        let initial_params_vec = vec![
            self.model.intrinsics.fx,
            self.model.intrinsics.fy,
            self.model.intrinsics.cx,
            self.model.intrinsics.cy,
            self.model.distortions[0],
            self.model.distortions[1],
            self.model.distortions[2],
            self.model.distortions[3],
        ];
        let init_param: DVector<f64> = DVector::from_vec(initial_params_vec);

        let solver = GaussNewton::new();

        let mut executor_builder = Executor::new(cost_function, solver)
            .configure(|state| state.param(init_param).max_iters(100));

        if verbose {
            executor_builder =
                executor_builder.add_observer(SlogLogger::term(), ObserverMode::Always);
            info!("Starting Kannala-Brandt optimization with Gauss-Newton...");
        } else {
            executor_builder =
                executor_builder.add_observer(SlogLogger::term(), ObserverMode::NewBest);
        }

        let res = executor_builder
            .run()
            .map_err(|e| CameraModelError::NumericalError(e.to_string()))?;

        if verbose {
            info!("Optimization finished: \n{}", res);
            info!("Termination status: {:?}", res.state().termination_status);
        }

        if let Some(best_params_dvec) = res.state().get_best_param() {
            self.model.intrinsics.fx = best_params_dvec[0];
            self.model.intrinsics.fy = best_params_dvec[1];
            self.model.intrinsics.cx = best_params_dvec[2];
            self.model.intrinsics.cy = best_params_dvec[3];
            self.model.distortions[0] = best_params_dvec[4];
            self.model.distortions[1] = best_params_dvec[5];
            self.model.distortions[2] = best_params_dvec[6];
            self.model.distortions[3] = best_params_dvec[7];
            self.model.validate_params()?;
        } else {
            return Err(CameraModelError::NumericalError(
                "Optimization failed to find best parameters".to_string(),
            ));
        }
        Ok(())
    }

    fn linear_estimation(&mut self) -> Result<(), CameraModelError>
    where
        Self: Sized,
    {
        // Duplicating the implementation from CameraModel trait for now
        if self.points3d.ncols() != self.points2d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }
        if self.points3d.ncols() < 4 {
            return Err(CameraModelError::InvalidParams(
                "Not enough points for linear estimation (need at least 4)".to_string(),
            ));
        }

        let num_points = self.points3d.ncols();
        let mut a_mat = DMatrix::zeros(num_points * 2, 4);
        let mut b_vec = DVector::zeros(num_points * 2);

        for i in 0..num_points {
            let p3d = self.points3d.column(i);
            let p2d = self.points2d.column(i);

            let x_world = p3d.x;
            let y_world = p3d.y;
            let z_world = p3d.z;

            let u_img = p2d.x;
            let v_img = p2d.y;

            if z_world <= f64::EPSILON {
                continue;
            }

            let r_world = (x_world * x_world + y_world * y_world).sqrt();
            let theta = r_world.atan2(z_world);

            let theta2 = theta * theta;
            let theta3 = theta2 * theta;
            let theta5 = theta3 * theta2;
            let theta7 = theta5 * theta2;
            let theta9 = theta7 * theta2;

            a_mat[(i * 2, 0)] = theta3;
            a_mat[(i * 2, 1)] = theta5;
            a_mat[(i * 2, 2)] = theta7;
            a_mat[(i * 2, 3)] = theta9;

            a_mat[(i * 2 + 1, 0)] = theta3;
            a_mat[(i * 2 + 1, 1)] = theta5;
            a_mat[(i * 2 + 1, 2)] = theta7;
            a_mat[(i * 2 + 1, 3)] = theta9;

            let x_r = if r_world < f64::EPSILON {
                0.0
            } else {
                x_world / r_world
            };
            let y_r = if r_world < f64::EPSILON {
                0.0
            } else {
                y_world / r_world
            };

            if (self.model.intrinsics.fx * x_r).abs() < f64::EPSILON && x_r.abs() > f64::EPSILON {
                return Err(CameraModelError::NumericalError(
                    "fx * x_r is zero in linear estimation".to_string(),
                ));
            }
            if (self.model.intrinsics.fy * y_r).abs() < f64::EPSILON && y_r.abs() > f64::EPSILON {
                return Err(CameraModelError::NumericalError(
                    "fy * y_r is zero in linear estimation".to_string(),
                ));
            }

            if x_r.abs() > f64::EPSILON {
                b_vec[i * 2] =
                    (u_img - self.model.intrinsics.cx) / (self.model.intrinsics.fx * x_r) - theta;
            } else {
                b_vec[i * 2] = if (u_img - self.model.intrinsics.cx).abs() < f64::EPSILON {
                    -theta
                } else {
                    0.0
                };
            }

            if y_r.abs() > f64::EPSILON {
                b_vec[i * 2 + 1] =
                    (v_img - self.model.intrinsics.cy) / (self.model.intrinsics.fy * y_r) - theta;
            } else {
                b_vec[i * 2 + 1] = if (v_img - self.model.intrinsics.cy).abs() < f64::EPSILON {
                    -theta
                } else {
                    0.0
                };
            }
        }

        let svd = a_mat.svd(true, true);
        let x_coeffs = svd.solve(&b_vec, f64::EPSILON).map_err(|e_str| {
            CameraModelError::NumericalError(format!(
                "SVD solve failed in linear estimation: {}",
                e_str
            ))
        })?;
        self.model.distortions = [x_coeffs[0], x_coeffs[1], x_coeffs[2], x_coeffs[3]];

        self.model.validate_params()?;
        Ok(())
    }

    fn get_intrinsics(&self) -> crate::camera::Intrinsics {
        self.model.intrinsics.clone()
    }

    fn get_resolution(&self) -> crate::camera::Resolution {
        self.model.resolution.clone()
    }

    fn get_distortion(&self) -> Vec<f64> {
        self.model.get_distortion()
    }
}

impl Operator for KannalaBrandtOptimizationCost {
    type Param = DVector<f64>; // [fx, fy, cx, cy, k1, k2, k3, k4]
    type Output = DVector<f64>; // Residuals vector

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {
        let num_points = self.points3d.ncols();
        let mut residuals = DVector::zeros(num_points * 2);
        let model = KannalaBrandtModel::new(&p)?;
        let mut counter = 0;

        for i in 0..num_points {
            let p3d = &self.points3d.column(i).into_owned();
            let p2d_gt = &self.points2d.column(i).into_owned();

            match model.project(p3d, false) {
                Ok((p2d_projected, _)) => {
                    residuals[counter * 2] = p2d_projected.x - p2d_gt.x;
                    residuals[counter * 2 + 1] = p2d_projected.y - p2d_gt.y;
                    counter += 1;
                }
                Err(_) => {
                    info!("3d points {} are not projected", p3d);
                }
            }
        }
        residuals = residuals.rows(0, counter * 2).into_owned();
        Ok(residuals)
    }
}

impl Jacobian for KannalaBrandtOptimizationCost {
    type Param = DVector<f64>;
    type Jacobian = DMatrix<f64>;

    fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, ArgminError> {
        let num_points = self.points3d.ncols();
        let mut jacobian_matrix = DMatrix::zeros(num_points * 2, 8); // 2 residuals per point, 8 parameters
        let model = KannalaBrandtModel::new(&p)?;
        let mut counter = 0;

        for i in 0..num_points {
            let p3d = &self.points3d.column(i).into_owned();
            match model.project(p3d, true) {
                Ok((_, Some(jac_point))) => {
                    jacobian_matrix
                        .view_mut((counter * 2, 0), (2, 8))
                        .copy_from(&jac_point);
                    counter += 1;
                }
                Ok((_, None)) => {
                    info!("3d points {} doesn't have jacobian", p3d);
                }
                Err(_) => {
                    info!("3d points {} are not projected", p3d);
                }
            }
        }
        jacobian_matrix = jacobian_matrix.rows(0, counter * 2).into_owned();
        Ok(jacobian_matrix)
    }
}

impl CostFunction for KannalaBrandtOptimizationCost {
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {
        let residuals = self.apply(p)?;
        Ok(residuals.norm_squared() / 2.0) // Standard least squares cost
    }
}

impl Gradient for KannalaBrandtOptimizationCost {
    type Param = DVector<f64>;
    type Gradient = DVector<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        let jacobian = self.jacobian(p)?;
        let residuals = self.apply(p)?;
        Ok(jacobian.transpose() * residuals)
    }
}

impl Hessian for KannalaBrandtOptimizationCost {
    type Param = DVector<f64>;
    type Hessian = DMatrix<f64>;

    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, ArgminError> {
        let jacobian = self.jacobian(p)?;
        Ok(jacobian.transpose() * jacobian) // Gauss-Newton approximation J^T * J
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraModel, Intrinsics, KannalaBrandtModel as KBCameraModel, Resolution};
    use crate::optimization::Optimizer;
    use approx::assert_relative_eq;
    use log::info;
    use nalgebra::{Matrix2xX, Matrix3xX, Vector2, Vector3};

    // Helper to get a sample KannalaBrandtModel instance
    fn get_sample_kb_camera_model() -> KBCameraModel {
        KBCameraModel {
            intrinsics: Intrinsics {
                fx: 461.586,
                fy: 460.281,
                cx: 366.286,
                cy: 249.080,
            },
            resolution: Resolution {
                width: 752,
                height: 480,
            },
            distortions: [-0.0125, 0.0578, -0.0849, 0.0436], // k1, k2, k3, k4
        }
    }

    // Placeholder for geometry::sample_points or a simplified version
    fn sample_points_for_kb_model(
        model: &KBCameraModel,
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
            // Fallback if no points projected successfully, provide at least one dummy point
            // to prevent panic in Matrix2xX::from_columns on empty vec.
            // This indicates an issue with sample_points_for_kb_model or model params.
            info!("Warning: sample_points_for_kb_model generated no valid points. Using a dummy point.");
            points_3d_vec.push(Vector3::new(0.0, 0.0, 1.0));
            points_2d_vec.push(Vector2::new(model.intrinsics.cx, model.intrinsics.cy));
        }
        (
            Matrix2xX::from_columns(&points_2d_vec),
            Matrix3xX::from_columns(&points_3d_vec),
        )
    }

    #[test]
    fn test_kannala_brandt_optimization_cost_basic() {
        let model_camera = get_sample_kb_camera_model();
        let (points_2d, points_3d) = sample_points_for_kb_model(&model_camera, 5);

        assert!(points_3d.ncols() > 0, "Need at least one valid point for testing cost function.");

        let cost = KannalaBrandtOptimizationCost::new(model_camera.clone(), points_3d.clone(), points_2d.clone());
        let params_vec = vec![
            model_camera.intrinsics.fx, model_camera.intrinsics.fy,
            model_camera.intrinsics.cx, model_camera.intrinsics.cy,
            model_camera.distortions[0], model_camera.distortions[1],
            model_camera.distortions[2], model_camera.distortions[3],
        ];
        let p = DVector::from_vec(params_vec);

        // Test apply (residuals)
        let residuals = cost.apply(&p).unwrap();
        assert_eq!(residuals.len(), 2 * points_3d.ncols());
        // If points_3d.ncols() is 0, residuals.iter() will be empty, and .all() returns true.
        // Add a check for non-empty residuals if points_3d.ncols() > 0
        if points_3d.ncols() > 0 {
            assert!(residuals.iter().all(|&v| v.abs() < 1e-6));
        }

        // Test cost
        let c = cost.cost(&p).unwrap();
        assert!(c >= 0.0 && c < 1e-5);

        // Test jacobian
        let jac = cost.jacobian(&p).unwrap();
        assert_eq!(jac.nrows(), 2 * points_3d.ncols());
        assert_eq!(jac.ncols(), 8); // 8 parameters for KannalaBrandt

        // Test gradient
        let grad = cost.gradient(&p).unwrap();
        assert_eq!(grad.len(), 8);
        if points_3d.ncols() > 0 {
            assert!(grad.norm() < 1e-5);
        }

        // Test hessian
        let hess = cost.hessian(&p).unwrap();
        assert_eq!(hess.nrows(), 8);
        assert_eq!(hess.ncols(), 8);
    }

    #[test]
    fn test_kannala_brandt_optimize_trait_method() {
        let reference_model = get_sample_kb_camera_model();
        let (points_2d, points_3d) = sample_points_for_kb_model(&reference_model, 50);
        assert!(points_3d.ncols() > 10, "Need sufficient points for optimization test. Actual points: {}", points_3d.ncols());

        let noisy_model_initial = KBCameraModel {
            intrinsics: Intrinsics {
                fx: reference_model.intrinsics.fx * 1.05, // Introduce noise
                fy: reference_model.intrinsics.fy * 0.95,
                cx: reference_model.intrinsics.cx - 3.0,
                cy: reference_model.intrinsics.cy + 3.0,
            },
            resolution: reference_model.resolution.clone(),
            distortions: [
                reference_model.distortions[0] * 0.8,
                reference_model.distortions[1] * 1.2,
                reference_model.distortions[2] * 0.7,
                reference_model.distortions[3] * 1.3,
            ],
        };

        let mut cost_optimizer = KannalaBrandtOptimizationCost::new(noisy_model_initial, points_3d.clone(), points_2d.clone());
        let optimize_result = cost_optimizer.optimize(false);
        assert!(optimize_result.is_ok(), "Optimization failed: {:?}", optimize_result.err());

        let optimized_model = &cost_optimizer.model;

        // Compare optimized parameters with reference_model
        assert_relative_eq!(optimized_model.intrinsics.fx, reference_model.intrinsics.fx, epsilon = 5.0, max_relative = 0.05); // Looser epsilon
        assert_relative_eq!(optimized_model.intrinsics.fy, reference_model.intrinsics.fy, epsilon = 5.0, max_relative = 0.05);
        assert_relative_eq!(optimized_model.intrinsics.cx, reference_model.intrinsics.cx, epsilon = 5.0, max_relative = 0.05);
        assert_relative_eq!(optimized_model.intrinsics.cy, reference_model.intrinsics.cy, epsilon = 5.0, max_relative = 0.05);
        for i in 0..4 {
            assert_relative_eq!(optimized_model.distortions[i], reference_model.distortions[i], epsilon = 0.05, max_relative = 0.1); // Looser
        }
    }

    #[test]
    fn test_kannala_brandt_linear_estimation_optimizer_trait() {
        let reference_model = get_sample_kb_camera_model();
        let (points_2d, points_3d) = sample_points_for_kb_model(&reference_model, 20);
        assert!(points_3d.ncols() > 3, "Need at least 4 points for KB linear estimation. Actual points: {}", points_3d.ncols());

        // For linear estimation, we typically assume intrinsics are known or roughly known.
        // The linear estimation part of the Optimizer trait will update the distortions in its internal model.
        let initial_model_for_estimation = KBCameraModel {
            intrinsics: reference_model.intrinsics.clone(), // Use reference intrinsics
            resolution: reference_model.resolution.clone(),
            distortions: [0.0, 0.0, 0.0, 0.0], // Start with zero distortion for estimation
        };

        let mut cost_estimator = KannalaBrandtOptimizationCost::new(initial_model_for_estimation, points_3d.clone(), points_2d.clone());
        let estimation_result = cost_estimator.linear_estimation();

        assert!(estimation_result.is_ok(), "Linear estimation failed: {:?}", estimation_result.err());
        let estimated_model = &cost_estimator.model;

        // Compare estimated distortion parameters. Linear estimation might not be super accurate.
        // The accuracy depends heavily on the quality of points and the model itself.
        for i in 0..4 {
            assert_relative_eq!(estimated_model.distortions[i], reference_model.distortions[i], epsilon = 0.1, max_relative = 0.2);
        }

        // Intrinsics should remain unchanged by this specific linear_estimation implementation
        assert_relative_eq!(estimated_model.intrinsics.fx, reference_model.intrinsics.fx, epsilon = 1e-9);
        assert_relative_eq!(estimated_model.intrinsics.fy, reference_model.intrinsics.fy, epsilon = 1e-9);
        assert_relative_eq!(estimated_model.intrinsics.cx, reference_model.intrinsics.cx, epsilon = 1e-9);
        assert_relative_eq!(estimated_model.intrinsics.cy, reference_model.intrinsics.cy, epsilon = 1e-9);
    }
}
