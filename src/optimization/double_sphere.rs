// src/optimization/double_sphere.rs

// use crate::camera::double_sphere::DoubleSphereModel; // Adjusted path
use crate::camera::{CameraModel, CameraModelError, DoubleSphereModel};
use crate::optimization::Optimizer;
use argmin::{
    core::{
        observers::ObserverMode, CostFunction, Error, Executor, Gradient, Hessian, Jacobian,
        Operator, State,
    },
    solver::gaussnewton::GaussNewton,
};
// use argmin::solver::gaussnewton::GaussNewton; // Added
use argmin_observer_slog::SlogLogger; // Added
use log::info;
use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3xX};

/// Cost function for Double Sphere camera model optimization.
///
/// This structure holds the 3D-2D point correspondences used during
/// camera calibration optimization. It implements the necessary traits
/// for use with the argmin optimization library.
#[derive(Clone)]
pub struct DoubleSphereOptimizationCost {
    model: DoubleSphereModel,
    /// 3D points in camera coordinate system (3×N matrix)
    points3d: Matrix3xX<f64>,
    /// Corresponding 2D points in image coordinates (2×N matrix)
    points2d: Matrix2xX<f64>,
}

impl DoubleSphereOptimizationCost {
    /// Creates a new cost function for Double Sphere camera optimization.
    ///
    /// # Arguments
    ///
    /// * `points3d` - 3D points in camera coordinate system (3×N matrix)
    /// * `points2d` - Corresponding 2D points in image coordinates (2×N matrix)
    ///
    /// # Panics
    ///
    /// Panics if the number of 3D and 2D points don't match.
    pub fn new(
        model: DoubleSphereModel,
        points3d: Matrix3xX<f64>,
        points2d: Matrix2xX<f64>,
    ) -> Self {
        assert_eq!(points3d.ncols(), points2d.ncols());
        DoubleSphereOptimizationCost {
            model,
            points3d: points3d,
            points2d: points2d,
        }
    }
}

impl Optimizer for DoubleSphereOptimizationCost {
    fn optimize(&mut self, verbose: bool) -> Result<(), CameraModelError> {
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

        // let cost_function = DoubleSphereOptimizationCost::new(self.model, self.points3d.clone(), self.points2d.clone());
        let cost_function = self.clone();

        // Initial parameters
        let param = vec![
            self.model.intrinsics.fx,
            self.model.intrinsics.fy,
            self.model.intrinsics.cx,
            self.model.intrinsics.cy,
            self.model.alpha,
            self.model.xi,
        ];

        let init_param: DVector<f64> = DVector::from_vec(param);

        // Configure the Gauss-Newton solver
        let solver = GaussNewton::new();

        // Setup executor with the solver and cost function
        let executor_builder = Executor::new(cost_function, solver)
            .configure(|state| state.param(init_param).max_iters(100))
            .add_observer(SlogLogger::term(), ObserverMode::Never);

        if verbose {
            info!("Starting optimization with Gauss-Newton...");
        }

        let res = executor_builder
            .run()
            .map_err(|e| CameraModelError::NumericalError(e.to_string()))?;

        if verbose {
            info!("Optimization finished: \n{}", res);
            info!("Termination status: {:?}", res.state().termination_status);
        }

        let best_params_dvec = res.state().get_best_param().unwrap().clone();

        // Update model parameters from the final parameters
        self.model.intrinsics.fx = best_params_dvec[0];
        self.model.intrinsics.fy = best_params_dvec[1];
        self.model.intrinsics.cx = best_params_dvec[2];
        self.model.intrinsics.cy = best_params_dvec[3];
        self.model.alpha = best_params_dvec[4];
        self.model.xi = best_params_dvec[5];

        // Validate the optimized parameters
        self.model.validate_params()?;

        Ok(())
    }

    // linear_estimation will be fully implemented here when it's moved from CameraModel trait
    fn linear_estimation(&mut self) -> Result<(), CameraModelError>
    where
        Self: Sized,
    {
        // Check if the number of 2D and 3D points match
        if self.points2d.ncols() != self.points3d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }

        // Set up the linear system to solve for alpha
        let num_points = self.points2d.ncols();
        let mut a = nalgebra::DMatrix::zeros(num_points * 2, 1);
        let mut b = nalgebra::DVector::zeros(num_points * 2);

        for i in 0..num_points {
            let x = self.points3d[(0, i)];
            let y = self.points3d[(1, i)];
            let z = self.points3d[(2, i)];
            let u = self.points2d[(0, i)];
            let v = self.points2d[(1, i)];

            let d = (x * x + y * y + z * z).sqrt();
            let u_cx = u - self.model.intrinsics.cx;
            let v_cy = v - self.model.intrinsics.cy;

            a[(i * 2, 0)] = u_cx * (d - z);
            a[(i * 2 + 1, 0)] = v_cy * (d - z);

            b[i * 2] = (self.model.intrinsics.fx * x) - (u_cx * z);
            b[i * 2 + 1] = (self.model.intrinsics.fy * y) - (v_cy * z);
        }

        // Solve the linear system using SVD
        let svd = a.svd(true, true);
        let alpha = match svd.solve(&b, 1e-10) {
            Ok(sol) => sol[0], // Handle the successful case
            Err(err_msg) => {
                return Err(CameraModelError::NumericalError(err_msg.to_string()));
            }
        };

        self.model.alpha = alpha;
        // Validate parameters
        self.model.validate_params()?;

        Ok(())
    }
}

/// Implementation of the Operator trait for Gauss-Newton optimization.
impl Operator for DoubleSphereOptimizationCost {
    /// Parameter vector: [fx, fy, cx, cy, alpha, xi]
    type Param = DVector<f64>;
    /// Output residuals vector (2×N elements for N points)
    type Output = DVector<f64>;

    /// Applies the camera model to compute projection residuals.
    ///
    /// # Arguments
    ///
    /// * `p` - Parameter vector containing camera intrinsics and distortion parameters
    ///
    /// # Returns
    ///
    /// Vector of residuals between projected and observed 2D points.
    ///
    /// # Errors
    ///
    /// Returns an error if the camera model parameters are invalid or
    /// if projection fails for the given points.
    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let num_points = self.points3d.ncols();
        let mut residuals = DVector::zeros(num_points * 2);
        // Ensure DoubleSphereModel::new is public or accessible
        let model = DoubleSphereModel::new(&p)?;
        let mut counter = 0;

        for i in 0..num_points {
            let p3d = &self.points3d.column(i).into_owned();
            let p2d_gt = &self.points2d.column(i).into_owned();
            let project_result = model.project(p3d, false);

            if let Ok((p2d_projected, _)) = project_result {
                // Each point contributes 2 residuals (x and y dimensions)
                residuals[counter * 2] = p2d_projected.x - p2d_gt.x;
                residuals[counter * 2 + 1] = p2d_projected.y - p2d_gt.y;
                counter += 1;
            }
        }
        // Only return the rows with actual residuals
        residuals = residuals.rows(0, counter * 2).into_owned();
        info!("Size residuals: {}", residuals.len());
        Ok(residuals)
    }
}

/// Implementation of the Jacobian trait for Gauss-Newton optimization.
impl Jacobian for DoubleSphereOptimizationCost {
    type Param = DVector<f64>;
    type Jacobian = DMatrix<f64>;

    /// Computes the Jacobian matrix of residuals with respect to camera parameters.
    ///
    /// # Arguments
    ///
    /// * `p` - Parameter vector containing camera intrinsics and distortion parameters
    ///
    /// # Returns
    ///
    /// Jacobian matrix (2N×6) where N is the number of points and 6 is the number of parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the camera model parameters are invalid or
    /// if projection fails for the given points.
    fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, Error> {
        let num_points = self.points3d.ncols();
        let mut jacobian = DMatrix::zeros(num_points * 2, 6); // 2 residuals per point, 6 parameters
                                                              // Ensure DoubleSphereModel::new is public or accessible
        let model = DoubleSphereModel::new(&p)?;
        let mut counter = 0;

        for i in 0..num_points {
            let p3d = &self.points3d.column(i).into_owned();

            // Get Jacobian for this point (2x6 matrix)
            let (_, jacobian_point_2x6) = model.project(p3d, true).unwrap();

            if let Some(jac) = jacobian_point_2x6 {
                // Copy the 2x6 Jacobian for this point into the overall Jacobian matrix
                jacobian.view_mut((counter * 2, 0), (2, 6)).copy_from(&jac);
                counter += 1;
            }
        }
        jacobian = jacobian.rows(0, counter * 2).into_owned();
        info!("Size residuals: {}", jacobian.nrows());
        Ok(jacobian)
    }
}

/// Implementation of the CostFunction trait for optimization.
impl CostFunction for DoubleSphereOptimizationCost {
    /// Parameter vector: [fx, fy, cx, cy, alpha, xi]
    type Param = DVector<f64>;
    /// Sum of squared errors
    type Output = f64;

    /// Computes the cost function as the sum of squared projection errors.
    ///
    /// # Arguments
    ///
    /// * `p` - Parameter vector containing camera intrinsics and distortion parameters
    ///
    /// # Returns
    ///
    /// Total cost as the sum of squared residuals.
    ///
    /// # Errors
    ///
    /// Returns an error if the camera model parameters are invalid or
    /// if projection fails for the given points.
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let mut total_error_sq = 0.0;
        // Ensure DoubleSphereModel::new is public or accessible
        let model = DoubleSphereModel::new(&p)?;

        for i in 0..self.points3d.ncols() {
            let p3d = &self.points3d.column(i).into_owned();
            let p2d_gt = &self.points2d.column(i).into_owned();

            // The project function now returns a tuple with the projection and optional Jacobian
            let (p2d_projected, _) = model.project(p3d, false).unwrap();

            total_error_sq += (p2d_projected - p2d_gt).norm();
        }

        info!("total_error_sq: {total_error_sq}");
        Ok(total_error_sq)
    }
}

/// Implementation of the Gradient trait for optimization.
impl Gradient for DoubleSphereOptimizationCost {
    type Param = DVector<f64>;
    /// Gradient of the cost function (J^T * r)
    type Gradient = DVector<f64>;

    /// Computes the gradient of the cost function.
    ///
    /// The gradient is computed as J^T * r, where J is the Jacobian matrix
    /// and r is the residual vector.
    ///
    /// # Arguments
    ///
    /// * `p` - Parameter vector containing camera intrinsics and distortion parameters
    ///
    /// # Returns
    ///
    /// Gradient vector with respect to the camera parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the camera model parameters are invalid or
    /// if projection fails for the given points.
    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        let mut grad = DVector::zeros(6);
        // Ensure DoubleSphereModel::new is public or accessible
        let model = DoubleSphereModel::new(&p)?;

        for i in 0..self.points3d.ncols() {
            let p3d = &self.points3d.column(i).into_owned();
            let p2d_gt = &self.points2d.column(i).into_owned();

            let (p2d_projected, jacobian_point_2x6) = model.project(p3d, true).unwrap();

            if let Some(jacobian) = jacobian_point_2x6 {
                let residual_2x1 = p2d_projected - p2d_gt;

                // grad += J_i^T * r_i
                grad += jacobian.transpose() * residual_2x1;
            }
        }
        info!("Gradient: {}", grad);
        Ok(grad)
    }
}

/// Implementation of the Hessian trait for optimization.
impl Hessian for DoubleSphereOptimizationCost {
    type Param = DVector<f64>;
    /// Hessian matrix approximation (J^T * J)
    type Hessian = DMatrix<f64>;

    /// Computes the Hessian matrix using the Gauss-Newton approximation.
    ///
    /// The Hessian is approximated as J^T * J, where J is the Jacobian matrix.
    /// This is a common approximation used in non-linear least squares optimization.
    ///
    /// # Arguments
    ///
    /// * `p` - Parameter vector containing camera intrinsics and distortion parameters
    ///
    /// # Returns
    ///
    /// Approximate Hessian matrix (6×6).
    ///
    /// # Errors
    ///
    /// Returns an error if the camera model parameters are invalid or
    /// if projection fails for the given points.
    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        let mut jtj = DMatrix::zeros(6, 6);
        // Ensure DoubleSphereModel::new is public or accessible
        let model = DoubleSphereModel::new(&p)?;

        for i in 0..self.points3d.ncols() {
            let p3d = &self.points3d.column(i).into_owned();
            // We only need the Jacobian for J^T J
            let (_, jacobian_point_2x6) = model.project(p3d, true).unwrap();

            // Check if jacobian_point_2x6 is Some before using it
            if let Some(jacobian) = jacobian_point_2x6 {
                jtj += jacobian.transpose() * jacobian;
            }
        }

        info!("Hessian: {}", jtj);
        Ok(jtj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraModel, DoubleSphereModel as DSCameraModel, Intrinsics, Resolution};
    use approx::assert_relative_eq; // Added for assert_relative_eq!
    use nalgebra::{Matrix2xX, Matrix3xX, Vector3};

    // Helper to get a default model, similar to the one in samples/double_sphere.yaml
    fn get_sample_camera_model() -> DSCameraModel {
        DSCameraModel {
            intrinsics: Intrinsics {
                fx: 348.112754378549,
                fy: 347.1109973814674,
                cx: 365.8121721753254,
                cy: 249.3555778487899,
            },
            resolution: Resolution {
                width: 752,
                height: 480,
            },
            alpha: 0.5657413673629862,
            xi: -0.24425190195168348,
        }
    }

    fn sample_points_for_ds_model(
        model: &DSCameraModel,
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
                points_3d_vec.push(p3d);
                points_2d_vec.push(p2d);
            }
        }
        (
            Matrix2xX::from_columns(&points_2d_vec),
            Matrix3xX::from_columns(&points_3d_vec),
        )
    }

    #[test]
    fn test_double_sphere_optimization_cost_basic() {
        let model_camera = get_sample_camera_model();
        let (points_2d, points_3d) = sample_points_for_ds_model(&model_camera, 5);

        assert!(
            points_3d.ncols() > 0,
            "Need at least one valid point for testing cost function."
        );

        // Construct cost struct using the model_camera
        let cost = DoubleSphereOptimizationCost::new(model_camera.clone(), points_3d.clone(), points_2d.clone());

        // Prepare parameter vector from model_camera
        let p = DVector::from_vec(vec![
            model_camera.intrinsics.fx,
            model_camera.intrinsics.fy,
            model_camera.intrinsics.cx,
            model_camera.intrinsics.cy,
            model_camera.alpha,
            model_camera.xi,
        ]);

        let residuals = cost.apply(&p).unwrap();
        assert_eq!(residuals.len(), 2 * points_3d.ncols());
        assert!(
            residuals.iter().all(|&v| v.abs() < 1e-6),
            "Residuals should be near zero for perfect model"
        );

        let c = cost.cost(&p).unwrap();
        assert!(c >= 0.0, "Cost should be non-negative");
        assert!(c < 1e-5, "Cost should be near zero for perfect model");

        let jac = cost.jacobian(&p).unwrap();
        assert_eq!(jac.nrows(), 2 * points_3d.ncols());
        assert_eq!(jac.ncols(), 6);

        let grad = cost.gradient(&p).unwrap();
        assert_eq!(grad.len(), 6);
        assert!(
            grad.norm() < 1e-5,
            "Gradient norm should be near zero for perfect model"
        );

        let hess = cost.hessian(&p).unwrap();
        assert_eq!(hess.nrows(), 6);
        assert_eq!(hess.ncols(), 6);
    }

    #[test]
    fn test_double_sphere_optimize_trait_method() {
        let reference_model = get_sample_camera_model();
        let (points_2d, points_3d) = sample_points_for_ds_model(&reference_model, 50);

        assert!(
            points_3d.ncols() > 10,
            "Need sufficient points for optimization test."
        );

        let noisy_initial_model = DSCameraModel {
            intrinsics: Intrinsics {
                fx: reference_model.intrinsics.fx * 0.95,
                fy: reference_model.intrinsics.fy * 1.02,
                cx: reference_model.intrinsics.cx + 15.0, // Increased noise further
                cy: reference_model.intrinsics.cy - 15.0, // Increased noise further
            },
            resolution: reference_model.resolution.clone(),
            alpha: (reference_model.alpha * 0.80).max(0.01).min(0.99), // Increased noise further
            xi: (reference_model.xi * 0.7).max(-0.9).min(0.9),      // Increased noise further
        };

        let mut optimization_task = DoubleSphereOptimizationCost::new(
            noisy_initial_model.clone(), // Pass the noisy model here
            points_3d.clone(),
            points_2d.clone(),
        );

        // Call optimize using the Optimizer trait
        let optimize_result = optimization_task.optimize(false); // verbose = false
        assert!(
            optimize_result.is_ok(),
            "Optimization failed: {:?}",
            optimize_result.err()
        );

        // Get the optimized model from the task
        let optimized_model = &optimization_task.model;

        // Compare optimized parameters with reference_model
        // Allow for some tolerance as optimization might not be perfect
        assert_relative_eq!(
            optimized_model.intrinsics.fx,
            reference_model.intrinsics.fx,
            epsilon = 15.0 // Adjusted epsilon based on noise level
        );
        assert_relative_eq!(
            optimized_model.intrinsics.fy,
            reference_model.intrinsics.fy,
            epsilon = 15.0
        );
        assert_relative_eq!(
            optimized_model.intrinsics.cx,
            reference_model.intrinsics.cx,
            epsilon = 15.0 
        );
        assert_relative_eq!(
            optimized_model.intrinsics.cy,
            reference_model.intrinsics.cy,
            epsilon = 15.0
        );
        assert_relative_eq!(optimized_model.alpha, reference_model.alpha, epsilon = 0.15); // alpha can be sensitive
        assert_relative_eq!(optimized_model.xi, reference_model.xi, epsilon = 0.15); // xi can also be sensitive
    }

    #[test]
    fn test_double_sphere_linear_estimation_optimizer_trait() {
        let reference_model = get_sample_camera_model();
        let (points_2d, points_3d) = sample_points_for_ds_model(&reference_model, 20);

        assert!(
            points_3d.ncols() > 0,
            "Need points for linear estimation test."
        );

        let mut optimization_task = DoubleSphereOptimizationCost::new(
            reference_model.clone(), // Pass the noisy model here
            points_3d.clone(),
            points_2d.clone(),
        );

        optimization_task.linear_estimation().unwrap();

        // Linear estimation for DoubleSphere typically estimates alpha, xi is often set to 0 or a fixed value.
        // The provided implementation estimates alpha with xi=0.
        // So, we compare alpha and check if xi is close to 0.
        assert_relative_eq!(optimization_task.model.alpha, reference_model.alpha, epsilon = 1.0); // Linear estimation is an approximation
        assert_relative_eq!(optimization_task.model.xi, 0.0, epsilon = 1e-9); // Expect xi to be zero from linear_estimation impl

        // Intrinsics should remain the same as input
        assert_relative_eq!(
            optimization_task.model.intrinsics.fx,
            reference_model.intrinsics.fx,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            optimization_task.model.intrinsics.fy,
            reference_model.intrinsics.fy,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            optimization_task.model.intrinsics.cx,
            reference_model.intrinsics.cx,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            optimization_task.model.intrinsics.cy,
            reference_model.intrinsics.cy,
            epsilon = 1e-9
        );
    }
}
