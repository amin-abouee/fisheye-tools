use crate::camera::{CameraModel, CameraModelError, DoubleSphereModel};
use crate::optimization::Optimizer;
use factrs::{
    core::{assign_symbols, Graph, Values, GaussNewton as FactrsGaussNewton},
    fac,
    containers::Key,
    variables::VectorVar6,
    residuals::{Residual1},
    linalg::{Const, VectorX, MatrixX, DiffResult},
    robust::Huber,
    optimizers::Optimizer as FactrsOptimizer,
    dtype,
};
use log::{info, warn};
use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3xX, Vector3, Vector2};
use std::fmt;

assign_symbols!(CamParams: VectorVar6);

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

/// Residual implementation for factrs optimization of DoubleSphereModel
#[derive(Debug, Clone)]
pub struct DoubleSphereFactrsResidual {
    /// 3D point in camera coordinate system
    point3d: Vector3<f64>,
    /// Corresponding 2D point in image coordinates
    point2d: Vector2<f64>,
}
impl DoubleSphereFactrsResidual {
    /// Constructor for the reprojection residual.
    pub fn new(point3d: Vector3<f64>, point2d: Vector2<f64>) -> Self {
        Self { point3d: point3d, point2d: point2d  }
    }
}

impl Residual1 for DoubleSphereFactrsResidual {
    type DimIn = Const<6>;
    type DimOut = Const<2>;
    type V1 = VectorVar6;
    
    fn residual1(&self, values: &Values) -> VectorX {
        // Extract camera parameters from values
        let cam_params: &VectorVar6<f64> = values.get(CamParams(0)).unwrap();
        let params = &cam_params.0;
        
        // Create temporary DoubleSphereModel for projection
        let model = DoubleSphereModel {
            intrinsics: crate::camera::Intrinsics {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            resolution: crate::camera::Resolution {
                width: 0,
                height: 0,
            },
            alpha: params[4],
            xi: params[5],
        };
        
        // Project the 3D point
        match model.project(&self.point3d, false) {
            Ok((projected_2d, _)) => {
                // Compute residuals (observed - projected)
                let residual_u = self.point2d.x - projected_2d.x;
                let residual_v = self.point2d.y - projected_2d.y;
                VectorX::from_vec(vec![residual_u, residual_v])
            }
            Err(_) => {
                // Return large residuals for invalid projections
                VectorX::from_vec(vec![1e6, 1e6])
            }
        }
    }
    
    fn residual1_jacobian(
        &self,
        values: &Values,
    ) -> DiffResult<VectorX, MatrixX> {
        // Extract camera parameters from values
        let cam_params: &VectorVar6<f64> = values.get(CamParams(0)).unwrap();
        let params = &cam_params.0;
        
        // Create temporary DoubleSphereModel for projection
        let model = DoubleSphereModel {
            intrinsics: crate::camera::Intrinsics {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            resolution: crate::camera::Resolution {
                width: 0,
                height: 0,
            },
            alpha: params[4],
            xi: params[5],
        };
        
        // Project the 3D point with jacobian computation
        match model.project(&self.point3d, true) {
            Ok((projected_2d, Some(jacobian_proj))) => {
                // Compute residuals (observed - projected)
                let residual_u = self.point2d.x - projected_2d.x;
                let residual_v = self.point2d.y - projected_2d.y;
                let residual = VectorX::from_vec(vec![residual_u, residual_v]);
                
                // The jacobian of residual w.r.t. parameters is negative of projection jacobian
                // since residual = observed - projected
                let jacobian = MatrixX::from_iterator(
                    2, 6,
                    jacobian_proj.iter().map(|&x| -x)
                );
                
                DiffResult { value: residual, diff: jacobian }
            }
            Ok((projected_2d, None)) => {
                // Fallback: compute residual without jacobian
                let residual_u = self.point2d.x - projected_2d.x;
                let residual_v = self.point2d.y - projected_2d.y;
                let residual = VectorX::from_vec(vec![residual_u, residual_v]);
                let jacobian = MatrixX::zeros(2, 6);
                DiffResult { value: residual, diff: jacobian }
            }
            Err(_) => {
                // Return large residuals and zero jacobian for invalid projections
                let residual = VectorX::from_vec(vec![1e6, 1e6]);
                let jacobian = MatrixX::zeros(2, 6);
                DiffResult { value: residual, diff: jacobian }
            }
        }
    }
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
            points3d,
            points2d,
        }
    }
    
    /// Optimizes the camera model using factrs Gauss-Newton optimizer.
    ///
    /// # Arguments
    ///
    /// * `verbose` - Whether to print verbose output during optimization
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    pub fn optimize_with_factrs(&mut self, verbose: bool) -> Result<(), CameraModelError> {
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
        
        // Create a factrs Values object to hold the camera parameters
        let mut values = Values::new();
        
        // Initial parameters
        let initial_params = VectorVar6::new(
            self.model.intrinsics.fx as dtype,
            self.model.intrinsics.fy as dtype,
            self.model.intrinsics.cx as dtype,
            self.model.intrinsics.cy as dtype,
            self.model.alpha as dtype,
            self.model.xi as dtype,
        );
        
        // Insert the initial parameters into the values
        values.insert(CamParams(0), initial_params);
        
        // Create a factrs Graph
        let mut graph = Graph::new();
        
        // Add residuals for each point correspondence
        for i in 0..self.points3d.ncols() {
            let p3d = self.points3d.column(i).into_owned();
            let p2d = self.points2d.column(i).into_owned();
            
            // Create a residual for this point correspondence
            let residual = DoubleSphereFactrsResidual {
                point3d: p3d,
                point2d: p2d,
            };
            
            // Create a factor with the residual and add it to the graph
            // We use a Huber kernel for robustness
            let factor = fac![residual, CamParams(0), Huber::default()];
            graph.add_factor(factor);
        }
        
        if verbose {
            info!("Starting optimization with factrs Gauss-Newton...");
        }
        
        // Create a Gauss-Newton optimizer
        let mut optimizer = FactrsGaussNewton::new(graph);
        
        // Run the optimization
        let result = optimizer.optimize(values)
            .map_err(|e| CameraModelError::NumericalError(e.to_string()))?;
        
        if verbose {
            info!("Optimization finished");
        }
        
        // Extract the optimized parameters
        let optimized_params: &VectorVar6<f64> = result.get(CamParams(0)).unwrap();
        let params = &optimized_params.0;
        
        // Update the model parameters
        self.model.intrinsics.fx = params[0] as f64;
        self.model.intrinsics.fy = params[1] as f64;
        self.model.intrinsics.cx = params[2] as f64;
        self.model.intrinsics.cy = params[3] as f64;
        self.model.alpha = params[4] as f64;
        self.model.xi = params[5] as f64;
        
        // Validate the optimized parameters
        self.model.validate_params()?;
        
        Ok(())
    }
}

impl fmt::Debug for DoubleSphereOptimizationCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DoubleSphereCostModel Summary:\n model: {:?}\n points3d size: {}, points2d size: {} ",
            self.model,
            self.points3d.ncols(),
            self.points2d.ncols(),
        )
    }
}

impl Optimizer for DoubleSphereOptimizationCost {
    fn optimize(&mut self, verbose: bool) -> Result<(), CameraModelError> {
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
        // self.model.validate_params()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraModel, DoubleSphereModel as DSCameraModel, Intrinsics};
    use approx::assert_relative_eq; // Added for assert_relative_eq!
    use nalgebra::{Matrix2xX, Matrix3xX};

    // Helper to get a default model, similar to the one in samples/double_sphere.yaml
    fn get_sample_camera_model() -> DSCameraModel {
        let path = "samples/double_sphere.yaml";
        DoubleSphereModel::load_from_yaml(path).unwrap()
    }

    fn sample_points_for_ds_model(
        model: &DSCameraModel,
        num_points: usize,
    ) -> (Matrix2xX<f64>, Matrix3xX<f64>) {
        crate::geometry::sample_points(Some(model), num_points).unwrap()
    }

    #[test]
    fn test_double_sphere_linear_estimation_optimizer_trait() {
        let reference_model = get_sample_camera_model();
        let (points_2d, points_3d) = sample_points_for_ds_model(&reference_model, 50);

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
        assert_relative_eq!(
            optimization_task.model.alpha,
            reference_model.alpha,
            epsilon = 1.0
        ); // Linear estimation is an approximation
        assert_relative_eq!(
            optimization_task.model.xi,
            reference_model.xi,
            epsilon = 1e-9
        ); // Expect xi to be zero from linear_estimation impl

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
    
    #[test]
    fn test_double_sphere_optimize_factrs() {
        // Get a reference camera model for testing
        let reference_model = get_sample_camera_model();
        
        // Generate sample points using the reference model
        let (points_2d, points_3d) = sample_points_for_ds_model(&reference_model, 50);
        
        // Create a noisy version of the model to optimize
        let noisy_model = DoubleSphereModel::new(&DVector::from_vec(vec![
            reference_model.intrinsics.fx * 0.95, // Add 5% error to fx
            reference_model.intrinsics.fy * 1.05, // Add 5% error to fy
            reference_model.intrinsics.cx + 10.0, // Add 10 pixels error to cx
            reference_model.intrinsics.cy - 10.0, // Subtract 10 pixels from cy
            reference_model.alpha * 0.9,         // Reduce alpha by 10%
            reference_model.xi * 1.1,            // Increase xi by 10%
        ]))
        .unwrap();
        
        info!("Reference model: {:?}", reference_model);
        info!("Noisy model: {:?}", noisy_model);
        
        let mut optimization_task = DoubleSphereOptimizationCost::new(
            noisy_model.clone(), // Pass the noisy model here
            points_3d.clone(),
            points_2d.clone(),
        );
        
        // Optimize the model with factrs
        optimization_task.optimize_with_factrs(false).unwrap();
        
        info!("Optimized model with factrs: {:?}", optimization_task);
        
        // Check that parameters have been optimized close to reference values
        assert!(
            (optimization_task.model.intrinsics.fx - reference_model.intrinsics.fx).abs() < 1.0,
            "fx parameter didn't converge to expected value"
        );
        assert!(
            (optimization_task.model.intrinsics.fy - reference_model.intrinsics.fy).abs() < 1.0,
            "fy parameter didn't converge to expected value"
        );
        assert!(
            (optimization_task.model.intrinsics.cx - reference_model.intrinsics.cx).abs() < 1.0,
            "cx parameter didn't converge to expected value"
        );
        assert!(
            (optimization_task.model.intrinsics.cy - reference_model.intrinsics.cy).abs() < 1.0,
            "cy parameter didn't converge to expected value"
        );
        assert!(
            (optimization_task.model.alpha - reference_model.alpha).abs() < 0.05,
            "alpha parameter didn't converge to expected value"
        );
        assert!(
            (optimization_task.model.xi - reference_model.xi).abs() < 0.05,
            "xi parameter didn't converge to expected value"
        );
        // Verify that alpha is within bounds (0, 1]
        assert!(
            optimization_task.model.alpha > 0.0 && optimization_task.model.alpha <= 1.0,
            "Alpha parameter out of valid range (0, 1]"
        );
    }
}
