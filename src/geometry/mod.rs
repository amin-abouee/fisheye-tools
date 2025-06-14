use crate::camera::{CameraModel, CameraModelError};
use nalgebra::{Matrix2xX, Matrix3xX, Vector2};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(thiserror::Error, Debug)]
pub enum GeometryError {
    #[error("Camera model does not exist")]
    CameraModelDoesNotExist,
    #[error("Numerical error in computation: {0}")]
    NumericalError(String),
    #[error("Matrix singularity detected")]
    SingularMatrix,
    #[error("Zero projection points")]
    ZeroProjectionPoints,
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ProjectionError {
    pub rmse: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub stddev: f64,
    pub median: f64,
}

impl fmt::Debug for ProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Projection Error [ rmse: {}, min: {}, max: {}, mean: {}, stddev: {}, median: {} ]",
            self.rmse, self.min, self.max, self.mean, self.stddev, self.median
        )
    }
}

/// Generate a grid of sample points that are evenly distributed across the image,
/// optionally unprojecting them to 3D using a provided camera model
///
/// # Arguments
///
/// * `width` - The width of the image in pixels
/// * `height` - The height of the image in pixels
/// * `n` - The approximate number of points to generate
/// * `camera_model` - Camera model to use for unprojection. If None, 3D points will be
///   on a plane at z=1.0
///
/// # Returns
///
/// * A tuple containing:
///   * Matrix2xX where each column represents a 2D point with pixel coordinates
///   * Matrix3xX where each column represents the corresponding 3D point
pub fn sample_points<T>(
    camera_model: Option<&T>,
    n: usize,
) -> Result<(Matrix2xX<f64>, Matrix3xX<f64>), CameraModelError>
where
    T: ?Sized + CameraModel,
{
    let width = camera_model.unwrap().get_resolution().width as f64;
    let height = camera_model.unwrap().get_resolution().height as f64;
    // Calculate the number of cells in each dimension
    let num_cells_x = (n as f64 * (width / height)).sqrt().round() as i32;
    let num_cells_y = (n as f64 * (height / width)).sqrt().round() as i32;

    // Calculate the dimensions of each cell
    let cell_width = width / num_cells_x as f64;
    let cell_height = height / num_cells_y as f64;

    // Calculate total number of points
    let total_points = (num_cells_x * num_cells_y) as usize;

    // Create a matrix with the appropriate size
    let mut points_2d_matrix = Matrix2xX::zeros(total_points);

    // Generate a point at the center of each cell
    let mut idx = 0;
    for i in 0..num_cells_y {
        for j in 0..num_cells_x {
            let x = (j as f64 + 0.5) * cell_width;
            let y = (i as f64 + 0.5) * cell_height;
            points_2d_matrix.set_column(idx, &Vector2::new(x, y));
            idx += 1;
        }
    }

    // Unwrap the camera model (safe because we checked it's Some)
    let camera_model = camera_model.unwrap();

    // Prepare vectors to store valid points
    let mut valid_2d_points = Vec::new();
    let mut valid_3d_points = Vec::new();

    // Unproject each 2D point and filter for z > 0
    for col_idx in 0..points_2d_matrix.ncols() {
        let point_2d = points_2d_matrix.column(col_idx);
        let p2d = Vector2::new(point_2d[0], point_2d[1]);

        // Try to unproject the point
        if let Ok(p3d) = camera_model.unproject(&p2d) {
            // Only keep points with z > 0
            if p3d.z > 0.0 {
                // Store the valid 2D point
                valid_2d_points.push(p2d);

                // Store the corresponding 3D point
                valid_3d_points.push(p3d);
            }
        }
    }

    // Convert vectors to matrices
    let n_valid = valid_2d_points.len();
    let mut points_2d_result = Matrix2xX::zeros(n_valid);
    let mut points_3d_result = Matrix3xX::zeros(n_valid);

    for (idx, (p2d, p3d)) in valid_2d_points
        .iter()
        .zip(valid_3d_points.iter())
        .enumerate()
    {
        points_2d_result.set_column(idx, p2d);
        points_3d_result.set_column(idx, p3d);
    }

    Ok((points_2d_result, points_3d_result))
}

pub fn compute_reprojection_error<T>(
    camera_model: Option<&T>,
    points3d: &Matrix3xX<f64>,
    points2d: &Matrix2xX<f64>,
) -> Result<ProjectionError, GeometryError>
where
    T: ?Sized + CameraModel,
{
    let camera_model = camera_model.unwrap();
    let mut errors = vec![];
    for i in 0..points3d.ncols() {
        let point3d = points3d.column(i).into_owned();
        let point2d = points2d.column(i).into_owned();

        if let Ok(point2d_projected) = camera_model.project(&point3d) {
            let reprojection_error = (point2d_projected - point2d).norm();
            errors.push(reprojection_error);
        }
    }

    if errors.is_empty() {
        return Err(GeometryError::ZeroProjectionPoints);
    }

    // Calculate statistics
    let n = errors.len() as f64;
    let sum: f64 = errors.iter().sum::<f64>();
    let mean = sum / n;

    // Calculate variance and standard deviation
    let variance: f64 = errors.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    // Calculate RMSE
    let sum_squared: f64 = errors.iter().map(|x| x.powi(2)).sum::<f64>();
    let rmse = (sum_squared / n).sqrt();

    // Find min and max
    let min = errors.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = errors.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Calculate median
    let mut sorted_errors = errors.clone();
    sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted_errors.len() % 2 == 0 {
        let mid = sorted_errors.len() / 2;
        (sorted_errors[mid - 1] + sorted_errors[mid]) / 2.0
    } else {
        sorted_errors[sorted_errors.len() / 2]
    };

    Ok(ProjectionError {
        rmse,
        min,
        max,
        mean,
        stddev,
        median,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::DoubleSphereModel;

    #[test]
    fn test_sample_points() {
        let input_path = "samples/double_sphere.yaml";
        let camera_model = DoubleSphereModel::load_from_yaml(input_path).unwrap();
        let n = 100_usize;
        let (points_2d, points_3d) = sample_points(Some(&camera_model), n).unwrap();

        // Check that we have some valid points
        assert!(!points_2d.is_empty(), "No valid 2D-points were generated");

        assert!(
            !points_3d.is_empty(),
            "No valid 3D-points were generated with camera model"
        );

        assert!(
            points_2d.ncols() == points_3d.ncols(),
            "Number of 2D and 3D points should be equal"
        );

        // Check that all 3D points have z > 0
        for col_idx in 0..points_3d.ncols() {
            let point_3d = points_3d.column(col_idx);
            assert!(point_3d[2] > 0.0, "3D point has z <= 0");
        }
    }
}
