use crate::camera::{CameraModel, CameraModelError};
use nalgebra::{Matrix2xX, Matrix3xX, Point2, Vector2, Vector3};

/// Generate a grid of sample points that are evenly distributed across the image,
/// optionally unprojecting them to 3D using a provided camera model
///
/// # Arguments
///
/// * `width` - The width of the image in pixels
/// * `height` - The height of the image in pixels
/// * `n` - The approximate number of points to generate
/// * `camera_model` - Camera model to use for unprojection. If None, 3D points will be
///                   on a plane at z=1.0
///
/// # Returns
///
/// * A tuple containing:
///   * Matrix2xX where each column represents a 2D point with pixel coordinates
///   * Matrix3xX where each column represents the corresponding 3D point
pub fn sample_points<T: CameraModel>(
    width: f64,
    height: f64,
    n: usize,
    camera_model: Option<&T>,
) -> Result<(Matrix2xX<f64>, Matrix3xX<f64>), CameraModelError> {
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

    // If no camera model is provided, create 3D points at z=1
    if camera_model.is_none() {
        let mut points_3d_result = Matrix3xX::zeros(total_points);
        for col_idx in 0..points_2d_matrix.ncols() {
            let point_2d = points_2d_matrix.column(col_idx);
            points_3d_result.set_column(col_idx, &Vector3::new(point_2d[0], point_2d[1], 1.0));
        }
        return Ok((points_2d_matrix, points_3d_result));
    }

    // Unwrap the camera model (safe because we checked it's Some)
    let camera_model = camera_model.unwrap();

    // Prepare vectors to store valid points
    let mut valid_2d_points = Vec::new();
    let mut valid_3d_points = Vec::new();

    // Unproject each 2D point and filter for z > 0
    for col_idx in 0..points_2d_matrix.ncols() {
        let point_2d = points_2d_matrix.column(col_idx);
        let p2d = Point2::new(point_2d[0], point_2d[1]);

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
        points_2d_result.set_column(idx, &Vector2::new(p2d.x, p2d.y));
        points_3d_result.set_column(idx, &Vector3::new(p3d.x, p3d.y, p3d.z));
    }

    Ok((points_2d_result, points_3d_result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{DoubleSphereModel, Intrinsics, Resolution};

    #[test]
    fn test_sample_points() {
        // Test without camera model
        let width = 800f64;
        let height = 600f64;
        let n = 100;

        let (points_2d, points_3d) =
            sample_points(width, height, n, None::<&DoubleSphereModel>).unwrap();

        // Test that the number of points is approximately n
        let expected_count = (n as f64 * 0.8) as usize..=(n as f64 * 1.2) as usize;
        assert!(
            expected_count.contains(&points_2d.ncols()),
            "Expected around {} points, got {}",
            n,
            points_2d.ncols()
        );

        // Test that all points are within the image bounds
        for col_idx in 0..points_2d.ncols() {
            let point = points_2d.column(col_idx);
            assert!(
                point[0] >= 0.0 && point[0] < width,
                "Point x-coordinate outside image bounds: {}",
                point[0]
            );
            assert!(
                point[1] >= 0.0 && point[1] < height,
                "Point y-coordinate outside image bounds: {}",
                point[1]
            );

            // Check that 3D points have z=1.0 when no camera model is provided
            let point_3d = points_3d.column(col_idx);
            assert_eq!(
                point_3d[2], 1.0,
                "z coordinate should be 1.0 without camera model"
            );
        }

        // Test with camera model
        let intrinsics = Intrinsics {
            fx: 400.0,
            fy: 400.0,
            cx: width / 2.0,
            cy: height / 2.0,
        };
        let resolution = Resolution {
            width: width as u32,
            height: height as u32,
        };
        let camera_model = DoubleSphereModel {
            intrinsics,
            resolution,
            xi: 0.1,
            alpha: 0.5,
        };

        let (points_2d, points_3d) = sample_points(width, height, n, Some(&camera_model)).unwrap();

        // Check that we have some valid points
        assert!(
            !points_2d.is_empty(),
            "No valid points were generated with camera model"
        );

        // Check that all 3D points have z > 0
        for col_idx in 0..points_3d.ncols() {
            let point_3d = points_3d.column(col_idx);
            assert!(point_3d[2] > 0.0, "3D point has z <= 0");
        }
    }
}
