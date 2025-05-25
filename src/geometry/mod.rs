//! Provides geometric utilities, primarily for generating sample points for camera calibration or testing.
//!
//! This module contains functions to create sets of 2D image points and their corresponding
//! 3D unprojected rays, based on a given camera model. It relies on the [`CameraModel`] trait
//! and [`CameraModelError`] from the `crate::camera` module to interact with different
//! camera model implementations and handle potential errors during geometric operations like unprojection.
//!
//! The main functionality offered is the [`sample_points`] function, which generates a grid
//! of 2D points across the image plane (defined by the camera's resolution) and then
//! unprojects these points into 3D space using the provided camera model.

use crate::camera::{CameraModel, CameraModelError};
use nalgebra::{Matrix2xX, Matrix3xX, Vector2};

/// Errors that can occur during geometric operations within this module.
///
/// This enum categorizes various issues that might arise, such as problems
/// with the camera model, numerical instabilities, or failures in projection/unprojection.
#[derive(thiserror::Error, Debug)]
pub enum GeometryError {
    /// Indicates that a required camera model was not provided or is not valid.
    #[error("Camera model does not exist")]
    CameraModelDoesNotExist,
    /// Represents an error due to numerical issues during a computation (e.g., division by zero, instability).
    /// Contains a string describing the specific numerical error.
    #[error("Numerical error in computation: {0}")]
    NumericalError(String),
    /// Indicates that a matrix involved in a computation was singular where a non-singular matrix was expected.
    #[error("Matrix singularity detected")]
    SingularMatrix,
    /// Represents an error that occurred during the projection of a 3D point to 2D image coordinates.
    /// Contains a string describing the specific projection error.
    #[error("Point projection failed: {0}")]
    ProjectionError(String),
}

/// Generates a grid of 2D sample points evenly distributed across an image and their corresponding 3D unprojected rays.
///
/// This function first determines the image resolution from the provided `camera_model`.
/// It then calculates a grid layout to distribute approximately `n` points across this
/// image. For each cell in the grid, a 2D point is generated at its center.
/// These 2D points are subsequently unprojected into 3D rays using the
/// `camera_model.unproject()` method. Only points that are successfully unprojected
/// and have a positive Z-coordinate (i.e., are in front of the camera) are included
/// in the final output.
///
/// # Arguments
///
/// *   `camera_model`: `Option<&T>` where `T: ?Sized + CameraModel` - A reference to an object
///     that implements the [`CameraModel`] trait. This model is used to:
///     1.  Obtain the image resolution (`width` and `height`).
///     2.  Perform the unprojection of 2D points to 3D rays.
///     The current implementation uses `unwrap()` on this option, so providing `None`
///     will cause a panic. Future extensions might handle `None` differently.
/// *   `n`: `usize` - The approximate total number of 2D points to generate and attempt
///     to unproject. The actual number of points returned may be less than `n` due to
///     the grid rounding and filtering of invalid unprojections.
///
/// # Return Value
///
/// Returns a `Result` containing a tuple of two matrices, or a [`CameraModelError`]:
/// *   `Ok((Matrix2xX<f64>, Matrix3xX<f64>))`: On success, a tuple where:
///     *   The first element is a `Matrix2xX<f64>` (2 rows, N columns). Each column
///         represents a 2D point `(u, v)` in pixel coordinates.
///     *   The second element is a `Matrix3xX<f64>` (3 rows, N columns). Each column
///         represents the corresponding 3D point (unprojected ray) `(X, Y, Z)`.
///         These 3D points are typically normalized direction vectors (as returned by
///         most `unproject` implementations) and are guaranteed to have `Z > 0`.
/// *   `Err(CameraModelError)`: If an error occurs during the unprojection process
///     (e.g., a point is outside the valid image area for unprojection, or a numerical
///     issue occurs within the camera model's `unproject` method).
///
/// # Panics
///
/// *   Panics if `camera_model` is `None`, as the current implementation calls `unwrap()`
///     on it to get the resolution and the model itself.
///
/// # Algorithm
///
/// 1.  Retrieves the image `width` and `height` from `camera_model.get_resolution()`.
///     (Panics if `camera_model` is `None`).
/// 2.  Calculates the number of grid cells in the x and y dimensions (`num_cells_x`, `num_cells_y`)
///     to approximate `n` total points while maintaining the image aspect ratio.
/// 3.  Determines the `cell_width` and `cell_height`.
/// 4.  Generates 2D points at the center of each grid cell. These are stored initially in
///     `points_2d_matrix`.
/// 5.  Retrieves the `camera_model` again by `unwrap()`. (Panics if `camera_model` was `None`).
/// 6.  Iterates through each generated 2D point:
///     a.  Calls `camera_model.unproject()` for the 2D point.
///     b.  If unprojection is successful and the resulting 3D point's Z-coordinate is positive,
///         the 2D point and the 3D point are stored in `valid_2d_points` and `valid_3d_points`
///         respectively.
/// 7.  Converts the vectors of valid points (`valid_2d_points`, `valid_3d_points`) into
///     `Matrix2xX<f64>` and `Matrix3xX<f64>` format.
/// 8.  Returns the resulting pair of matrices.
///
/// # Examples
///
/// ```rust
/// use nalgebra::{Matrix2xX, Matrix3xX};
/// use vision_toolkit_rs::camera::{CameraModel, CameraModelError, Intrinsics, Resolution, DoubleSphereModel};
/// use vision_toolkit_rs::geometry::sample_points;
///
/// // Example using DoubleSphereModel (ensure it's available in your scope)
/// fn generate_sample_data() -> Result<(), CameraModelError> {
///     // Create a sample DoubleSphereModel (parameters are illustrative)
///     let intrinsics = Intrinsics { fx: 300.0, fy: 300.0, cx: 320.0, cy: 240.0 };
///     let resolution = Resolution { width: 640, height: 480 };
///     let ds_model = DoubleSphereModel {
///         intrinsics,
///         resolution,
///         alpha: 0.5, // Example value
///         xi: 0.1,    // Example value
///     };
///
///     let num_points_to_generate = 50;
///     match sample_points(Some(&ds_model), num_points_to_generate) {
///         Ok((points_2d, points_3d)) => {
///             println!("Generated {} valid 2D points and {} valid 3D points.",
///                      points_2d.ncols(), points_3d.ncols());
///             // Further processing with points_2d and points_3d
///             assert_eq!(points_2d.ncols(), points_3d.ncols());
///             if points_3d.ncols() > 0 {
///                 assert!(points_3d.column(0)[2] > 0.0, "First 3D point should have Z > 0");
///             }
///         }
///         Err(e) => {
///             eprintln!("Error generating sample points: {:?}", e);
///             return Err(e);
///         }
///     }
///     Ok(())
/// }
/// // generate_sample_data().unwrap(); // Uncomment to run example
/// ```
pub fn sample_points<T>(
    camera_model: Option<&T>,
    n: usize,
) -> Result<(Matrix2xX<f64>, Matrix3xX<f64>), CameraModelError>
where
    T: ?Sized + CameraModel,
{
    // Panics if camera_model is None. Documented behavior.
    let width = camera_model.unwrap().get_resolution().width as f64;
    let height = camera_model.unwrap().get_resolution().height as f64;

    // Calculate the number of cells in each dimension
    // Ensure num_cells_x and num_cells_y are at least 1 to avoid division by zero if n is small or aspect ratio extreme.
    let num_cells_x = ((n as f64 * (width / height)).sqrt().round() as i32).max(1);
    let num_cells_y = ((n as f64 * (height / width)).sqrt().round() as i32).max(1);

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

    // Unwrap the camera model (Panics if camera_model was None. Documented behavior.)
    let camera_model_ref = camera_model.unwrap();

    // Prepare vectors to store valid points
    let mut valid_2d_points = Vec::new();
    let mut valid_3d_points = Vec::new();

    // Unproject each 2D point and filter for z > 0
    for col_idx in 0..points_2d_matrix.ncols() {
        let point_2d_col = points_2d_matrix.column(col_idx);
        let p2d = Vector2::new(point_2d_col[0], point_2d_col[1]);

        // Try to unproject the point
        match camera_model_ref.unproject(&p2d) {
            Ok(p3d) => {
                // Only keep points with z > 0
                if p3d.z > 0.0 {
                    // Store the valid 2D point
                    valid_2d_points.push(p2d);
                    // Store the corresponding 3D point
                    valid_3d_points.push(p3d);
                }
            }
            Err(_e) => {
                // If unprojection fails for a point, it's skipped.
                // Depending on verbosity settings, one might log this error.
                // For now, we silently skip as per original logic.
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
        points_2d_result.set_column(idx, &p2d);
        points_3d_result.set_column(idx, &p3d);
    }

    Ok((points_2d_result, points_3d_result))
}

/// Contains unit tests for the geometry module.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::DoubleSphereModel; // Using DoubleSphereModel for testing

    /// Tests the `sample_points` function with a `DoubleSphereModel`.
    ///
    /// This test verifies that:
    /// 1.  `sample_points` successfully generates some 2D and 3D points when provided
    ///     with a valid camera model.
    /// 2.  The number of generated 2D points matches the number of 3D points.
    /// 3.  All generated 3D points have a Z-coordinate greater than 0 (i.e., are in front of the camera).
    #[test]
    fn test_sample_points() {
        let input_path = "samples/double_sphere.yaml"; // Test depends on this sample file
        let camera_model = DoubleSphereModel::load_from_yaml(input_path).unwrap();
        let n = 100 as usize; // Approximate number of points to generate
        let (points_2d, points_3d) = sample_points(Some(&camera_model), n).unwrap();

        // Check that we have some valid points (actual number depends on grid and unprojection success)
        assert!(!points_2d.is_empty(), "No valid 2D-points were generated");
        assert!(
            !points_3d.is_empty(),
            "No valid 3D-points were generated with camera model"
        );

        assert_eq!(
            points_2d.ncols(), points_3d.ncols(),
            "Number of 2D and 3D points should be equal"
        );

        // Check that all 3D points have z > 0
        for col_idx in 0..points_3d.ncols() {
            let point_3d = points_3d.column(col_idx);
            assert!(point_3d[2] > 0.0, "3D point has z <= 0: {:?}", point_3d);
        }
    }
}
