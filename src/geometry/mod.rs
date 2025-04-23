use nalgebra::{Matrix2xX, Vector2};

/// Generate a grid of sample points that are evenly distributed across the image
///
/// # Arguments
///
/// * `width` - The width of the image in pixels
/// * `height` - The height of the image in pixels
/// * `n` - The approximate number of points to generate
///
/// # Returns
///
/// A Matrix2xX where each column represents a 2D point with pixel coordinates
pub fn sample_points(width: f64, height: f64, n: usize) -> Matrix2xX<f64> {
    // Calculate the number of cells in each dimension
    let num_cells_x = (n as f64 * (width / height)).sqrt().round() as i32;
    let num_cells_y = (n as f64 * (height / width)).sqrt().round() as i32;

    // Calculate the dimensions of each cell
    let cell_width = width / num_cells_x as f64;
    let cell_height = height / num_cells_y as f64;

    // Calculate total number of points
    let total_points = (num_cells_x * num_cells_y) as usize;

    // Create a matrix with the appropriate size
    let mut points_matrix = Matrix2xX::zeros(total_points);

    // Generate a point at the center of each cell
    let mut idx = 0;
    for i in 0..num_cells_y {
        for j in 0..num_cells_x {
            let x = (j as f64 + 0.5) * cell_width;
            let y = (i as f64 + 0.5) * cell_height;
            points_matrix.set_column(idx, &Vector2::new(x, y));
            idx += 1;
        }
    }
    points_matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_points() {
        let width = 800f64;
        let height = 600f64;
        let n = 100;

        let points_matrix = sample_points(width, height, n);

        // Test that the number of points is approximately n
        // (It might not be exactly n due to rounding)
        let expected_count = (n as f64 * 0.8) as usize..=(n as f64 * 1.2) as usize;
        assert!(
            expected_count.contains(&points_matrix.ncols()),
            "Expected around {} points, got {}",
            n,
            points_matrix.ncols()
        );

        // Test that all points are within the image bounds
        for col_idx in 0..points_matrix.ncols() {
            let point = points_matrix.column(col_idx);

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
        }
    }
}
