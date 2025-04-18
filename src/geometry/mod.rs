use nalgebra::Point2;

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
/// A vector of 2D points representing pixel coordinates
pub fn sample_points(width: f64, height: f64, n: usize) -> Vec<Point2<f64>> {
    let mut points = Vec::new();

    // Calculate the number of cells in each dimension
    let num_cells_x = (n as f64 * (width / height)).sqrt().round() as i32;
    let num_cells_y = (n as f64 * (height / width)).sqrt().round() as i32;

    // Calculate the dimensions of each cell
    let cell_width = width / num_cells_x as f64;
    let cell_height = height / num_cells_y as f64;

    // Generate a point at the center of each cell
    for i in 0..num_cells_y {
        for j in 0..num_cells_x {
            let x = (j as f64 + 0.5) * cell_width;
            let y = (i as f64 + 0.5) * cell_height;
            points.push(Point2::new(x, y));
        }
    }

    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_points() {
        let width = 800f64;
        let height = 600f64;
        let n = 100;

        let points = sample_points(width, height, n);

        // Test that the number of points is approximately n
        // (It might not be exactly n due to rounding)
        let expected_count = (n as f64 * 0.8) as usize..=(n as f64 * 1.2) as usize;
        assert!(
            expected_count.contains(&points.len()),
            "Expected around {} points, got {}",
            n,
            points.len()
        );

        // Test that all points are within the image bounds
        for point in &points {
            assert!(
                point.x >= 0.0 && point.x < width as f64,
                "Point x-coordinate outside image bounds: {}",
                point.x
            );
            assert!(
                point.y >= 0.0 && point.y < height as f64,
                "Point y-coordinate outside image bounds: {}",
                point.y
            );
        }
    }
}
