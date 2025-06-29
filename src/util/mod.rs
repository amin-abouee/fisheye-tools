use crate::camera::{CameraModel, CameraModelEnum, CameraModelError};
use image::{GrayImage, Rgb, RgbImage};
use nalgebra::{Matrix2xX, Matrix3xX, Vector2};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

/// Ensure the output directory exists
pub fn ensure_output_dir() -> Result<(), UtilError> {
    let output_dir = Path::new("output");
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).map_err(|e| {
            UtilError::InvalidParams(format!("Failed to create output directory: {e}"))
        })?;
    }
    Ok(())
}

#[derive(thiserror::Error, Debug)]
pub enum UtilError {
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

/// Image quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageQualityMetrics {
    pub psnr: f64,
    pub ssim: f64,
}

/// Parameter estimation metrics with detailed model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionMetrics {
    pub model: CameraModelEnum,
    pub model_name: String,
    pub final_reprojection_error: ProjectionError,
    pub initial_reprojection_error: ProjectionError,
    pub optimization_time_ms: f64,
    pub convergence_status: String,
    pub validation_results: ValidationResults,
    pub image_quality: Option<ImageQualityMetrics>,
}

/// Validation results for conversion accuracy testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub center_error: f64,
    pub near_center_error: f64,
    pub mid_region_error: f64,
    pub edge_region_error: f64,
    pub far_edge_error: f64,
    pub average_error: f64,
    pub max_error: f64,
    pub status: String,
    pub region_data: Vec<RegionValidation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionValidation {
    pub name: String,
    pub input_projection: Option<(f64, f64)>,
    pub output_projection: Option<(f64, f64)>,
    pub error: f64,
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
) -> Result<ProjectionError, UtilError>
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
        return Err(UtilError::ZeroProjectionPoints);
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

/// Calculate Peak Signal-to-Noise Ratio (PSNR) between two images
///
/// # Arguments
///
/// * `img1` - First image
/// * `img2` - Second image
///
/// # Returns
///
/// * `Result<f64, UtilError>` - PSNR value in dB
pub fn calculate_psnr(img1: &RgbImage, img2: &RgbImage) -> Result<f64, UtilError> {
    if img1.dimensions() != img2.dimensions() {
        return Err(UtilError::InvalidParams(
            "Images must have the same dimensions".to_string(),
        ));
    }

    let (width, height) = img1.dimensions();
    let mut mse = 0.0;
    let mut valid_pixels = 0;

    for y in 0..height {
        for x in 0..width {
            let pixel1 = img1.get_pixel(x, y);
            let pixel2 = img2.get_pixel(x, y);

            // Skip black pixels (consider them as invalid regions)
            if pixel1[0] != 0
                || pixel1[1] != 0
                || pixel1[2] != 0
                || pixel2[0] != 0
                || pixel2[1] != 0
                || pixel2[2] != 0
            {
                for c in 0..3 {
                    let diff = pixel1[c] as f64 - pixel2[c] as f64;
                    mse += diff * diff;
                }
                valid_pixels += 3; // 3 channels
            }
        }
    }

    if valid_pixels == 0 {
        return Ok(f64::INFINITY); // Perfect match for empty images
    }

    mse /= valid_pixels as f64;

    if mse <= 1e-10 {
        Ok(f64::INFINITY) // Perfect match
    } else {
        Ok(10.0 * (255.0 * 255.0 / mse).log10())
    }
}

/// Calculate Structural Similarity Index (SSIM) between two images
///
/// # Arguments
///
/// * `img1` - First image
/// * `img2` - Second image
///
/// # Returns
///
/// * `Result<f64, UtilError>` - SSIM value between 0 and 1
pub fn calculate_ssim(img1: &RgbImage, img2: &RgbImage) -> Result<f64, UtilError> {
    if img1.dimensions() != img2.dimensions() {
        return Err(UtilError::InvalidParams(
            "Images must have the same dimensions".to_string(),
        ));
    }

    // Convert to grayscale for SSIM calculation
    let gray1 = rgb_to_grayscale(img1);
    let gray2 = rgb_to_grayscale(img2);

    // SSIM constants
    let c1 = (0.01 * 255.0_f64).powi(2);
    let c2 = (0.03 * 255.0_f64).powi(2);

    let (width, height) = gray1.dimensions();
    let mut sum1 = 0.0;
    let mut count = 0;

    // Simple sliding window SSIM calculation
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mut local_sum1 = 0.0;
            let mut local_sum2 = 0.0;
            let mut local_count = 0;

            // 3x3 window
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = (x as i32 + dx) as u32;
                    let ny = (y as i32 + dy) as u32;

                    let val1 = gray1.get_pixel(nx, ny)[0] as f64;
                    let val2 = gray2.get_pixel(nx, ny)[0] as f64;

                    local_sum1 += val1;
                    local_sum2 += val2;
                    local_count += 1;
                }
            }

            if local_count > 0 {
                let mu1 = local_sum1 / local_count as f64;
                let mu2 = local_sum2 / local_count as f64;

                let mut sigma1_sq = 0.0;
                let mut sigma2_sq = 0.0;
                let mut sigma12 = 0.0;

                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let nx = (x as i32 + dx) as u32;
                        let ny = (y as i32 + dy) as u32;

                        let val1 = gray1.get_pixel(nx, ny)[0] as f64;
                        let val2 = gray2.get_pixel(nx, ny)[0] as f64;

                        sigma1_sq += (val1 - mu1).powi(2);
                        sigma2_sq += (val2 - mu2).powi(2);
                        sigma12 += (val1 - mu1) * (val2 - mu2);
                    }
                }

                sigma1_sq /= (local_count - 1) as f64;
                sigma2_sq /= (local_count - 1) as f64;
                sigma12 /= (local_count - 1) as f64;

                let numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2);
                let denominator = (mu1.powi(2) + mu2.powi(2) + c1) * (sigma1_sq + sigma2_sq + c2);

                if denominator > 0.0 {
                    sum1 += numerator / denominator;
                    count += 1;
                }
            }
        }
    }

    if count > 0 {
        Ok(sum1 / count as f64)
    } else {
        Ok(1.0) // Perfect similarity for empty images
    }
}

/// Convert RGB image to grayscale
fn rgb_to_grayscale(img: &RgbImage) -> GrayImage {
    let (width, height) = img.dimensions();
    let mut gray_img = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let gray_value =
                (0.299 * pixel[0] as f64 + 0.587 * pixel[1] as f64 + 0.114 * pixel[2] as f64) as u8;
            gray_img.put_pixel(x, y, image::Luma([gray_value]));
        }
    }

    gray_img
}

/// Load an image from file path
///
/// # Arguments
///
/// * `image_path` - Path to the image file
///
/// # Returns
///
/// * `Result<RgbImage, UtilError>` - Loaded RGB image
pub fn load_image(image_path: &str) -> Result<RgbImage, UtilError> {
    let img = image::open(image_path)
        .map_err(|e| UtilError::InvalidParams(format!("Failed to load image: {e}")))?;

    Ok(img.to_rgb8())
}

/// Export point correspondences to CSV and Rust format files
///
/// # Arguments
///
/// * `points_3d` - Matrix of 3D points
/// * `points_2d` - Matrix of corresponding 2D points
/// * `filename_prefix` - Prefix for output files
///
/// # Returns
///
/// * `Result<(), UtilError>` - Success or error
pub fn export_point_correspondences(
    points_3d: &Matrix3xX<f64>,
    points_2d: &Matrix2xX<f64>,
    filename_prefix: &str,
) -> Result<(), UtilError> {
    println!("\n💾 Exporting Point Correspondences:");

    if points_3d.ncols() != points_2d.ncols() {
        return Err(UtilError::InvalidParams(
            "3D and 2D point counts must match".to_string(),
        ));
    }

    // Ensure output directory exists
    ensure_output_dir()?;

    // Export to CSV format
    let csv_filename = format!("output/{filename_prefix}.csv");
    let mut csv_file = File::create(&csv_filename)
        .map_err(|e| UtilError::NumericalError(format!("Failed to create CSV file: {e}")))?;

    writeln!(
        csv_file,
        "# 3D-2D Point Correspondences from Rust Implementation"
    )?;
    writeln!(csv_file, "# Format: x3d,y3d,z3d,x2d,y2d")?;
    writeln!(csv_file, "# Total points: {}", points_3d.ncols())?;

    for i in 0..points_3d.ncols() {
        let p3d = points_3d.column(i);
        let p2d = points_2d.column(i);
        writeln!(
            csv_file,
            "{:.15},{:.15},{:.15},{:.15},{:.15}",
            p3d[0], p3d[1], p3d[2], p2d[0], p2d[1]
        )?;
    }

    // Export to Rust format
    let rust_filename = format!("output/{filename_prefix}_rust.txt");
    let mut rust_file = File::create(&rust_filename)
        .map_err(|e| UtilError::NumericalError(format!("Failed to create Rust file: {e}")))?;

    writeln!(rust_file, "// 3D-2D Point Correspondences for Rust Import")?;
    writeln!(rust_file, "// Generated from Rust fisheye-tools")?;
    writeln!(rust_file, "let points_3d = Matrix3xX::from_columns(&[")?;

    for i in 0..points_3d.ncols() {
        let p3d = points_3d.column(i);
        write!(
            rust_file,
            "    Vector3::new({:.15}, {:.15}, {:.15})",
            p3d[0], p3d[1], p3d[2]
        )?;
        if i < points_3d.ncols() - 1 {
            writeln!(rust_file, ",")?;
        } else {
            writeln!(rust_file)?;
        }
    }
    writeln!(rust_file, "]);")?;
    writeln!(rust_file)?;

    writeln!(rust_file, "let points_2d = Matrix2xX::from_columns(&[")?;
    for i in 0..points_2d.ncols() {
        let p2d = points_2d.column(i);
        write!(
            rust_file,
            "    Vector2::new({:.15}, {:.15})",
            p2d[0], p2d[1]
        )?;
        if i < points_2d.ncols() - 1 {
            writeln!(rust_file, ",")?;
        } else {
            writeln!(rust_file)?;
        }
    }
    writeln!(rust_file, "]);")?;

    println!("Exported {} point correspondences to:", points_3d.ncols());
    println!("  - {csv_filename} (CSV format)");
    println!("  - {rust_filename} (Rust code format)");

    Ok(())
}

/// Validate conversion accuracy between concrete and trait object camera models
///
/// # Arguments
///
/// * `output_model` - Target camera model (concrete type)
/// * `input_model` - Source camera model (trait object)
///
/// # Returns
///
/// * `Result<ValidationResults, UtilError>` - Validation results
pub fn validate_conversion_accuracy<T>(
    output_model: &T,
    input_model: &dyn CameraModel,
) -> Result<ValidationResults, UtilError>
where
    T: CameraModel,
{
    // Get image resolution from input model
    let resolution = input_model.get_resolution();
    let width = resolution.width as f64;
    let height = resolution.height as f64;

    // Define test points based on image resolution as fractions
    let test_regions = [
        ("Center", Vector2::new(width * 0.5, height * 0.5)),
        ("Near Center", Vector2::new(width * 0.55, height * 0.55)),
        ("Mid Region", Vector2::new(width * 0.65, height * 0.65)),
        ("Edge Region", Vector2::new(width * 0.8, height * 0.8)),
        ("Far Edge", Vector2::new(width * 0.95, height * 0.95)),
    ];

    let mut total_error = 0.0;
    let mut max_error = 0.0;
    let mut valid_projections = 0;
    let mut region_errors = [f64::NAN; 5];
    let mut region_data = Vec::new();

    for (i, (name, test_point_2d)) in test_regions.iter().enumerate() {
        // First unproject the test point using input model to get 3D point
        match input_model.unproject(test_point_2d) {
            Ok(point_3d) => {
                // Now project the 3D point using both models
                match (
                    input_model.project(&point_3d),
                    output_model.project(&point_3d),
                ) {
                    (Ok(input_proj), Ok(output_proj)) => {
                        let error = (input_proj - output_proj).norm();
                        total_error += error;
                        max_error = f64::max(max_error, error);
                        valid_projections += 1;
                        region_errors[i] = error;

                        // Store the projection data
                        region_data.push(RegionValidation {
                            name: name.to_string(),
                            input_projection: Some((input_proj.x, input_proj.y)),
                            output_projection: Some((output_proj.x, output_proj.y)),
                            error,
                        });
                    }
                    (Ok(_), Err(_)) => {
                        region_errors[i] = f64::NAN;
                        region_data.push(RegionValidation {
                            name: name.to_string(),
                            input_projection: None,
                            output_projection: None,
                            error: f64::NAN,
                        });
                    }
                    (Err(_), Ok(_)) => {
                        region_errors[i] = f64::NAN;
                        region_data.push(RegionValidation {
                            name: name.to_string(),
                            input_projection: None,
                            output_projection: None,
                            error: f64::NAN,
                        });
                    }
                    (Err(_), Err(_)) => {
                        region_errors[i] = f64::NAN;
                        region_data.push(RegionValidation {
                            name: name.to_string(),
                            input_projection: None,
                            output_projection: None,
                            error: f64::NAN,
                        });
                    }
                }
            }
            Err(_) => {
                // If unprojection fails, skip this region
                region_errors[i] = f64::NAN;
                region_data.push(RegionValidation {
                    name: name.to_string(),
                    input_projection: None,
                    output_projection: None,
                    error: f64::NAN,
                });
            }
        }
    }

    let average_error = if valid_projections > 0 {
        total_error / valid_projections as f64
    } else {
        f64::NAN
    };

    let status = if average_error.is_nan() {
        "NEEDS IMPROVEMENT".to_string()
    } else if average_error < 0.001 {
        "EXCELLENT".to_string()
    } else if average_error < 0.1 {
        "GOOD".to_string()
    } else {
        "NEEDS IMPROVEMENT".to_string()
    };

    Ok(ValidationResults {
        center_error: region_errors[0],
        near_center_error: region_errors[1],
        mid_region_error: region_errors[2],
        edge_region_error: region_errors[3],
        far_edge_error: region_errors[4],
        average_error,
        max_error,
        status,
        region_data,
    })
}

/// Assess image quality between different camera model projections
///
/// # Arguments
///
/// * `input_model` - Source camera model
/// * `output_model` - Target camera model
/// * `image_path` - Path to reference image
///
/// # Returns
///
/// Compute image quality metrics (PSNR and SSIM) by comparing projections from input and output models
///
/// # Arguments
///
/// * `input_model` - Input camera model
/// * `output_model` - Output camera model  
/// * `optimization_points_3d` - The 3D points used in optimization
/// * `output_model_name` - Name of the output model for file naming
/// * `reference_image` - Optional reference image for coloring pixels
///
/// # Returns
///
/// * `Result<ImageQualityMetrics, UtilError>` - PSNR and SSIM values
pub fn compute_image_quality_metrics(
    input_model: &dyn CameraModel,
    output_model: &dyn CameraModel,
    optimization_points_3d: &Matrix3xX<f64>,
    output_model_name: &str,
    reference_image: Option<&RgbImage>,
) -> Result<ImageQualityMetrics, UtilError> {
    // Determine image dimensions - prefer reference image size if available
    let (width, height) = if let Some(ref_img) = reference_image {
        (ref_img.width(), ref_img.height())
    } else {
        let resolution = input_model.get_resolution();
        (resolution.width, resolution.height)
    };

    // Project the optimization 3D points using both models
    let mut input_projections = Vec::new();
    let mut output_projections = Vec::new();
    let mut valid_colors = Vec::new();

    for col_idx in 0..optimization_points_3d.ncols() {
        let point_3d = optimization_points_3d.column(col_idx);

        // Project using both models
        if let (Ok(input_proj), Ok(output_proj)) = (
            input_model.project(&point_3d.into()),
            output_model.project(&point_3d.into()),
        ) {
            // Check if output projection is within image bounds
            if output_proj.x >= 0.0
                && output_proj.x < width as f64
                && output_proj.y >= 0.0
                && output_proj.y < height as f64
            {
                input_projections.push(input_proj);
                output_projections.push(output_proj);

                // Colors will be determined when drawing projections
                valid_colors.push(Rgb([255, 255, 255]));
            }
        }
    }

    if output_projections.is_empty() {
        return Err(UtilError::ZeroProjectionPoints);
    }

    // Create projection image with both input and output projections for comparison
    let combined_display_image = if let Some(ref_img) = reference_image {
        create_combined_projection_image_on_reference(
            &input_projections,
            &output_projections,
            ref_img,
        )?
    } else {
        create_combined_projection_image(&input_projections, &output_projections, width, height)?
    };

    // Save the combined projection image
    save_model_projection_image(&combined_display_image, output_model_name)?;

    // Create black background images for PSNR/SSIM computation (not saved)
    let input_comp_image =
        create_projection_image(&input_projections, &valid_colors, width, height)?;
    let output_comp_image =
        create_projection_image(&output_projections, &valid_colors, width, height)?;

    // Calculate quality metrics using black background comparison images
    let psnr = calculate_psnr(&input_comp_image, &output_comp_image)?;
    let ssim = calculate_ssim(&input_comp_image, &output_comp_image)?;

    Ok(ImageQualityMetrics { psnr, ssim })
}

/// Create an image with projected points drawn as colored circles
///
/// # Arguments
///
/// * `projections` - Vector of 2D projection points
/// * `colors` - Vector of colors for each point
/// * `width` - Image width
/// * `height` - Image height
///
/// # Returns
///
/// * `Result<RgbImage, UtilError>` - Generated image
pub fn create_projection_image(
    projections: &[Vector2<f64>],
    colors: &[Rgb<u8>],
    width: u32,
    height: u32,
) -> Result<RgbImage, UtilError> {
    let mut img = RgbImage::new(width, height);

    // Fill with black background
    for pixel in img.pixels_mut() {
        *pixel = Rgb([0, 0, 0]);
    }

    // Draw each projection as a small colored circle
    for (projection, color) in projections.iter().zip(colors.iter()) {
        let center_x = projection.x.round() as i32;
        let center_y = projection.y.round() as i32;

        // Draw a small circle (radius 2)
        let radius = 2;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    let x = center_x + dx;
                    let y = center_y + dy;

                    if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                        img.put_pixel(x as u32, y as u32, *color);
                    }
                }
            }
        }
    }

    Ok(img)
}

/// Create an image with both input and output projected points drawn as colored circles on a reference image
///
/// # Arguments
///
/// * `input_projections` - Vector of 2D projection points from input model
/// * `output_projections` - Vector of 2D projection points from output model
/// * `reference_image` - Reference image to draw points on
///
/// # Returns
///
/// * `Result<RgbImage, UtilError>` - Generated image with both projections
pub fn create_combined_projection_image_on_reference(
    input_projections: &[Vector2<f64>],
    output_projections: &[Vector2<f64>],
    reference_image: &RgbImage,
) -> Result<RgbImage, UtilError> {
    let mut img = reference_image.clone();

    // Draw input projections as cyan circles
    for projection in input_projections.iter() {
        let center_x = projection.x.round() as i32;
        let center_y = projection.y.round() as i32;

        // Draw a small circle (radius 2)
        let radius = 2;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    let x = center_x + dx;
                    let y = center_y + dy;

                    if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
                        img.put_pixel(x as u32, y as u32, Rgb([0, 255, 0])); // Cyan for input
                    }
                }
            }
        }
    }

    // Draw output projections as magenta circles
    for projection in output_projections.iter() {
        let center_x = projection.x.round() as i32;
        let center_y = projection.y.round() as i32;

        // Draw a small circle (radius 2)
        let radius = 2;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    let x = center_x + dx;
                    let y = center_y + dy;

                    if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
                        img.put_pixel(x as u32, y as u32, Rgb([255, 0, 255])); // Magenta for output
                    }
                }
            }
        }
    }

    Ok(img)
}

/// Create an image with both input and output projected points drawn as colored circles on black background
///
/// # Arguments
///
/// * `input_projections` - Vector of 2D projection points from input model
/// * `output_projections` - Vector of 2D projection points from output model
/// * `width` - Image width
/// * `height` - Image height
///
/// # Returns
///
/// * `Result<RgbImage, UtilError>` - Generated image with both projections
pub fn create_combined_projection_image(
    input_projections: &[Vector2<f64>],
    output_projections: &[Vector2<f64>],
    width: u32,
    height: u32,
) -> Result<RgbImage, UtilError> {
    let mut img = RgbImage::new(width, height);

    // Fill with black background
    for pixel in img.pixels_mut() {
        *pixel = Rgb([0, 0, 0]);
    }

    // Draw input projections as cyan circles
    for projection in input_projections.iter() {
        let center_x = projection.x.round() as i32;
        let center_y = projection.y.round() as i32;

        // Draw a small circle (radius 2)
        let radius = 2;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    let x = center_x + dx;
                    let y = center_y + dy;

                    if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                        img.put_pixel(x as u32, y as u32, Rgb([0, 255, 0])); // Green for input
                    }
                }
            }
        }
    }

    // Draw output projections as magenta circles
    for projection in output_projections.iter() {
        let center_x = projection.x.round() as i32;
        let center_y = projection.y.round() as i32;

        // Draw a small circle (radius 2)
        let radius = 2;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    let x = center_x + dx;
                    let y = center_y + dy;

                    if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                        img.put_pixel(x as u32, y as u32, Rgb([255, 0, 255])); // Magenta for output
                    }
                }
            }
        }
    }

    Ok(img)
}

/// Save model projection image to the output directory with model-specific naming
///
/// # Arguments
///
/// * `image` - Image to save
/// * `model_name` - Name of the model for filename
///
/// # Returns
///
/// * `Result<(), UtilError>` - Success or error
pub fn save_model_projection_image(image: &RgbImage, model_name: &str) -> Result<(), UtilError> {
    ensure_output_dir()?;

    // Convert model name to lowercase and replace spaces with underscores
    let filename_safe_name = model_name.to_lowercase().replace(' ', "_");
    let filename = format!("output/{filename_safe_name}_projection.png");

    image
        .save(&filename)
        .map_err(|e| UtilError::InvalidParams(format!("Failed to save projection image: {e}")))?;

    println!("📸 Saved projection image: {filename}");
    Ok(())
}

/// Display input model parameters
///
/// # Arguments
///
/// * `model_type` - Type of the camera model
/// * `intrinsics` - Camera intrinsics
/// * `distortion` - Distortion parameters
pub fn display_input_model_parameters(model_type: &str, camera_model: &dyn CameraModel) {
    println!("📷 Input Model Parameters:");
    let intrinsics = camera_model.get_intrinsics();
    let distortion = camera_model.get_distortion();

    match model_type.to_lowercase().as_str() {
        "ds" | "double_sphere" => {
            println!(
                "DS parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, alpha={:.6}, xi={:.6}",
                intrinsics.fx,
                intrinsics.fy,
                intrinsics.cx,
                intrinsics.cy,
                distortion[0],
                distortion[1]
            );
        }
        "kb" | "kannala_brandt" => {
            println!(
                "KB parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, k1={:.6}, k2={:.6}, k3={:.6}, k4={:.6}",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                distortion[0], distortion[1], distortion[2], distortion[3]
            );
        }
        "radtan" | "rad_tan" => {
            println!(
                "RadTan parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, k1={:.6}, k2={:.6}, p1={:.6}, p2={:.6}, k3={:.6}",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                distortion[0], distortion[1], distortion[2], distortion[3], distortion[4]
            );
        }
        "ucm" | "unified" => {
            println!(
                "UCM parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, alpha={:.6}",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, distortion[0]
            );
        }
        "eucm" | "extended_unified" => {
            println!(
                "EUCM parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, alpha={:.6}, beta={:.6}",
                intrinsics.fx,
                intrinsics.fy,
                intrinsics.cx,
                intrinsics.cy,
                distortion[0],
                distortion[1]
            );
        }
        "pinhole" => {
            println!(
                "Pinhole parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
            );
        }
        _ => {
            println!(
                "Model parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
            );
        }
    }
}

/// Display detailed results for a conversion
///
/// # Arguments
///
/// * `metrics` - Conversion metrics containing all the results
pub fn display_detailed_results(metrics: &ConversionMetrics) {
    println!("\n📊 Final Output Model Parameters:");
    println!("{:?}", metrics.model);
    println!("computing time(ms): {:.0}", metrics.optimization_time_ms);

    println!("\n🧪 EVALUATION AND VALIDATION:");
    println!("=============================");

    // Print detailed reprojection error information
    let final_error = &metrics.final_reprojection_error;
    println!("Reprojection Error Statistics:");
    println!("  Mean: {:.8} px", final_error.mean);
    println!("  RMSE: {:.8} px", final_error.rmse);
    println!("  Min: {:.8} px", final_error.min);
    println!("  Max: {:.8} px", final_error.max);
    println!("  Std Dev: {:.8} px", final_error.stddev);
    println!("  Median: {:.8} px", final_error.median);

    // Print validation results with actual calculated values
    let validation = &metrics.validation_results;
    println!("\n🎯 Conversion Accuracy Validation:");

    let region_names = [
        "Center",
        "Near Center",
        "Mid Region",
        "Edge Region",
        "Far Edge",
    ];
    let region_errors = [
        validation.center_error,
        validation.near_center_error,
        validation.mid_region_error,
        validation.edge_region_error,
        validation.far_edge_error,
    ];

    for (i, (name, error)) in region_names.iter().zip(region_errors.iter()).enumerate() {
        if !error.is_nan() {
            // Use actual projected coordinates if available from region_data
            if i < validation.region_data.len() {
                let region = &validation.region_data[i];
                if let (Some(input_coords), Some(output_coords)) =
                    (&region.input_projection, &region.output_projection)
                {
                    println!(
                        "  {}: Input({:.2}, {:.2}) → Output({:.2}, {:.2}) | Error: {:.4} px",
                        name,
                        input_coords.0,
                        input_coords.1,
                        output_coords.0,
                        output_coords.1,
                        error
                    );
                } else {
                    println!("  {name}: Projection failed");
                }
            } else {
                // Fallback for when region_data is not populated
                println!("  {name}: Error: {error:.4} px");
            }
        } else {
            println!("  {name}: Projection failed");
        }
    }

    println!(
        "  📈 Average Error: {:.4} px, Max Error: {:.4} px",
        validation.average_error, validation.max_error
    );

    if validation.status == "EXCELLENT" {
        println!("  ✅ Conversion Accuracy: {}", validation.status);
    } else {
        println!("  ⚠️  Conversion Accuracy: {}", validation.status);
    }

    // Print image quality metrics if available
    if let Some(ref image_quality) = metrics.image_quality {
        println!("\n📸 Image Quality Assessment:");
        println!("  PSNR: {:.2} dB", image_quality.psnr);
        println!("  SSIM: {:.4}", image_quality.ssim);
    }
}

/// Export conversion results to a detailed text file
///
/// # Arguments
///
/// * `metrics` - Vector of conversion metrics for all models
/// * `input_model_type` - Type of input model
///
/// # Returns
///
/// * `Result<(), UtilError>` - Success or error
pub fn export_conversion_results(
    metrics: &[ConversionMetrics],
    input_model_type: &str,
) -> Result<(), UtilError> {
    ensure_output_dir()?;

    let report_filename = format!(
        "output/camera_conversion_results_{}.txt",
        input_model_type.to_lowercase()
    );
    let mut report_file = File::create(&report_filename)
        .map_err(|e| UtilError::NumericalError(format!("Failed to create report file: {e}")))?;

    writeln!(
        report_file,
        "FISHEYE CAMERA MODEL CONVERSION ANALYSIS REPORT - RUST IMPLEMENTATION"
    )?;
    writeln!(
        report_file,
        "====================================================================="
    )?;
    writeln!(report_file)?;
    writeln!(
        report_file,
        "INPUT MODEL TYPE: {}",
        input_model_type.to_uppercase()
    )?;
    writeln!(report_file, "OPTIMIZATION FRAMEWORK: tiny-solver")?;
    writeln!(report_file, "ALGORITHM: Levenberg-Marquardt")?;
    writeln!(report_file)?;

    if metrics.is_empty() {
        writeln!(report_file, "❌ No conversions performed (input model type not supported for conversion or no target models available)")?;
        return Ok(());
    }

    // Export detailed results table
    writeln!(report_file, "CONVERSION RESULTS TABLE")?;
    writeln!(report_file, "========================")?;
    writeln!(
        report_file,
        "{:<32} | {:>15} | {:>15} | {:>13} | {:>15}",
        "Target Model", "Final Error", "Improvement", "Time (ms)", "Convergence"
    )?;
    writeln!(
        report_file,
        "{:<32} | {:>15} | {:>15} | {:>13} | {:>15}",
        "", "(pixels)", "(pixels)", "", "Status"
    )?;
    writeln!(
        report_file,
        "{:-<32}-+-{:-<15}-+-{:-<15}-+-{:-<13}-+-{:-<15}",
        "", "", "", "", ""
    )?;

    for metric in metrics {
        let improvement =
            metric.initial_reprojection_error.mean - metric.final_reprojection_error.mean;
        writeln!(
            report_file,
            "{:<32} | {:>13.6}   | {:>13.6}   | {:>11.2}   | {:<15}",
            metric.model_name,
            metric.final_reprojection_error.mean,
            improvement,
            metric.optimization_time_ms,
            metric.convergence_status
        )?;
    }
    writeln!(report_file)?;

    // Performance analysis
    writeln!(report_file, "PERFORMANCE ANALYSIS")?;
    writeln!(report_file, "====================")?;

    let best_accuracy = metrics.iter().min_by(|a, b| {
        a.final_reprojection_error
            .mean
            .partial_cmp(&b.final_reprojection_error.mean)
            .unwrap()
    });
    let fastest_conversion = metrics.iter().min_by(|a, b| {
        a.optimization_time_ms
            .partial_cmp(&b.optimization_time_ms)
            .unwrap()
    });

    if let Some(best) = best_accuracy {
        writeln!(
            report_file,
            "🏆 Best Accuracy: {} ({:.6} pixels)",
            best.model_name, best.final_reprojection_error.mean
        )?;
    }
    if let Some(fastest) = fastest_conversion {
        writeln!(
            report_file,
            "⚡ Fastest Conversion: {} ({:.2} ms)",
            fastest.model_name, fastest.optimization_time_ms
        )?;
    }

    let avg_error = metrics
        .iter()
        .map(|m| m.final_reprojection_error.mean)
        .sum::<f64>()
        / metrics.len() as f64;
    let avg_time =
        metrics.iter().map(|m| m.optimization_time_ms).sum::<f64>() / metrics.len() as f64;

    writeln!(
        report_file,
        "📊 Average Reprojection Error: {avg_error:.6} pixels"
    )?;
    writeln!(
        report_file,
        "📊 Average Optimization Time: {avg_time:.2} ms"
    )?;
    writeln!(report_file)?;

    // Detailed metrics for each model
    writeln!(report_file, "DETAILED MODEL RESULTS")?;
    writeln!(report_file, "======================")?;

    for metric in metrics {
        writeln!(report_file, "\n{} MODEL:", metric.model_name.to_uppercase())?;
        writeln!(report_file, "{}", "-".repeat(metric.model_name.len() + 7))?;
        writeln!(report_file, "Final Parameters: {:?}", metric.model)?;
        writeln!(
            report_file,
            "Optimization Time: {:.2} ms",
            metric.optimization_time_ms
        )?;
        writeln!(
            report_file,
            "Convergence Status: {}",
            metric.convergence_status
        )?;

        writeln!(report_file, "\nReprojection Error Statistics:")?;
        writeln!(
            report_file,
            "  Mean: {:.8} px",
            metric.final_reprojection_error.mean
        )?;
        writeln!(
            report_file,
            "  RMSE: {:.8} px",
            metric.final_reprojection_error.rmse
        )?;
        writeln!(
            report_file,
            "  Min: {:.8} px",
            metric.final_reprojection_error.min
        )?;
        writeln!(
            report_file,
            "  Max: {:.8} px",
            metric.final_reprojection_error.max
        )?;
        writeln!(
            report_file,
            "  Std Dev: {:.8} px",
            metric.final_reprojection_error.stddev
        )?;
        writeln!(
            report_file,
            "  Median: {:.8} px",
            metric.final_reprojection_error.median
        )?;

        writeln!(report_file, "\nConversion Accuracy:")?;
        let validation = &metric.validation_results;
        writeln!(
            report_file,
            "  Average Error: {:.4} px",
            validation.average_error
        )?;
        writeln!(report_file, "  Max Error: {:.4} px", validation.max_error)?;
        writeln!(report_file, "  Status: {}", validation.status)?;

        if let Some(ref image_quality) = metric.image_quality {
            writeln!(report_file, "\nImage Quality Assessment:")?;
            writeln!(report_file, "  PSNR: {:.2} dB", image_quality.psnr)?;
            writeln!(report_file, "  SSIM: {:.4}", image_quality.ssim)?;
        }
    }

    Ok(())
}

/// Display results summary table and analysis
///
/// # Arguments
///
/// * `metrics` - Vector of conversion metrics for all models
/// * `input_model_type` - Type of input model
pub fn display_results_summary(metrics: &[ConversionMetrics], input_model_type: &str) {
    // Step 4: Generate comprehensive benchmark report
    println!("\n📊 Step 4: Conversion Results Summary");
    println!("====================================");

    if metrics.is_empty() {
        println!("❌ No conversions performed (input model type not supported for conversion or no target models available)");
        return;
    }

    // Print detailed results table
    println!("\n📋 CONVERSION RESULTS TABLE");
    println!("┌────────────────────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐");
    println!("│ Target Model                   │ Final Error     │ Improvement     │ Time (ms)       │ Convergence     │");
    println!("│                                │ (pixels)        │ (pixels)        │                 │ Status          │");
    println!("├────────────────────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤");

    for metric in metrics {
        let improvement =
            metric.initial_reprojection_error.mean - metric.final_reprojection_error.mean;
        println!(
            "│ {:<30} │ {:>13.6}   │ {:>13.6}   │ {:>13.2}   │ {:<15} │",
            metric.model_name,
            metric.final_reprojection_error.mean,
            improvement,
            metric.optimization_time_ms,
            metric.convergence_status
        );
    }
    println!("└────────────────────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘");

    // Step 5: Performance analysis
    println!("\n📈 Step 5: Performance Analysis");
    println!("===============================");

    let best_accuracy = metrics.iter().min_by(|a, b| {
        a.final_reprojection_error
            .mean
            .partial_cmp(&b.final_reprojection_error.mean)
            .unwrap()
    });
    let fastest_conversion = metrics.iter().min_by(|a, b| {
        a.optimization_time_ms
            .partial_cmp(&b.optimization_time_ms)
            .unwrap()
    });

    if let Some(best) = best_accuracy {
        println!(
            "🏆 Best Accuracy: {} ({:.6} pixels)",
            best.model_name, best.final_reprojection_error.mean
        );
    }
    if let Some(fastest) = fastest_conversion {
        println!(
            "⚡ Fastest Conversion: {} ({:.2} ms)",
            fastest.model_name, fastest.optimization_time_ms
        );
    }

    // Calculate average metrics
    let avg_error = metrics
        .iter()
        .map(|m| m.final_reprojection_error.mean)
        .sum::<f64>()
        / metrics.len() as f64;
    let avg_time =
        metrics.iter().map(|m| m.optimization_time_ms).sum::<f64>() / metrics.len() as f64;

    println!("📊 Average Reprojection Error: {avg_error:.6} pixels");
    println!("📊 Average Optimization Time: {avg_time:.2} ms");

    // Step 6: Export results using util module
    println!("\n💾 Step 6: Exporting Results");
    println!("============================");

    // Export detailed analysis report
    if let Err(e) = export_conversion_results(metrics, input_model_type) {
        println!("⚠️  Failed to export results: {e}");
    } else {
        let report_filename = format!(
            "output/camera_conversion_results_{}.txt",
            input_model_type.to_lowercase()
        );
        println!("📄 Results exported to: {report_filename}");
    }
}

/// Create and save input model projection visualization from 2D points
///
/// This function creates a visualization of 2D projected points, either on a reference image
/// (if provided) or on a black background. Points are drawn as green circles.
///
/// # Arguments
///
/// * `points_2d` - Matrix of 2D projection points
/// * `model_name` - Name of the input model for filename generation
/// * `reference_image` - Optional reference image to project points onto
/// * `camera_resolution` - Camera resolution for image dimensions when no reference image
///
/// # Returns
///
/// * `Result<(), UtilError>` - Success or error
pub fn model_projection_visualization(
    points_2d: &Matrix2xX<f64>,
    model_name: &str,
    reference_image: Option<&RgbImage>,
    camera_resolution: (u32, u32), // (width, height)
) -> Result<(), UtilError> {
    // Convert Matrix2xX to Vec<Vector2<f64>>
    let mut projections = Vec::new();
    for col_idx in 0..points_2d.ncols() {
        let point = points_2d.column(col_idx);
        projections.push(Vector2::new(point[0], point[1]));
    }

    // Create the projection image
    let projection_image = if let Some(ref_img) = reference_image {
        // Project points onto the reference image
        let mut img = ref_img.clone();

        // Draw each projection as a green circle
        for projection in projections.iter() {
            let center_x = projection.x.round() as i32;
            let center_y = projection.y.round() as i32;

            // Draw a small circle (radius 2)
            let radius = 2;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx * dx + dy * dy <= radius * radius {
                        let x = center_x + dx;
                        let y = center_y + dy;

                        if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
                            img.put_pixel(x as u32, y as u32, Rgb([0, 255, 0]));
                            // Green
                        }
                    }
                }
            }
        }
        img
    } else {
        // Create projection on black background
        let green_colors = vec![Rgb([0, 255, 0]); projections.len()];
        create_projection_image(
            &projections,
            &green_colors,
            camera_resolution.0,
            camera_resolution.1,
        )?
    };

    // Save the image
    ensure_output_dir()?;
    let filename = format!("output/{}_projection.png", model_name.to_lowercase());
    projection_image
        .save(&filename)
        .map_err(|e| UtilError::NumericalError(format!("Failed to save projection image: {e}")))?;

    println!("Saved input model projection visualization: {filename}");
    Ok(())
}

impl From<std::io::Error> for UtilError {
    fn from(err: std::io::Error) -> Self {
        UtilError::NumericalError(err.to_string())
    }
}

impl From<CameraModelError> for UtilError {
    fn from(err: CameraModelError) -> Self {
        UtilError::NumericalError(err.to_string())
    }
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
