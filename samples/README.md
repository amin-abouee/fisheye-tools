# Camera Model Sample YAML Files

This directory contains YAML configuration files for different camera models:

- `pinhole.yaml`: Sample parameters for a pinhole camera model
- `rad_tan.yaml`: Sample parameters for a radial-tangential distortion camera model
- `double_sphere.yaml`: Sample parameters for a double sphere fisheye camera model

These files are used for testing the camera model implementations and serve as examples for how to structure camera parameter files.

## Format

Each YAML file follows a common structure:

```yaml
cam0:
  camera_model: <model_type>
  intrinsics: [fx, fy, cx, cy]  # Focal lengths and principal point
  resolution: [width, height]   # Image resolution in pixels
  # Additional parameters specific to each model
```

Additional parameters vary by model type:
- Radial-tangential model includes distortion coefficients
- Double sphere model includes xi and alpha parameters 