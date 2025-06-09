//! Fisheye Tools Library
//!
//! A comprehensive Rust library for fisheye camera model conversions and calibration.
//! This library provides implementations of various camera models including:
//! - Pinhole camera model
//! - Radial-Tangential distortion model
//! - Unified Camera Model (UCM)
//! - Extended Unified Camera Model (EUCM)
//! - Double Sphere camera model
//! - Kannala-Brandt camera model
//!
//! The library also includes optimization routines for camera calibration using
//! the tiny-solver optimization framework.

pub mod camera;
pub mod geometry;
pub mod optimization;

// Re-export commonly used types
pub use camera::{
    CameraModel, CameraModelError, DoubleSphereModel, EucmModel, Intrinsics, KannalaBrandtModel,
    PinholeModel, RadTanModel, Resolution, UcmModel,
};

pub use optimization::{
    DoubleSphereOptimizationCost, EucmOptimizationCost, KannalaBrandtOptimizationCost, Optimizer,
    RadTanOptimizationCost, UcmOptimizationCost,
};
