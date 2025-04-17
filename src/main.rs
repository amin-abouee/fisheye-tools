pub mod camera;

pub mod pinhole;
pub mod rad_tan;
// pub mod kannala_brandt;
// pub mod double_sphere;

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use crate::camera::CameraModel;
    use crate::pinhole::PinholeModel;
    use crate::rad_tan::RadTanModel;
    use nalgebra::Point3;

    #[test]
    fn test_pinhole_camera() {
        let model = PinholeModel::load_from_yaml("src/pinhole/pinhole.yaml").unwrap();
        let point_3d = Point3::new(1.0, 1.0, 3.0);
        let point_2d = model.project(&point_3d).unwrap();
        assert!(point_2d.x > 0.0);
        assert!(point_2d.y > 0.0);
    }

    #[test]
    fn test_radtan_camera() {
        let model = RadTanModel::load_from_yaml("src/rad_tan/radtan.yaml").unwrap();
        let point_3d = Point3::new(1.0, 1.0, 3.0);
        let point_2d = model.project(&point_3d).unwrap();
        assert!(point_2d.x > 0.0);
        assert!(point_2d.y > 0.0);
    }
}
