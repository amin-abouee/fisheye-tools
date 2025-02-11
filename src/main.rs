pub mod camera;

pub mod rad_tan;
// pub mod kannala_brandt;
// pub mod double_sphere;
pub mod pinhole;

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use crate::pinhole::PinholeModel;
    use crate::camera::CameraModel;
    use nalgebra::Point3;

    #[test]
    fn test_pinhole_camera() {
        let model = PinholeModel::load_from_yaml("src/pinhole/pinhole.yaml").unwrap();
        let point_3d = Point3::new(1.0, 1.0, 3.0);
        let point_2d = model.project(&point_3d).unwrap();
        println!("Projected point: {:?}", point_2d);
        assert!(point_2d.x > 0.0);
        assert!(point_2d.y > 0.0);
    }
}
