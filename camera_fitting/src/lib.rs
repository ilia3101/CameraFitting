/* Library for building camera transforms. They are intended positive RGB -> Positive RGB
 * (I go for an LMS funamental space, as that encloses the full spectral locus, could also use xyz) */

pub use maths::linear_algebra::Vector;

/* TODO: switch this to RGB */
pub trait CameraTransform<T> {
    fn apply(&self, rgb: Vector<T, 3>) -> Vector<T, 3>;
}

pub mod root_polynomial;
pub mod utility_transforms;
