/* Library for building camera transforms. They are intended positive RGB -> Positive RGB
 * (I go for an LMS funamental space, as that encloses the full spectral locus, could also use xyz) */

pub use maths::linear_algebra::{Matrix3x3, Vector};
use maths::traits::Float;
use optimisation::mappable::Mappable;

/* TODO: switch this to R/G/B as separate variables? */
pub trait CameraTransform<T> {
    fn apply(&self, rgb: Vector<T, 3>) -> Vector<T, 3>;
}

#[derive(Debug, Copy, Clone)]
pub struct MatrixTransform<T> {
    pub matrix: Matrix3x3<T>,
}

impl<A> Mappable<A> for MatrixTransform<A> {
    type Wrapped<B> = MatrixTransform<B>;
    #[inline]
    fn fmap<F, B>(self, f: F) -> MatrixTransform<B>
    where
        F: FnMut(A) -> B,
    {
        MatrixTransform {
            matrix: self.matrix.fmap(f),
        }
    }
}

impl<T: Float> CameraTransform<T> for MatrixTransform<T> {
    fn apply(&self, rgb: Vector<T, 3>) -> Vector<T, 3> {
        self.matrix * rgb
    }
}

pub mod root_polynomial;
pub mod utility_transforms;
