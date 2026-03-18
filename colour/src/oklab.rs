use maths::linear_algebra::{Matrix, Matrix3x3, Vector3D};
use maths::traits::Float;

pub struct OkLab<T> {
    linear_srgb_to_oklab_lms: Matrix3x3<T>,
    oklab_lms_to_linear_srgb: Matrix3x3<T>,
    opponency_matrix: Matrix3x3<T>,
    opponency_inverse: Matrix3x3<T>,
}

impl<T: Float> OkLab<T> {
    #[rustfmt::skip]
    pub fn new() -> Self {
        let to_lms = Matrix([
            [T::frac(4122214708, 10000000000), T::frac(5363325363, 10000000000), T::frac( 514459929, 10000000000)],
            [T::frac(2119034982, 10000000000), T::frac(6806995451, 10000000000), T::frac(1073969566, 10000000000)],
            [T::frac(0883024619, 10000000000), T::frac(2817188376, 10000000000), T::frac(6299787005, 10000000000)]
        ]);
        let opponency = Matrix([
            [T::frac( 2104542553, 10000000000), T::frac(  7936177850, 10000000000), T::frac(  -40720468, 10000000000)],
            [T::frac(19779984951, 10000000000), T::frac(-24285922050, 10000000000), T::frac( 4505937099, 10000000000)],
            [T::frac(  259040371, 10000000000), T::frac(  7827717662, 10000000000), T::frac(-8086757660, 10000000000)]
        ]);
        Self {
            linear_srgb_to_oklab_lms: to_lms,
            oklab_lms_to_linear_srgb: to_lms.invert3x3(),
            opponency_matrix: opponency,
            opponency_inverse: opponency.invert3x3(),
        }
    }

    pub fn srgb_to_oklab(&self, rgb: Vector3D<T>) -> Vector3D<T> {
        let lms = self.linear_srgb_to_oklab_lms * rgb;
        let _lms = lms.map(|x| x.cbrt());
        self.opponency_matrix * _lms
    }

    pub fn oklab_to_srgb(&self, oklab: Vector3D<T>) -> Vector3D<T> {
        let _lms = self.opponency_inverse * oklab;
        let lms = _lms.map(|x| x.powi(3));
        self.oklab_lms_to_linear_srgb * lms
    }
}

impl<T> Mappable<T> for OkLab<T> {
    type Wrapped<B> = OkLab<B>;
    #[inline]
    fn fmap<F, B>(self, mut f: F) -> Self::Wrapped<B>
    where
        F: FnMut(T) -> B,
    {
        OkLab {
            linear_srgb_to_oklab_lms: self.linear_srgb_to_oklab_lms.fmap(&mut f),
            oklab_lms_to_linear_srgb: self.oklab_lms_to_linear_srgb.fmap(&mut f),
            opponency_matrix: self.opponency_matrix.fmap(&mut f),
            opponency_inverse: self.opponency_inverse.fmap(&mut f),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Ipt<T> {
    pub power: T,
    pub power_inv: T,
    pub opponency_matrix: Matrix3x3<T>,
    pub opponency_inverse: Matrix3x3<T>,
}

impl<T: Float> Ipt<T> {
    pub const POWER: f64 = 0.43;
    #[inline]
    pub fn new() -> Self {
        let f = T::from_f64;
        let opponency_matrix = Matrix([
            [f(0.6557), f(0.3279), f(0.0164)], /* This is 2+1+0.05 */
            // [f(0.4750), f(0.4750), f(0.0500)],
            // [f(0.4000),f(0.4000),f(0.2000)],
            [f(4.4550), f(-4.8510), f(0.3960)],
            [f(0.8056), f(0.3572), f(-1.1628)],
        ]);
        Self {
            power: f(Self::POWER),
            power_inv: f(1.0 / Self::POWER),
            opponency_matrix,
            opponency_inverse: opponency_matrix.invert3x3(),
        }
    }

    #[inline]
    pub fn lms_to_ipt(&self, lms: Vector3D<T>) -> Vector3D<T> {
        self.opponency_matrix * lms.map(|x| x.powf(self.power))
    }

    #[inline]
    pub fn ipt_to_lms(&self, ipt: Vector3D<T>) -> Vector3D<T> {
        (self.opponency_inverse * ipt).map(|x| x.powf(self.power_inv))
    }
}

use optimisation::mappable::Mappable;
impl<A> Mappable<A> for Ipt<A> {
    type Wrapped<B> = Ipt<B>;
    #[inline]
    fn fmap<F, B>(self, mut f: F) -> Self::Wrapped<B>
    where
        F: FnMut(A) -> B,
    {
        Ipt {
            power: f(self.power),
            power_inv: f(self.power_inv),
            opponency_matrix: self.opponency_matrix.fmap(&mut f),
            opponency_inverse: self.opponency_inverse.fmap(&mut f),
        }
    }
}
