use maths::linear_algebra::Vector;
use maths::traits::Float;

#[inline]
pub fn uvy_to_xyy<T: Float>(Vector([u, v, luminance]): Vector<T, 3>) -> Vector<T, 3> {
    let d = (T::int(6) * u) - (T::int(16) * v) + T::int(12);
    Vector([(T::int(9) * u) / d, (T::int(4) * v) / d, luminance])
}

#[inline]
pub fn xyy_to_uvy<T: Float>(Vector([x, y, luminance]): Vector<T, 3>) -> Vector<T, 3> {
    let d = (T::int(-2) * x) + (T::int(12) * y) + T::int(3);
    Vector([(T::int(4) * x) / d, (T::int(9) * y) / d, luminance])
}

#[inline]
pub fn xyz_to_xyy<T: Float>(Vector([x, y, z]): Vector<T, 3>) -> Vector<T, 3> {
    let sum = x + y + z;
    Vector([x / sum, y / sum, y])
}

#[inline]
pub fn xyy_to_xyz<T: Float>(Vector([x, y, luminance]): Vector<T, 3>) -> Vector<T, 3> {
    Vector([x * luminance / y, luminance, (T::one() - x - y) * luminance / y])
}

/******************* Conversions between xy and uv1960 *******************/
#[inline]
pub fn xy_to_uv1960<T: Float>(Vector([x, y]): Vector<T, 2>) -> Vector<T, 2> {
    let denom = T::int(12) * y - T::int(2) * x + T::int(3);
    Vector([T::int(4) * x / denom, T::int(6) * y / denom])
}

#[inline]
pub fn uv1960_to_xy<T: Float>(Vector([u, v]): Vector<T, 2>) -> Vector<T, 2> {
    let denom = T::int(2) * u - T::int(8) * v + T::int(4);
    Vector([(T::int(3) * u) / denom, (T::int(2) * v) / denom])
}
