use super::CameraTransform;
use maths::{
    linear_algebra::Vector,
    traits::{Float, One, Zero},
};
use optimisation::mappable::Mappable;

/* https://ieeexplore.ieee.org/document/7047834 */

/****************************** 2nd order ******************************/

#[derive(Copy, Clone, Debug)]
pub struct Rpcc2<T>(pub Vector<T, 6>, pub Vector<T, 6>, pub Vector<T, 6>);

impl<T> Rpcc2<T> {
    #[inline]
    pub fn identity() -> Self
    where
        T: One + Zero,
    {
        Self(Vector::unit(0), Vector::unit(1), Vector::unit(2))
    }

    #[inline]
    pub fn apply(&self, Vector([r, g, b]): Vector<T, 3>) -> Vector<T, 3>
    where
        T: Float + Copy,
    {
        let v = Vector([r, g, b, (r * g).sqrt(), (g * b).sqrt(), (r * b).sqrt()]);
        Vector([v.dot(self.0), v.dot(self.1), v.dot(self.2)])
    }
}

impl<A> Mappable<A> for Rpcc2<A> {
    type Wrapped<B> = Rpcc2<B>;
    #[inline]
    fn fmap<F, B>(self, mut f: F) -> Rpcc2<B>
    where
        F: FnMut(A) -> B,
    {
        Rpcc2(self.0.map(&mut f), self.1.map(&mut f), self.2.map(&mut f))
    }
}

impl<T> CameraTransform<T> for Rpcc2<T>
where
    T: Float + Copy,
{
    #[inline]
    fn apply(&self, rgb: Vector<T, 3>) -> Vector<T, 3> {
        self.apply(rgb)
    }
}

/****************************** 3rd order ******************************/

#[derive(Copy, Clone, Debug)]
pub struct Rpcc3<T>(pub Vector<T, 13>, pub Vector<T, 13>, pub Vector<T, 13>);

impl<T: Zero> From<Rpcc2<T>> for Rpcc3<T> {
    #[inline]
    fn from(Rpcc2(a, b, c): Rpcc2<T>) -> Self {
        Self(a.zero_pad(), b.zero_pad(), c.zero_pad())
    }
}

impl<T> Rpcc3<T> {
    #[inline]
    pub fn identity() -> Self
    where
        T: One + Zero,
    {
        Self(Vector::unit(0), Vector::unit(1), Vector::unit(2))
    }

    #[inline]
    pub fn apply(&self, Vector([r, g, b]): Vector<T, 3>) -> Vector<T, 3>
    where
        T: Float + Copy,
    {
        let v = Vector([
            r,
            g,
            b,
            (r * g).sqrt(),
            (g * b).sqrt(),
            (r * b).sqrt(),
            (r * g.powi(2)).cbrt(),
            (g * b.powi(2)).cbrt(),
            (r * b.powi(2)).cbrt(),
            (g * r.powi(2)).cbrt(),
            (b * g.powi(2)).cbrt(),
            (b * r.powi(2)).cbrt(),
            (r * g * b).cbrt(),
        ]);
        Vector([v.dot(self.0), v.dot(self.1), v.dot(self.2)])
    }
}

impl<A> Mappable<A> for Rpcc3<A> {
    type Wrapped<B> = Rpcc3<B>;
    #[inline]
    fn fmap<F, B>(self, mut f: F) -> Rpcc3<B>
    where
        F: FnMut(A) -> B,
    {
        Rpcc3(self.0.map(&mut f), self.1.map(&mut f), self.2.map(&mut f))
    }
}

impl<T: Float + Copy> CameraTransform<T> for Rpcc3<T> {
    #[inline]
    fn apply(&self, rgb: Vector<T, 3>) -> Vector<T, 3> {
        self.apply(rgb)
    }
}

/****************************** 4th order ******************************/

#[derive(Copy, Clone, Debug)]
pub struct Rpcc4<T>(pub Vector<T, 22>, pub Vector<T, 22>, pub Vector<T, 22>);

impl<T: Zero> From<Rpcc3<T>> for Rpcc4<T> {
    #[inline]
    fn from(Rpcc3(a, b, c): Rpcc3<T>) -> Self {
        Self(a.zero_pad(), b.zero_pad(), c.zero_pad())
    }
}

impl<T> Rpcc4<T> {
    #[inline]
    pub fn identity() -> Self
    where
        T: One + Zero,
    {
        Self(Vector::unit(0), Vector::unit(1), Vector::unit(2))
    }

    #[inline]
    pub fn apply(&self, Vector([r, g, b]): Vector<T, 3>) -> Vector<T, 3>
    where
        T: Float + Copy,
    {
        // let (r, g, b) = (r * self.0[0], g * self.1[1], b * self.2[2]);
        let (r2, g2, b2) = (r.powi(2), g.powi(2), b.powi(2));
        let (r3, g3, b3) = (r.powi(3), g.powi(3), b.powi(3));
        let v = Vector([
            r,
            g,
            b,
            (r * g).sqrt(),
            (g * b).sqrt(),
            (r * b).sqrt(),
            (r * g2).cbrt(),
            (g * b2).cbrt(),
            (r * b2).cbrt(),
            (g * r2).cbrt(),
            (b * g2).cbrt(),
            (b * r2).cbrt(),
            (r * g * b).cbrt(),
            /* 4th: r3g,r3b,g3r, g3b,b3r,b3g, r2gb,g2rb,b2rg */
            (r3 * g).sqrt().sqrt(),
            (r3 * b).sqrt().sqrt(),
            (g3 * r).sqrt().sqrt(),
            (g3 * b).sqrt().sqrt(),
            (b3 * r).sqrt().sqrt(),
            (b3 * g).sqrt().sqrt(),
            (r2 * g * b).sqrt().sqrt(),
            (g2 * r * b).sqrt().sqrt(),
            (b2 * r * g).sqrt().sqrt(),
        ]);
        Vector([v.dot(self.0), v.dot(self.1), v.dot(self.2)])
    }
}

impl<A> Mappable<A> for Rpcc4<A> {
    type Wrapped<B> = Rpcc4<B>;
    #[inline]
    fn fmap<F, B>(self, mut f: F) -> Rpcc4<B>
    where
        F: FnMut(A) -> B,
    {
        Rpcc4(self.0.map(&mut f), self.1.map(&mut f), self.2.map(&mut f))
    }
}

impl<T: Float + Copy> CameraTransform<T> for Rpcc4<T> {
    #[inline]
    fn apply(&self, rgb: Vector<T, 3>) -> Vector<T, 3> {
        self.apply(rgb)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GainedTransform<Rpcc, T> {
    pub rpcc: Rpcc,
    pub gain: Vector<T, 3>,
}

impl<A, Rpcc: Mappable<A>> Mappable<A> for GainedTransform<Rpcc, A> {
    type Wrapped<B> = GainedTransform<Rpcc::Wrapped<B>, B>;
    #[inline]
    fn fmap<F, B>(self, mut f: F) -> Self::Wrapped<B>
    where
        F: FnMut(A) -> B,
    {
        GainedTransform { rpcc: self.rpcc.fmap(&mut f), gain: self.gain.fmap(&mut f) }
    }
}

impl<T, Rpcc> GainedTransform<Rpcc, T> {
    pub fn convert<Rpcc2>(self) -> GainedTransform<Rpcc2, T>
    where
        Rpcc2: From<Rpcc>,
    {
        GainedTransform { rpcc: self.rpcc.into(), gain: self.gain }
    }
}

impl<T, Rpcc> CameraTransform<T> for GainedTransform<Rpcc, T>
where
    Rpcc: CameraTransform<T> + Copy,
    T: Float + Copy,
{
    #[inline]
    fn apply(&self, rgb: Vector<T, 3>) -> Vector<T, 3> {
        self.rpcc.apply(rgb * self.gain)
    }
}
