use super::CameraTransform;
use maths::{linear_algebra::Vector, traits::Float};
use optimisation::mappable::Mappable;

#[derive(Clone, Copy, Debug)]
pub struct Gained<Transform, T> {
    pub trans: Transform,
    pub gain: Vector<T, 3>,
}

impl<A, Transform: Mappable<A>> Mappable<A> for Gained<Transform, A> {
    type Wrapped<B> = Gained<Transform::Wrapped<B>, B>;
    #[inline]
    fn fmap<F, B>(self, mut f: F) -> Self::Wrapped<B>
    where
        F: FnMut(A) -> B,
    {
        Gained {
            trans: self.trans.fmap(&mut f),
            gain: self.gain.fmap(&mut f),
        }
    }
}

impl<T, Transform> Gained<Transform, T> {
    pub fn convert<Transform2>(self) -> Gained<Transform2, T>
    where
        Transform2: From<Transform>,
    {
        Gained {
            trans: self.trans.into(),
            gain: self.gain,
        }
    }
}

impl<T, Transform> CameraTransform<T> for Gained<Transform, T>
where
    Transform: CameraTransform<T> + Copy,
    T: Float + Copy,
{
    #[inline]
    fn apply(&self, rgb: Vector<T, 3>) -> Vector<T, 3> {
        self.trans.apply(rgb * self.gain)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RefinedTransform<Transform> {
    pub trans1: Transform,
    pub trans2: Transform,
}

// impl<T, Transform> RefinedTransform<Transform>
