/* A not very good functor trait (i now call it mappable), inspired by this comment:
 * https://www.reddit.com/r/rust/comments/vog2lc/comment/j2yzwcr/?utm_source=share&utm_medium=web2x&context=3
 * When going A->B->A, rust doesn't know it's the same type based
 * on the trait alone, so I added a method which does't change the type. */
/* Also inspired by https://www.reddit.com/r/rust/comments/10bqmfs/comment/j4col9l/
 * and https://github.com/mtomassoli/HKTs, but those trait bounds still didn't help the compiler
 * realise that (A)->(B)->(A) == (A) */
/* Update: Just saw this exists also: https://github.com/bodil/higher */
pub trait Mappable<A>: Sized {
    type Wrapped<B>: Mappable<B> + Mappable<B, Wrapped<A> = Self> + Mappable<B, Wrapped<B> = Self::Wrapped<B>>;

    fn fmap<F, B>(self, f: F) -> Self::Wrapped<B>
    where
        F: FnMut(A) -> B;

    /* Same as fmap, but when you don't change the inner type
     * (as the rust compiler is too dumb to realise that the
     * type doesn't change, despite the trait bounds I have
     * added to Wrapped above) */
    #[inline]
    fn tmap<F>(self, f: F) -> Self
    where
        F: FnMut(A) -> A,
    {
        unsafe { core::mem::transmute_copy(&self.fmap(f)) }
    }
}

/************** Some implementations of functor for common types **************/

use maths::linear_algebra::{Matrix, Vector};

impl<A> Mappable<A> for () {
    type Wrapped<B> = ();
    #[inline]
    fn fmap<F, B>(self, _f: F) -> () {
        ()
    }
}

impl<A> Mappable<A> for Vec<A> {
    type Wrapped<B> = Vec<B>;
    #[inline]
    fn fmap<F, B>(self, f: F) -> Vec<B>
    where
        F: FnMut(A) -> B,
    {
        self.into_iter().map(f).collect()
    }
}

impl<A, const N: usize> Mappable<A> for Vector<A, N> {
    type Wrapped<B> = Vector<B, N>;
    #[inline]
    fn fmap<F, B>(self, f: F) -> Vector<B, N>
    where
        F: FnMut(A) -> B,
    {
        self.map(f)
    }
}

impl<A, const N: usize> Mappable<A> for [A; N] {
    type Wrapped<B> = [B; N];
    #[inline]
    fn fmap<F, B>(self, f: F) -> [B; N]
    where
        F: FnMut(A) -> B,
    {
        self.map(f)
    }
}

impl<A, const N: usize, const M: usize> Mappable<A> for Matrix<A, N, M> {
    type Wrapped<B> = Matrix<B, N, M>;
    #[inline]
    fn fmap<F, B>(self, f: F) -> Matrix<B, N, M>
    where
        F: FnMut(A) -> B,
    {
        self.map(f)
    }
}

impl<T: Mappable<A>, A> Mappable<A> for Option<T> {
    type Wrapped<B> = Option<T::Wrapped<B>>;
    #[inline]
    fn fmap<F, B>(self, f: F) -> Self::Wrapped<B>
    where
        F: FnMut(A) -> B,
    {
        self.map(|t| t.fmap(f))
    }
}

/* Implement for all tuples up to 12 */
macro_rules! tuple_impls {
    ( $( $name:ident )+ ) => {
        #[allow(non_camel_case_types,non_snake_case)]
        impl<T1, $($name: Mappable<T1>),+> Mappable<T1> for ($($name,)+) {
            type Wrapped<T2> = ($(<$name as Mappable<T1>>::Wrapped<T2>,)+);
            fn fmap<F: FnMut(T1) -> T2, T2>(self, mut f: F) -> Self::Wrapped<T2> {
                let ($($name,)+) = self;
                ($($name.fmap(&mut f),)+)
            }
        }
    };
}

tuple_impls! { a }
tuple_impls! { a b }
tuple_impls! { a b c }
tuple_impls! { a b c d }
tuple_impls! { a b c d e }
tuple_impls! { a b c d e f }
tuple_impls! { a b c d e f g }
tuple_impls! { a b c d e f g h }
tuple_impls! { a b c d e f g h i }
tuple_impls! { a b c d e f g h i j }
tuple_impls! { a b c d e f g h i j k }
tuple_impls! { a b c d e f g h i j k l }
