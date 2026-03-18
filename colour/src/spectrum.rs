use core::ops::{Add, Div, Index, Mul};
use maths::linear_algebra::Vector;
use maths::traits::{Float, Zero};

pub trait Spectrum<T> {
    /** Return value at wavelength */
    fn get(&self, wavelength: T) -> T;

    /** Multiplies two spectra */
    #[inline]
    fn discretise<const N: usize, const WL_FROM: u32, const STEP: u32>(self) -> DiscreteSpectrum<T, N, WL_FROM, STEP>
    where
        Self: Sized,
        T: Float,
    {
        DiscreteSpectrum::from_fn(|wl| self.get(wl))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GaussianSpectrum<T> {
    pub cwl: T,
    pub fwhm: T,
}

impl<T: Float> Spectrum<T> for GaussianSpectrum<T> {
    #[inline]
    fn get(&self, wl: T) -> T {
        // Avoid division by zero or NaN if fwhm == 0
        if self.fwhm == T::zero() {
            return if wl == self.cwl { T::one() } else { T::zero() };
        }

        let diff = wl - self.cwl;
        // Compute exponent: -4 * ln(2) * (diff^2) / (fwhm^2)
        let exponent = -(T::int(4) * T::int(2).ln()) * diff.powi(2) / (self.fwhm * self.fwhm);
        // let norm = (T::int() * 2.0) / self.fwhm * (ln_2 / T::from(PI).unwrap()).sqrt();
        exponent.exp()
    }
}

// Spectrum based on x and y axis data (should be monotonic). TODO: make it generic over containers, perhaps using a len and index trait.
pub struct XySpectrum<T>(Vec<T>, Vec<T>);

impl<T> XySpectrum<T> {
    pub fn new(data_x: Vec<T>, data_y: Vec<T>) -> Option<Self> {
        (data_x.len() == data_y.len()).then_some(Self(data_x, data_y))
    }
}

impl<T: Float> Spectrum<T> for XySpectrum<T> {
    #[inline]
    fn get(&self, wl: T) -> T {
        let (x, y) = (&self.0, &self.1);
        let pos = x.partition_point(|&x| x < wl);
        if pos == 0 {
            if wl < x[0] {
                return T::zero();
            } else {
                return y[0];
            }
        } else if pos == x.len() {
            if wl > x[x.len() - 1] {
                return T::zero();
            } else {
                return y[pos - 1];
            }
        } else if x[pos] == wl {
            return y[pos];
        } else {
            let x1 = pos - 1;
            let x2 = pos;
            let a = (wl - x[x1]) / (x[x2] - x[x1]);
            return y[x1] * (T::one() - a) + y[x2] * a;
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DiscreteSpectrum<T, const N: usize, const WL_FROM: u32, const STEP: u32> {
    pub data: Vector<T, N>,
}

#[cfg_attr(rustfmt, rustfmt_skip)]
impl<T, const N: usize, const WL_FROM: u32, const STEP: u32> DiscreteSpectrum<T, N, WL_FROM, STEP>
{
    #[inline]
    pub fn map<U>(self, f: impl FnMut(T) -> U) -> DiscreteSpectrum<U, N, WL_FROM, STEP> {
        DiscreteSpectrum { data: self.data.map(f) }
    }

    /** Initialise spectrum from wavelengths using a function */
    #[inline]
    pub fn from_fn(mut f: impl FnMut(T) -> T) -> Self where T: Float {
        DiscreteSpectrum { data: Vector::from_fn(|i| f(T::int((WL_FROM + i as u32 * STEP) as i32))) }
    }

    #[inline]
    pub fn max(&self) -> Option<T> where T: Zero + PartialOrd + Clone {
        self.data.0.iter().reduce(|a,b| if b > a {b} else {a}).map(|x| x.clone())
    }

    #[inline]
    pub fn normalise(&self) -> Option<Self> where T: Float {
        self.max().map(|max| self.map(|x| x / max))
    }

    #[inline]
    pub fn min(&self) -> Option<T> where T: Zero + PartialOrd + Clone {
        self.data.0.iter().reduce(|a,b| if b < a {b} else {a}).map(|x| x.clone())
    }

    #[inline]
    pub fn sum(&self) -> T where T: Zero + Add<Output=T> + Clone {
        self.data.0.iter().cloned().fold(T::zero(), |a,b| a + b)
    }

    //TODO: sort out the mess of optional returns, make them all non optional like this maybe
    #[inline]
    pub fn normalise_area(&self, target_area: T) -> Self where T: Float {
        *self * target_area / self.sum()
    }

    #[inline]
    pub fn peak(&self) -> Option<(T, T)> where T: Float {
        if self.data.0.is_empty() {
            return None;
        }
        let mut max_idx = 0;
        let mut max_val = self.data.0[0];
        for i in 1..self.data.0.len() {
            if self.data.0[i] > max_val {
                max_val = self.data.0[i];
                max_idx = i;
            }
        }
        let wavelength = T::int((WL_FROM + max_idx as u32 * STEP) as i32);
        Some((wavelength, max_val))
    }

    /* TODO: linear interpolation */
    #[inline]
    pub fn get(&self, wavelength: T) -> T where T: Float {
        let wl_i = (wavelength.as_i32() - WL_FROM as i32 + (STEP / 2) as i32) / STEP as i32;
        if wl_i as usize >= N { return T::zero() }
        else if wl_i < 0 { return T::zero() }
        else { self.data[wl_i as usize] }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
impl<T, I, const N: usize, const A: u32, const B: u32> Index<I> for DiscreteSpectrum<T,N,A,B>
  where Vector<T,N>: Index<I> {
    type Output = <Vector<T,N> as Index<I>>::Output;
    #[inline] fn index(&self, i: I) -> &Self::Output { &self.data[i] }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
impl<T: Mul<T,Output=T> + Copy, const N: usize, const A: u32, const B: u32> Mul<Self> for DiscreteSpectrum<T,N,A,B> {
    type Output = Self;
    #[inline] fn mul(self, rhs: Self) -> Self { DiscreteSpectrum { data: self.data * rhs.data } }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
impl<T: Div<T,Output=T> + Copy, const N: usize, const A: u32, const B: u32> Div<Self> for DiscreteSpectrum<T,N,A,B> {
    type Output = Self;
    #[inline] fn div(self, rhs: Self) -> Self { DiscreteSpectrum { data: self.data / rhs.data } }
}
#[cfg_attr(rustfmt, rustfmt_skip)]
impl<T: Add<T,Output=T> + Copy, const N: usize, const A: u32, const B: u32> Add<Self> for DiscreteSpectrum<T,N,A,B> {
    type Output = Self;
    #[inline] fn add(self, rhs: Self) -> Self { DiscreteSpectrum { data: self.data + rhs.data } }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
impl<T: Mul<T,Output=T> + Copy, const N: usize, const A: u32, const B: u32> Mul<T> for DiscreteSpectrum<T,N,A,B> {
    type Output = Self;
    #[inline] fn mul(self, rhs: T) -> Self { DiscreteSpectrum { data: self.data * rhs } }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
impl<T: Div<T,Output=T> + Copy, const N: usize, const A: u32, const B: u32> Div<T> for DiscreteSpectrum<T,N,A,B> {
    type Output = Self;
    #[inline] fn div(self, rhs: T) -> Self { DiscreteSpectrum { data: self.data / rhs } }
}
