#![allow(dead_code, unused)]
use core::iter::Sum;
use core::ops::{Add, Div, Sub};
use maths::{
    self,
    linear_algebra::{Matrix, Vector},
    traits::{Float, NumOps, One, Zero},
};
use optimisation::mappable::Mappable;

pub type ColourMatrix<T = f64> = maths::linear_algebra::Matrix3x3<T>;
pub type Triplet<T = f64> = maths::linear_algebra::Vector3D<T>;
// pub type ColourMatrixF32 = maths::Matrix3x3::<f32>;

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Xy<T = f64> {
    pub x: T,
    pub y: T,
}

impl<T> Xy<T> {
    #[inline]
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
    #[inline]
    pub fn to_xyz(self) -> [T; 3]
    where
        T: One + Zero + Div<T, Output = T> + Sub<T, Output = T> + Copy,
    {
        /* Don't normalise if not possible */
        if self.y.is_zero() {
            [self.x, self.y, T::one() - self.x - self.y]
        } else {
            [
                self.x / self.y,
                T::one(),
                (T::one() - self.x - self.y) / self.y,
            ]
        }
    }
    #[inline]
    pub fn from_xyz(xyz: [T; 3]) -> Self
    where
        T: Add<T, Output = T> + Div<T, Output = T> + Copy,
    {
        let sum = xyz[0] + xyz[1] + xyz[2];
        Xy::new(xyz[0] / sum, xyz[1] / sum)
    }
    #[inline]
    pub fn as_vector(self) -> Vector<T, 2> {
        let Self { x, y } = self;
        Vector([x, y])
    }
}

impl<T> From<Vector<T, 2>> for Xy<T> {
    #[inline]
    fn from(vector: Vector<T, 2>) -> Self {
        let Vector([x, y]) = vector;
        return Self { x, y };
    }
}

impl<A> Mappable<A> for Xy<A> {
    type Wrapped<B> = Xy<B>;
    #[inline]
    fn fmap<F, B>(self, mut f: F) -> Xy<B>
    where
        F: FnMut(A) -> B,
    {
        Xy {
            x: f(self.x),
            y: f(self.y),
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct RGBxy<T> {
    pub r: Xy<T>,
    pub g: Xy<T>,
    pub b: Xy<T>,
    pub w: Xy<T>,
}

impl<T> RGBxy<T> {
    #[inline]
    pub fn get_matrix_to_xyz(self) -> ColourMatrix<T>
    where
        T: Copy + NumOps<Output = T> + One + Zero + Sum,
    {
        self.get_matrix_to_rgb().invert3x3()
    }

    #[inline]
    pub fn get_matrix_to_rgb(self) -> ColourMatrix<T>
    where
        T: One + Zero + Copy + NumOps<Output = T> + Sum,
    {
        /* This has wrong white point but correct R/G/B primaries */
        let xyz_to_rgb = ColourMatrix::new([self.r.to_xyz(), self.g.to_xyz(), self.b.to_xyz()])
            .transpose()
            .invert3x3();

        /* Balance white to = (1,1,1) in RGB */
        let white = xyz_to_rgb * Vector(self.w.to_xyz());
        Matrix::from_fn(|r, c| xyz_to_rgb[r][c] / white[r])
    }
}

impl<A> Mappable<A> for RGBxy<A> {
    type Wrapped<B> = RGBxy<B>;
    #[inline]
    fn fmap<F, B>(self, mut f: F) -> RGBxy<B>
    where
        F: FnMut(A) -> B,
    {
        RGBxy {
            r: self.r.fmap(&mut f),
            g: self.g.fmap(&mut f),
            b: self.b.fmap(&mut f),
            w: self.w.fmap(&mut f),
        }
    }
}

impl<T> From<RGBSpace<T>> for RGBxy<T>
where
    T: One + Zero + NumOps<Output = T> + Copy + Sum,
{
    #[inline]
    fn from(space: RGBSpace<T>) -> Self {
        let (one, zero) = (T::one, T::zero);
        match space {
            RGBSpace::FromPrimaries(c) => c,
            RGBSpace::FromMatrixToRGB(m) => RGBSpace::FromMatrixToXYZ(m.invert3x3()).into(),
            RGBSpace::FromMatrixToXYZ(m) => RGBxy {
                r: Xy::from_xyz((m * Vector([one(), zero(), zero()])).0),
                g: Xy::from_xyz((m * Vector([zero(), one(), zero()])).0),
                b: Xy::from_xyz((m * Vector([zero(), zero(), one()])).0),
                w: Xy::from_xyz((m * Vector([one(), one(), one()])).0),
            },
        }
    }
}

/* Three ways to define an RGB space */
#[derive(PartialEq, Clone, Copy, Debug)]
pub enum RGBSpace<T = f64> {
    FromMatrixToXYZ(ColourMatrix<T>),
    FromMatrixToRGB(ColourMatrix<T>),
    FromPrimaries(RGBxy<T>),
}

impl<T> RGBSpace<T> {
    #[inline]
    pub fn get_matrix_to_xyz(self) -> ColourMatrix<T>
    where
        T: One + Zero + Copy + NumOps<Output = T> + Sum,
    {
        match self {
            Self::FromMatrixToXYZ(to_xyz) => to_xyz,
            Self::FromMatrixToRGB(to_rgb) => to_rgb.invert3x3(),
            Self::FromPrimaries(c) => c.get_matrix_to_xyz(),
        }
    }

    #[inline]
    pub fn get_matrix_to_rgb(self) -> ColourMatrix<T>
    where
        T: One + Zero + Copy + NumOps<Output = T> + Sum,
    {
        match self {
            Self::FromMatrixToXYZ(to_xyz) => to_xyz.invert3x3(),
            Self::FromMatrixToRGB(to_rgb) => to_rgb,
            Self::FromPrimaries(c) => c.get_matrix_to_rgb(),
        }
    }

    #[inline]
    pub fn get_primaries(self) -> RGBxy<T>
    where
        T: One + Zero + NumOps<Output = T> + Copy + Sum,
    {
        match self {
            Self::FromPrimaries(c) => c,
            _ => RGBxy::from(self),
        }
    }

    #[inline]
    pub fn adapt_white_point(self, adaptation_space: Self, to: Xy<T>) -> Self
    where
        T: PartialEq + One + Zero + Copy + Float,
    {
        /* DIRTY solution ignores adaptation space */
        if self == adaptation_space {
            match (self, adaptation_space) {
                // TODO: DONT IGNOERE ADAPT SPCAE
                (Self::FromPrimaries(space), Self::FromPrimaries(adapt_space)) => {
                    Self::FromPrimaries(RGBxy {
                        r: space.r,
                        g: space.g,
                        b: space.b,
                        w: to,
                    })
                }
                _ => Self::FromPrimaries(self.get_primaries())
                    .adapt_white_point(Self::FromPrimaries(adaptation_space.get_primaries()), to),
            }
        } else {
            todo!()
        }
    }
}

impl RGBSpace {
    pub const LMS_HPE: Self = Self::FromMatrixToRGB(Matrix([
        [0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340, 0.04641],
        [0.00000, 0.00000, 1.00000],
    ]));

    // same primaries as HPE
    pub const LMS_VONKRIES: Self = Self::FromMatrixToRGB(Matrix([
        [0.40024, 0.7076, -0.08081],
        [-0.2263, 1.16532, 0.0457],
        [0., 0., 0.91822],
    ]));

    pub const LMS_CIE_2006_10: Self = Self::FromMatrixToXYZ(Matrix([
        [1.93986443, -1.34664359, 0.43044935],
        [0.69283932, 0.34967567, 0.00000000],
        [0.00000000, 0.00000000, 2.14687945],
    ]));

    pub const LMS_CIE_2006_2: Self = Self::FromMatrixToXYZ(Matrix([
        [1.94735469, -1.41445123, 0.36476327],
        [0.68990272, 0.34832189, 0.00000000],
        [0.00000000, 0.00000000, 1.93485343],
    ]));
}

impl RGBSpace {
    #![allow(non_upper_case_globals)]

    /* TODO: why is the resulting sRGB matrix slightly off WTF?? */
    pub const sRGB: Self = Self::FromPrimaries(RGBxy {
        r: Xy::new(0.64, 0.33),
        g: Xy::new(0.30, 0.60),
        b: Xy::new(0.15, 0.06),
        w: Xy::new(0.3127, 0.3290),
    });
    pub const STUDIO_DISPLAY_SRGB: Self = Self::FromPrimaries(RGBxy {
        r: Xy {
            x: 0.639981,
            y: 0.338009,
        },
        g: Xy {
            x: 0.303760,
            y: 0.595236,
        },
        b: Xy {
            x: 0.141197,
            y: 0.067588,
        },
        w: Xy {
            x: 0.306508,
            y: 0.319743,
        },
    });
    pub const AdobeRGB: Self = Self::FromPrimaries(RGBxy {
        r: Xy::new(0.64, 0.33),
        g: Xy::new(0.21, 0.71),
        b: Xy::new(0.15, 0.06),
        w: Xy::new(0.3127, 0.3290),
    });
    pub const DisplayP3: Self = Self::FromPrimaries(RGBxy {
        r: Xy::new(0.680, 0.320),
        g: Xy::new(0.265, 0.690),
        b: Xy::new(0.150, 0.060),
        w: Xy::new(0.3127, 0.3290),
    });
    pub const Rec2020: Self = Self::FromPrimaries(RGBxy {
        r: Xy::new(0.708, 0.292),
        g: Xy::new(0.170, 0.797),
        b: Xy::new(0.131, 0.046),
        w: Xy::new(0.3127, 0.3290),
    });
    pub const ACES_AP0: Self = Self::FromPrimaries(RGBxy {
        r: Xy::new(0.7347, 0.2653),
        g: Xy::new(0.0000, 1.0000),
        b: Xy::new(0.0010, -0.0770),
        w: Xy::new(0.32168, 0.33767),
    });
    pub const ACES_AP1: Self = Self::FromPrimaries(RGBxy {
        r: Xy::new(0.713, 0.293),
        g: Xy::new(0.165, 0.839),
        b: Xy::new(0.128, 0.044),
        w: Xy::new(0.32168, 0.33767),
    });
    pub const ProPhoto: Self = Self::FromPrimaries(RGBxy {
        r: Xy::new(0.734699, 0.265301),
        g: Xy::new(0.159597, 0.840403),
        b: Xy::new(0.036598, 0.000105),
        w: Xy::new(0.345704, 0.358540),
    });
    pub const CIE_RGB: Self = Self::FromPrimaries(RGBxy {
        r: Xy::new(0.7347, 0.2653),
        g: Xy::new(0.2738, 0.7174),
        b: Xy::new(0.1666, 0.0089),
        w: Xy::new(0.3333, 0.3333),
    });
    pub const ARRI_ALEXA_WIDE_GAMUT: Self = Self::FromPrimaries(RGBxy {
        r: Xy::new(0.6840, 0.3130),
        g: Xy::new(0.2210, 0.8480),
        b: Xy::new(0.0861, -0.1020),
        w: Xy::new(0.3127, 0.3290),
    });
}

impl<A> Mappable<A> for RGBSpace<A> {
    type Wrapped<B> = RGBSpace<B>;
    #[inline]
    fn fmap<F, B>(self, f: F) -> RGBSpace<B>
    where
        F: FnMut(A) -> B,
    {
        match self {
            RGBSpace::FromMatrixToRGB(m) => RGBSpace::FromMatrixToRGB(m.fmap(f)),
            RGBSpace::FromMatrixToXYZ(m) => RGBSpace::FromMatrixToRGB(m.fmap(f)),
            RGBSpace::FromPrimaries(p) => RGBSpace::FromPrimaries(p.fmap(f)),
        }
    }
}

#[inline]
pub fn rgb_to_hsv<T: Float>(rgb: Vector<T, 3>) -> Vector<T, 3> {
    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    let c_max = r.max(g).max(b);
    let c_min = r.min(g).min(b);
    let delta = c_max - c_min;

    let hue = if delta.is_zero() {
        T::zero()
    } else if c_max == r {
        ((g - b) / delta) % T::int(6)
    } else if c_max == g {
        ((b - r) / delta + T::int(2)) % T::int(6)
    } else {
        ((r - g) / delta + T::int(4)) % T::int(6)
    };

    Vector([
        hue.rem_euclid(T::int(6)) / T::int(6),
        if c_max.is_zero() {
            T::zero()
        } else {
            delta / c_max
        },
        c_max,
    ])
}

#[inline]
pub fn hsv_to_rgb<T: Float>(hsv: Vector<T, 3>) -> Vector<T, 3> {
    let (hue, saturation, value) = (hsv[0].rem_euclid(T::one()), hsv[1], hsv[2]);

    let c = value * saturation;
    let x = c * (T::one() - ((hue * T::int(6)) % T::int(2) - T::one()).abs());
    let m = value - c;

    let rgb = if hue < T::frac(1, 6) {
        Vector([c, x, T::zero()])
    } else if hue < T::frac(2, 6) {
        Vector([x, c, T::zero()])
    } else if hue < T::frac(3, 6) {
        Vector([T::zero(), c, x])
    } else if hue < T::frac(4, 6) {
        Vector([T::zero(), x, c])
    } else if hue < T::frac(5, 6) {
        Vector([x, T::zero(), c])
    } else {
        Vector([c, T::zero(), x])
    };

    rgb + Vector([m, m, m])
}
