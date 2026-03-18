#![cfg_attr(rustfmt, rustfmt_skip)]
use super::traits::{Float, Zero, One, NumOps};
use core::ops::{Add, Sub, Mul, Div, AddAssign, DivAssign, MulAssign, SubAssign, Neg, Index, IndexMut};
use core::iter::Sum;
use core::array;

/*********************************************************************/
/*********************** Matrix implementation ***********************/
/*********************************************************************/

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Matrix<T, const R: usize, const C: usize> (pub [[T; C]; R]);
pub type MatrixNxN<T, const N: usize> = Matrix<T, N, N>;
pub type Matrix3x3<T> = MatrixNxN<T,3>;
pub type Matrix2x2<T> = MatrixNxN<T,2>;

impl <T, const R: usize, const C: usize> Matrix<T, R, C> {
    #[inline]
    pub fn from_fn<F: FnMut(usize, usize) -> T>(mut f: F) -> Self {
        Self(array::from_fn(|row| array::from_fn(|col| f(row, col))))
    }
    #[inline]
    pub fn map<F: FnMut(T) -> U, U>(self, mut f: F) -> Matrix<U,R,C> {
        Matrix(self.0.map(|row| row.map(&mut f)))
    }

    /** Return a flat view of the matrix elements */
    #[inline] pub const fn as_flattened(&self) -> &[T] { self.0.as_flattened() }
    #[inline] pub const fn as_flattened_mut(&mut self) -> &mut [T] { self.0.as_flattened_mut() }
    #[inline] pub const fn repeat(value: T) -> Self where T: Copy { Self ([[value; C]; R]) }
    #[inline] pub fn transpose(self) -> Matrix<T,C,R> where T: Copy { Matrix::from_fn(|r,c| self[c][r]) }
    #[inline] pub const fn new(data: [[T; C]; R]) -> Self { Self (data) }
    #[inline] pub fn as_array(self) -> [[T; C]; R] { self.0 }
    #[inline] pub const fn swap_rows(&mut self, a: usize, b: usize) { if a != b { self.0.swap(a, b); } }


    #[inline]
    pub fn from_slice(slice: &[T]) -> Option<Self>
    where
        T: Clone
    {
        (slice.len() == R * C).then(|| Self::from_fn(|r,c| slice[r*C+c].clone()))
    }
}

impl<T: Clone, const R: usize, const C: usize> TryFrom<&[T]> for Matrix<T,R,C> {
    type Error = ();
    #[inline]
    fn try_from(data: &[T]) -> Result<Self, ()> {
        Self::from_slice(data).ok_or(())
    }
}

impl<T, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T,R,C> {
    #[inline] fn from(data: [[T; C]; R]) -> Self { Self(data) }
}

impl<T> From<[T; 4]> for Matrix2x2<T> {
    #[inline] fn from([a,b,c,d]: [T; 4]) -> Self { Self([[a,b],[c,d]]) }
}

impl<T> From<[T; 9]> for Matrix3x3<T> {
    #[inline]
    fn from([a,b,c,d,e,f,g,h,i]: [T; 9]) -> Self { Self([[a,b,c],[d,e,f],[g,h,i]]) }
}

impl<T> From<[T; 16]> for Matrix<T,4,4> {
    #[inline]
    fn from([a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p]: [T; 16]) -> Self {
        Self([[a,b,c,d],[e,f,g,h],[i,j,k,l],[m,n,o,p]])
    }
}

impl<T, const R: usize, const C: usize> Into<[[T; C]; R]> for Matrix<T,R,C> {
    #[inline] fn into(self) -> [[T; C]; R] { self.0 }
}

impl<T> Into<[T; 4]> for Matrix2x2<T> {
    #[inline]
    fn into(self) -> [T; 4] {
        let [[a,b],[c,d]] = self.0;
        [a,b,c,d]
    }
}

impl<T> Into<[T; 9]> for Matrix3x3<T> {
    #[inline]
    fn into(self) -> [T; 9] {
        let [[a,b,c],[d,e,f],[g,h,i]] = self.0;
        [a,b,c,d,e,f,g,h,i]
    }
}

impl<T> Into<[T; 16]> for Matrix<T,4,4> {
    #[inline]
    fn into(self) -> [T; 16] {
        let [[a,b,c,d],[e,f,g,h],[i,j,k,l],[m,n,o,p]] = self.0;
        [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p]
    }
}

impl<const R: usize, const C: usize> Matrix<f64,R,C> {
    #[inline] pub fn as_f32(self) -> Matrix<f32,R,C> { self.map(|x| x as f32) }
}

impl<const R: usize, const C: usize> Matrix<f32,R,C> {
    #[inline] pub fn as_f64(self) -> Matrix<f64,R,C> { self.map(|x| x as f64) }
}

/* 3x3 specific functions */
impl<T> Matrix3x3<T> {
    /* Creates a rotation matrix using rodrigues rotation formula (axis + angle) */
    #[inline]
    pub fn rotation_axis_angle(axis: Vector3D<T>, angle: T) -> Self where T: Float {
        Self(array::from_fn(|r| Vector::unit(r).rotate(axis, angle).as_array()))
    }

    #[inline]
    pub fn rotation_euler(x: T, y: T, z: T) -> Self where T: Float {
        let (cosx, cosy, cosz) = (x.cos(), y.cos(), z.cos());
        let (sinx, siny, sinz) = (x.sin(), y.sin(), z.sin());
        Self ([[                cosy*cosz,                -cosy*sinz,       siny],
               [ sinx*siny*cosz+cosx*sinz, -sinx*siny*sinz+cosx*cosz, -sinx*cosy],
               [-cosx*siny*cosz+sinx*sinz,  cosx*siny*sinz+sinx*cosz,  cosx*cosy]])
    }

    #[inline]
    pub fn invert3x3(self) -> Self
      where T: Copy + NumOps<Output=T> {
        let [[a,b,c],[d,e,f],[g,h,i]] = self.0;
        let determinant = a*(i*e-h*f) - d*(i*b-h*c) + g*(f*b-e*c);
        Self::from_fn(|c,r| Matrix2x2::from_fn(|r2,c2| self[(r+r2+1)%3][(c+c2+1)%3]).det2x2() / determinant)
    }

    #[inline]
    pub fn transpose3x3(self) -> Self {
        let Self([[a,b,c],[d,e,f],[g,h,i]]) = self;
        return Self ([[a,d,g],[b,e,h],[c,f,i]])
    }
}

impl<T> Matrix2x2<T> {
    #[inline]
    pub fn invert2x2(self) -> Self
      where T: Clone + NumOps<Output=T> + Neg<Output=T> {
        // made this clone-compatible for fun, dont know why
        let det = self.clone().det2x2();
        let [[a,b],[c,d]] = self.0;
        Self([[d / det.clone(),-b / det.clone()],[-c / det.clone(),a / det]])
    }
    #[inline]
    fn det2x2(self) -> T where T: Mul<Output=T> + Sub<Output=T> {
        let [[a,b],[c,d]] = self.0;
        a*d - b*c
    }
}

#[inline]
fn eliminate_spine_flat<T: Float>(
    lhs: &mut [T], rhs: &mut [T],
    dimension: usize, row_stride: usize, spine: usize
) -> Result<(),()> {
    /* Find largest absolute valued row */
    let (mut index, mut largest_value) = (spine, lhs[spine * row_stride + spine]);
    for row in (spine+1)..dimension {
        if lhs[row * row_stride + spine].abs() > largest_value.abs() {
            (index, largest_value) = (row, lhs[row * row_stride + spine]);
        }
    }
    if !largest_value.is_zero() {
        /* Swap rows */
        let (row_spine, row_index) = (spine * row_stride, index * row_stride);
        for i in 0..dimension {
            lhs.swap(row_spine+i, row_index+i);
            rhs.swap(row_spine+i, row_index+i);
        }
        for row in (spine+1)..dimension {
            let x = lhs[row * row_stride + spine];
            if !x.is_zero() {
                for col in 0..dimension {
                    let (idx, s_idx) = (row * row_stride + col, spine * row_stride + col);
                    lhs[idx] -= lhs[s_idx] * x / largest_value;
                    rhs[idx] -= rhs[s_idx] * x / largest_value;
                }
            }
        }
        return Ok(());
    } else {
        return Err(());
    }
}

/* Inverts a matrix stored in a flat/1D array.
 * Will mess up the values inside "input", inverted result placed
 * into output. May fail, check the result. */
#[inline]
pub fn invert_mat<T: Float>(
    input: &mut [T], output: &mut [T], dimension: usize, row_stride: usize
) -> Result<(),()> {
    /* Initialise output to identitiy matrix */
    for row in 0..dimension {
        for col in 0..dimension {
            output[row * row_stride + col] = if row == col {T::one()} else {T::zero()};
        }
    }
    /* Gaussian elimination to zero everything on the bottom left */
    for x in 0..dimension { eliminate_spine_flat(input, output, dimension, row_stride, x)? }
    /* Now it's in row echelon form, do backward substitution */
    for row_from in (0..dimension).rev() {
        for row in 0..row_from {
            let m = input[row*row_stride+row_from] / input[row_from*row_stride+row_from];
            for c in 0..dimension {
                output[row*row_stride+c] -= output[row_from*row_stride+c] * m;
                input[row*row_stride+c] -= input[row_from*row_stride+c] * m;
            }
        }
    }
    /* Normalise rows */
    for row in 0..dimension {
        for col in 0..dimension {
            output[row*row_stride+col] /= input[row*row_stride+row];
        }
    }
    Ok(())
}

#[inline]
pub fn invert_mat_vec<T: Float>(mut matrix: Vec<T>, dimension: usize) -> Option<Vec<T>> {
    let mut output = Vec::with_capacity(dimension * dimension);
    unsafe { output.set_len(dimension * dimension); }
    invert_mat(matrix.as_mut_slice(), output.as_mut_slice(), dimension, dimension).ok()?;
    Some(output)
}

/* Multiplies the transpose of a matrix with itself to get a square matrix, used in least squares to produce JT*J */
#[inline]
pub fn transpose_mul<T: Float>(mat: &[T], out: &mut [T], cols: usize, rows: usize, row_stride: usize) {
    for y in 0..cols {
        for x in 0..cols {
            out[y*row_stride+x] = (0..rows).map(|r| r*row_stride).map(|off| mat[off+x] * mat[off+y]).sum()
        }
    }
}

/* Square (NxN) matrix functions */
impl <T, const N: usize> MatrixNxN<T,N> {
    /* Square (NxN) matrix functions, create identity matrix */
    #[inline]
    pub fn id() -> Self where T: Zero + One {
        Self::from_fn(|r,c| if r == c {T::one()} else {T::zero()} )
    }

    /* Eliminate a spine (everythign below a diagonal element) for gaussian elimination.
     * Returns false if all rows are zero and matrix cannot be inverted */
    #[inline(always)]
    fn eliminate_spine(&mut self, rhs: &mut Self, spine: usize) -> Result<(),()> where T: Float {
        /* Find largest absolute valued row */
        let (mut index, mut largest_value) = (spine, self[spine][spine]);
        for row in (spine+1)..N {
            if self[row][spine].abs() > largest_value.abs() {
                (index, largest_value) = (row, self[row][spine]);
            }
        }
        if !largest_value.is_zero() {
            self.swap_rows(spine, index);
            rhs.swap_rows(spine, index);
            for row in (spine+1)..N {
                let x = self[row][spine];
                if !x.is_zero() {
                    for col in 0..N {
                        self[row][col] = self[row][col] - self[spine][col] * x / largest_value;
                        rhs[row][col] = rhs[row][col] - rhs[spine][col] * x / largest_value;
                    }
                }
            }
            return Ok(());
        } else {
            return Err(());
        }
    }

    #[inline]
    /* Invert matrix using gaussian elimination */
    pub fn invert(self) -> Option<Self> where T: Float {
        let (mut lhs, mut rhs) = (self, Self::id());

        /* Gaussian elimination to zero everything on the bottom left */
        for x in 0..N { lhs.eliminate_spine(&mut rhs, x).ok()? }

        /* Now it's in row echelon form, do backward substitution */
        for row_from in (0..N).rev() {
            for row in 0..row_from {
                let m = lhs[row][row_from] / lhs[row_from][row_from];
                for c in 0..N {
                    rhs[row][c] = rhs[row][c] - rhs[row_from][c] * m;
                    // lhs[row][c] = lhs[row][c] - lhs[row_from][c] * m; /* Is the LHS part necessary to evaluate?? */
                }
            }
        }

        /* Normalise rows */
        for r in 0..N {
            for c in 0..N {
                rhs[r][c] /= lhs[r][r];
            }
        }

        Some(rhs)
    }
}

/***************** Standard operator implementations *****************/
/* Matrix multiplication */
impl<T,U,V, const R: usize, const C: usize, const N: usize> Mul<Matrix<U,N,C>> for Matrix<T,R,N>
  where T: Copy + Float + Mul<U,Output=V>, U: Copy, V: Sum {
    type Output = Matrix<V,R,C>;
    #[inline]
    fn mul(self, m: Matrix<U,N,C>) -> Self::Output {
        Matrix::from_fn(|r,c| (0..N).map(|i| self[r][i] * m[i][c]).sum())
        /* TODO: If matrix is very large, transposing B might make it faster (?) */
    }
}

/***************************** Indexing *****************************/
impl<T, const R: usize, const C: usize> core::ops::Index<usize> for Matrix<T,R,C> {
    type Output = [T;C];
    #[inline] fn index(& self, i: usize) -> &[T;C] { &self.0[i] }
}
impl<T, const R: usize, const C: usize> core::ops::IndexMut<usize> for Matrix<T,R,C> {
    #[inline] fn index_mut(& mut self, i: usize) -> &mut [T;C] { &mut self.0[i] }
}

/* Element-wise operations */
impl<T: Copy + Neg<Output=T>, const R: usize, const C: usize> Neg for Matrix<T,R,C> {
    type Output = Self;
    #[inline] fn neg(self) -> Self { self.map(|x| -x) }
}

/* Column vector multiplication */
impl<T: Copy + Mul<Output=T> + Sum, const C: usize, const N: usize> Mul<Vector<T,C>> for Matrix<T,N,C> {
    type Output = Vector<T,N>;
    #[inline] fn mul(self, v: Vector<T,C>) -> Vector<T,N> {
        Vector::from_fn(|i| (Vector(self[i]) * v).into_iter().sum())
    }
}

/* Matrix-scalar operations */
impl<T: Copy + Mul<Output=T>, const R: usize, const C: usize> Mul<T> for Matrix<T,R,C> {
    type Output = Self;
    #[inline] fn mul(self, rhs: T) -> Self { self.map(|x| x * rhs) }
}
impl<T: Copy + Div<Output=T>, const R: usize, const C: usize> Div<T> for Matrix<T,R,C> {
    type Output = Self;
    #[inline] fn div(self, rhs: T) -> Self { self.map(|x| x / rhs) }
}

/*************************** From and Into ***************************/
impl<const R: usize, const C: usize> From<Matrix<f32, R, C>> for Matrix<f64, R, C> {
    #[inline] fn from(item: Matrix<f32,R,C>) -> Self { item.map(|x| x as f64) }
}
impl<const R: usize, const C: usize> From<Matrix<f64, R, C>> for Matrix<f32, R, C> {
    #[inline] fn from(item: Matrix<f64,R,C>) -> Self { item.map(|x| x as f32) }
}

/****************************** Default ******************************/
impl<T: One + Zero, const N: usize> Default for MatrixNxN<T,N> {
    #[inline] fn default() -> Self { Self::id() }
}



/*********************************************************************/
/*********************** Vector implementation ***********************/
/*********************************************************************/

#[derive(Copy,Clone,Debug)]
#[repr(transparent)]
pub struct Vector<T, const N: usize> (pub [T; N]);

pub type Vector2D<T> = Vector<T,2>;
pub type Vector3D<T> = Vector<T,3>;
pub type Point2D<T> = Vector2D<T>;
pub type Point3D<T> = Vector3D<T>;

#[allow(non_snake_case)]
#[inline] pub fn Vector2D<T>(x:T, y:T) -> Vector2D<T> {Vector([x,y])}
#[allow(non_snake_case)]
#[inline] pub fn Vector3D<T>(x:T, y:T, z:T) -> Vector3D<T> {Vector([x,y,z])}
#[allow(non_snake_case)]
#[inline] pub fn Point2D<T>(x:T, y:T) -> Vector2D<T> {Vector2D(x,y)}
#[allow(non_snake_case)]
#[inline] pub fn Point3D<T>(x:T, y:T, z:T) -> Vector3D<T> {Vector3D(x,y,z)}

impl <T, const N: usize> IntoIterator for Vector<T,N> {
    type Item = T;
    type IntoIter = <array::IntoIter<T, N> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter { self.0.into_iter() }
}

impl<T: Default, const N: usize> Default for Vector<T,N> {
    #[inline] fn default() -> Self { Self::from_fn(|_| T::default()) }
}

impl<T, const N: usize> Vector<T,N> {
    #[inline(always)] pub fn get(&self, i: usize) -> Option<&T> { self.0.get(i) }
    #[inline(always)] pub fn as_array(self) -> [T; N] { self.0 }
    #[inline(always)] pub fn as_slice(&self) -> &[T] { self.0.as_slice() }
    #[inline(always)] pub fn as_mut_slice(&mut self) -> &mut [T] { self.0.as_mut_slice() }
    #[inline(always)] pub fn swap(&mut self, a: usize, b: usize) { self.0.swap(a,b) }

    #[inline(always)]
    pub fn pad<const M: usize>(self, pad_value: impl Fn() -> T) -> Vector<T,M> {
        /* I used iterator to avoid having a clone trait bound (IDK if this will
         * lead to any good optimisations or a slowdown because iterators are slow) */
        let mut iter = self.into_iter();
        Vector::from_fn(|_| iter.next().unwrap_or(pad_value()))
    }

    /* Max function works like this: takes first element, and compares it to every
     * subsequent element, replacing it if > comparison returns larger than. Does not consider partial/invalid values. */
    #[inline]
    pub fn max(self) -> T where T: PartialOrd + Clone {
        let mut max = self[0].clone();
        for i in 1..N {
            if self[i] > max {
                max = self[i].clone()
            }
        }
        return max;
    }

    #[inline]
    pub fn min(self) -> T where T: PartialOrd + Clone {
        let mut max = self[0].clone();
        for i in 1..N {
            if self[i] < max {
                max = self[i].clone()
            }
        }
        return max;
    }

    #[inline] pub fn zero_pad<const M: usize>(self) -> Vector<T,M> where T:Zero { self.pad(T::zero) }

    #[inline] pub fn from_fn<F: FnMut(usize) -> T>(f: F) -> Self { Self(array::from_fn(f)) }

    #[inline]
    pub fn shift(self, n: usize) -> Self where T: Clone {
        Self::from_fn(|i| self[(i + n) % N].clone())
    }

    /* Creates a unit vector consisting of zeros and a one at given index */
    #[inline]
    pub fn unit(index: usize) -> Self
      where T: One + Zero {
        Self::from_fn(|i| if i == index {T::one()} else {T::zero()})
    }

    #[inline] pub fn map<U>(self, f: impl FnMut(T) -> U) -> Vector<U,N> { Vector(self.0.map(f)) }

    /* Eughh */
    #[inline(always)] pub fn x(self) -> T where T: Copy {self[0]}
    #[inline(always)] pub fn y(self) -> T where T: Copy {self[1]}
    #[inline(always)] pub fn z(self) -> T where T: Copy {self[2]}
    #[inline(always)] pub fn xy(self) -> Vector2D<T> where T: Copy { Vector2D(self[0],self[1]) }

    #[inline] pub fn dot(self, with: Self) -> T where T: Copy + Mul<Output=T> + Sum { (0..N).map(|i| self[i] * with[i]).sum() }
    #[inline] pub fn sum_of_squares(self) -> T where T: Float { self.into_iter().map(|x| x.powi(2)).sum() }
    #[inline] pub fn sum(self) -> T where T: Float { self.into_iter().sum() }
    #[inline] pub fn magnitude(&self) -> T where T: Float { self.sum_of_squares().sqrt() }
    #[inline] pub fn normalised(self) -> Self where T: Float { self / self.magnitude() }
    #[inline] pub fn distance(self, to: Self) -> T where T: Float { (self - to).magnitude() }
}

impl<T> Vector2D<T> {
    #[inline]
    pub fn perpendicular_clockwise(self) -> Self where T: Neg<Output=T> {
        let Vector([x,y]) = self;
        return Vector([y, -x]);
    }
    #[inline]
    pub fn atan2(self) -> T where T: Float {
        let Vector([x,y]) = self;
        return y.atan2(x);
    }
    #[inline]
    pub fn angle(self, other: Self) -> T where T: Float {
        let (dot, det) = (self.dot(other), self[0]*other[1] - self[1]*other[0]);
        det.atan2(dot)
    }
    // #[inline]
    // pub fn cross(self, Vector([b0,b1]): Self) -> T where T: Mul<Output=T> + Sub<Output=T> {
    //     let Vector([a0,a1]) = self;
    //     a0 * b1 - a1 * b0
    // }
    #[inline]
    pub fn cross(self, Vector([a0,a1]): Self, Vector([b0,b1]): Self) -> T where T: Mul<Output=T> + Sub<Output=T> + Copy {
        let Vector([o0,o1]) = self;
        (a0 - o0) * (b1 - o1) - (a1 - o1) * (b0 - o0)
    }
    // #[inline] pub fn rotate(self, theta: T) -> Self where T: Float {
    //    Vector2D(self[0]*theta.cos()-self[1]*theta.sin(), self[0]*theta.sin()+self[1]*theta.cos())
    // }
}

impl<T> Vector3D<T> {
    #[inline(always)]
    pub fn cross(&self, r: Self) -> Self where T: Copy + Sub<Output=T> + Mul<Output=T> {
        Vector3D(self[1]*r[2]-self[2]*r[1], self[2]*r[0]-self[0]*r[2], self[0]*r[1]-self[1]*r[0])
    }
    /* Rodrigues rotation */
    #[inline(always)]
    pub fn rotate(self, axis: Self, theta: T) -> Self where T: Float {
        let (sintheta, costheta) = (theta.sin(), theta.cos());
        (self * costheta) + (axis.cross(self) * sintheta) + (axis * axis.dot(self) * (T::one() - costheta))
    }
}

/***************************** Sum trait ****************************/
impl<T: Add<T,Output=T> + Zero + Copy, const N: usize> Sum<Self> for Vector<T,N> {
    #[inline] fn sum<I>(iter: I) -> Self where I: Iterator<Item=Self> { iter.fold(Self::zero(), |a,b| a+b) }
}

/***************************** Indexing *****************************/
impl<T, I, const N: usize> Index<I> for Vector<T,N> where [T]: Index<I> {
    type Output = <[T] as Index<I>>::Output;
    #[inline] fn index(&self, i: I) -> &Self::Output { self.0.index(i) }
}
impl<T, const N: usize> IndexMut<usize> for Vector<T,N> {
    #[inline] fn index_mut(&mut self, i: usize) -> &mut T { IndexMut::index_mut(&mut self.0 as &mut [T], i) }
}

/***************** Standard operator implementations *****************/
/* Element-wise operations */
impl<T: Copy + Add<Output=T>, const N: usize> Add<Self> for Vector<T,N> {
    type Output = Self;
    #[inline] fn add(self, rhs: Self) -> Self { Self::from_fn(|i| self[i] + rhs[i]) }
}
impl<T: Copy + Sub<Output=T>, const N: usize> Sub<Self> for Vector<T,N> {
    type Output = Self;
    #[inline] fn sub(self, rhs: Self) -> Self { Self::from_fn(|i| self[i] - rhs[i]) }
}
impl<T: Copy + Mul<Output=T>, const N: usize> Mul<Self> for Vector<T,N> {
    type Output = Self;
    #[inline] fn mul(self, rhs: Self) -> Self { Self::from_fn(|i| self[i] * rhs[i]) }
}
impl<T: Copy + Div<Output=T>, const N: usize> Div<Self> for Vector<T,N> {
    type Output = Self;
    #[inline] fn div(self, rhs: Self) -> Self { Self::from_fn(|i| self[i] / rhs[i]) }
}

impl<T: Copy + Add<Output=T>, const N: usize> AddAssign<Self> for Vector<T,N> {
    #[inline] fn add_assign(&mut self, rhs: Self) { *self = *self + rhs }
}
impl<T: Copy + Sub<Output=T>, const N: usize> SubAssign<Self> for Vector<T,N> {
    #[inline] fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs }
}
impl<T: Copy + Mul<Output=T>, const N: usize> MulAssign<Self> for Vector<T,N> {
    #[inline] fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs }
}
impl<T: Copy + Div<Output=T>, const N: usize> DivAssign<Self> for Vector<T,N> {
    #[inline] fn div_assign(&mut self, rhs: Self) { *self = *self / rhs }
}
impl<T: Neg<Output=U>, U, const N: usize> Neg for Vector<T,N> {
    type Output = Vector<U,N>;
    #[inline] fn neg(self) -> Self::Output { self.map(|x| -x) }
}

impl<T: Zero, const N: usize> Zero for Vector<T,N> {
    #[inline] fn zero() -> Self { Vector::from_fn(|_| T::zero()) }
    #[inline] fn is_zero(&self) -> bool { self.0.iter().any(|x| !x.is_zero()) }
}

/* Vector-scalar operations */
impl<T: Copy + Mul<Output=T>, const N: usize> Mul<T> for Vector<T,N> {
    type Output = Self;
    #[inline] fn mul(self, x: T) -> Self { self.map(|y| y * x) }
}
impl<T: Copy + Div<Output=T>, const N: usize> Div<T> for Vector<T,N> {
    type Output = Self;
    #[inline] fn div(self, x: T) -> Self { self.map(|y| y / x) }
}
