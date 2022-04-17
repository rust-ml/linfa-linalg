//! Cholesky decomposition on symmetric positive definite matrices.
//!
//! This module also exports related functionality on symmetric positive definite matrices, such as
//! solving systems and inversion.

use crate::{
    check_square,
    index::*,
    triangular::{IntoTriangular, SolveTriangularInplace, UPLO},
    LinalgError, Result,
};

use ndarray::{Array, Array2, ArrayBase, Data, DataMut, Ix2, NdFloat};

/// Cholesky decomposition of a symmetric positive definite matrix
pub trait CholeskyInplace {
    /// Computes decomposition `A = L * L.t` where L is a lower-triangular matrix in place.
    /// The upper triangle portion is not zeroed out.
    fn cholesky_inplace_dirty(&mut self) -> Result<&mut Self>;

    /// Computes decomposition `A = L * L.t` where L is a lower-triangular matrix, passing by
    /// value.
    /// The upper triangle portion is not zeroed out.
    fn cholesky_into_dirty(mut self) -> Result<Self>
    where
        Self: Sized,
    {
        self.cholesky_inplace_dirty()?;
        Ok(self)
    }

    /// Computes decomposition `A = L * L.t` where L is a lower-triangular matrix in place.
    fn cholesky_inplace(&mut self) -> Result<&mut Self>;

    /// Computes decomposition `A = L * L.t` where L is a lower-triangular matrix, passing by
    /// value.
    fn cholesky_into(mut self) -> Result<Self>
    where
        Self: Sized,
    {
        self.cholesky_inplace()?;
        Ok(self)
    }
}

impl<A, S> CholeskyInplace for ArrayBase<S, Ix2>
where
    A: NdFloat,
    S: DataMut<Elem = A>,
{
    fn cholesky_inplace_dirty(&mut self) -> Result<&mut Self> {
        let n = check_square(self)?;

        for j in 0..n {
            let mut d = A::zero();
            unsafe {
                for k in 0..j {
                    let mut s = A::zero();
                    for i in 0..k {
                        s += *self.at((k, i)) * *self.at((j, i));
                    }
                    s = (*self.at((j, k)) - s) / *self.at((k, k));
                    *self.atm((j, k)) = s;
                    d += s * s;
                }
                d = *self.at((j, j)) - d;
            }

            if d <= A::zero() {
                return Err(LinalgError::NotPositiveDefinite);
            }

            unsafe { *self.atm((j, j)) = d.sqrt() };
        }
        Ok(self)
    }

    fn cholesky_inplace(&mut self) -> Result<&mut Self> {
        self.cholesky_inplace_dirty()?;
        self.triangular_inplace(UPLO::Lower)?;
        Ok(self)
    }
}

/// Cholesky decomposition of a symmetric positive definite matrix, without modifying the original
pub trait Cholesky {
    type Output;

    /// Computes decomposition `A = L * L.t` where L is a lower-triangular matrix without modifying
    /// or consuming the original.
    /// The upper triangle portion is not zeroed out.
    fn cholesky_dirty(&self) -> Result<Self::Output>;

    /// Computes decomposition `A = L * L.t` where L is a lower-triangular matrix without modifying
    /// or consuming the original.
    fn cholesky(&self) -> Result<Self::Output>;
}

impl<A, S> Cholesky for ArrayBase<S, Ix2>
where
    A: NdFloat,
    S: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn cholesky_dirty(&self) -> Result<Self::Output> {
        let arr = self.to_owned();
        arr.cholesky_into_dirty()
    }

    fn cholesky(&self) -> Result<Self::Output> {
        let arr = self.to_owned();
        arr.cholesky_into()
    }
}

/// Solves a symmetric positive definite system
pub trait SolveCInplace<B> {
    /// Solves `self * x = b`, where `self` is symmetric positive definite, modifying `b` inplace.
    ///
    /// As a side effect, `self` is used to calculate an in-place Cholesky decomposition.
    fn solvec_inplace<'a>(&mut self, b: &'a mut B) -> Result<&'a mut B>;

    /// Solves `self * x = b`, where `self` is symmetric positive definite, consuming `b`.
    ///
    /// As a side effect, `self` is used to calculate an in-place Cholesky decomposition.
    fn solvec_into(&mut self, mut b: B) -> Result<B> {
        self.solvec_inplace(&mut b)?;
        Ok(b)
    }
}

impl<A: NdFloat, Si: DataMut<Elem = A>, So: DataMut<Elem = A>> SolveCInplace<ArrayBase<So, Ix2>>
    for ArrayBase<Si, Ix2>
{
    fn solvec_inplace<'a>(
        &mut self,
        b: &'a mut ArrayBase<So, Ix2>,
    ) -> Result<&'a mut ArrayBase<So, Ix2>> {
        let chol = self.cholesky_inplace_dirty()?;
        chol.solve_triangular_inplace(b, UPLO::Lower)?;
        chol.t().solve_triangular_inplace(b, UPLO::Upper)?;
        Ok(b)
    }
}

/// Solves a symmetric positive definite system
pub trait SolveC<B> {
    type Output;

    /// Solves `self * x = b`, where `self` is symmetric positive definite.
    fn solvec(&mut self, b: &B) -> Result<Self::Output>;
}

impl<A: NdFloat, Si: DataMut<Elem = A>, So: Data<Elem = A>> SolveC<ArrayBase<So, Ix2>>
    for ArrayBase<Si, Ix2>
{
    type Output = Array<A, Ix2>;

    fn solvec(&mut self, b: &ArrayBase<So, Ix2>) -> Result<Self::Output> {
        self.solvec_into(b.to_owned())
    }
}

/// Inverse of a symmetric positive definite matrix
pub trait InverseCInplace {
    type Output;

    /// Computes inverse of symmetric positive definite matrix.
    ///
    /// As a side effect, `self` is used to calculate an in-place Cholesky decomposition.
    fn invc_inplace(&mut self) -> Result<Self::Output>;
}

impl<A: NdFloat, S: DataMut<Elem = A>> InverseCInplace for ArrayBase<S, Ix2> {
    type Output = Array2<A>;

    fn invc_inplace(&mut self) -> Result<Self::Output> {
        let eye = Array2::eye(self.nrows());
        let res = self.solvec_into(eye)?;
        Ok(res)
    }
}

/// Inverse of a symmetric positive definite matrix
pub trait InverseC {
    type Output;

    /// Computes inverse of symmetric positive definite matrix.
    fn invc(&self) -> Result<Self::Output>;
}

impl<A: NdFloat, S: Data<Elem = A>> InverseC for ArrayBase<S, Ix2> {
    type Output = Array2<A>;

    fn invc(&self) -> Result<Self::Output> {
        self.to_owned().invc_inplace()
    }
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn decompose() {
        let arr = array![[25., 15., -5.], [15., 18., 0.], [-5., 0., 11.]];
        let lower = array![[5.0, 0.0, 0.0], [3.0, 3.0, 0.0], [-1., 1., 3.]];

        let chol = arr.cholesky().unwrap();
        assert_abs_diff_eq!(chol, lower, epsilon = 1e-7);
        assert_abs_diff_eq!(chol.dot(&chol.t()), arr, epsilon = 1e-7);
    }

    #[test]
    fn bad_matrix() {
        let mut row = array![[1., 2., 3.], [3., 4., 5.]];
        assert!(matches!(
            row.cholesky(),
            Err(LinalgError::NotSquare { rows: 2, cols: 3 })
        ));
        assert!(matches!(
            row.solvec(&Array2::zeros((2, 3))),
            Err(LinalgError::NotSquare { rows: 2, cols: 3 })
        ));

        let mut non_pd = array![[1., 2.], [2., 1.]];
        assert!(matches!(
            non_pd.cholesky(),
            Err(LinalgError::NotPositiveDefinite)
        ));
        assert!(matches!(
            non_pd.solvec(&Array2::zeros((2, 3))),
            Err(LinalgError::NotPositiveDefinite)
        ));

        let zeros = array![[0., 0.], [0., 0.]];
        assert!(matches!(
            zeros.cholesky(),
            Err(LinalgError::NotPositiveDefinite)
        ));
    }

    #[test]
    fn solvec() {
        let mut arr = array![[25., 15., -5.], [15., 18., 0.], [-5., 0., 11.]];
        let x = array![
            [10., -3., 2.2, 4.],
            [0., 2.4, -0.9, 1.1],
            [5.5, 7.6, 8.1, 10.]
        ];
        let b = arr.dot(&x);

        let out = arr.solvec(&b).unwrap();
        assert_abs_diff_eq!(out, x, epsilon = 1e-7);
    }

    #[test]
    fn invc() {
        let arr = array![[25., 15., -5.], [15., 18., 0.], [-5., 0., 11.]];
        let inv = arr.invc().unwrap();
        assert_abs_diff_eq!(arr.dot(&inv), Array2::eye(3));
    }

    #[test]
    fn corner_cases() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert_eq!(empty.cholesky().unwrap(), empty);
        assert_eq!(empty.clone().invc().unwrap(), empty);

        let one = array![[1.]];
        assert_eq!(one.cholesky().unwrap(), one);
        assert_eq!(one.clone().invc().unwrap(), one);
    }
}
