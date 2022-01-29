use crate::{triangular::IntoTriangular, LinalgError, Result};

use ndarray::{Array2, ArrayBase, Data, DataMut, Ix2};
use num_traits::{real::Real, NumAssignOps, NumRef};

pub trait CholeskyInplace {
    fn cholesky_inplace(&mut self) -> Result<&mut Self>;
}

impl<A, S> CholeskyInplace for ArrayBase<S, Ix2>
where
    A: Real + NumRef + NumAssignOps,
    S: DataMut<Elem = A>,
{
    fn cholesky_inplace(&mut self) -> Result<&mut Self> {
        let m = self.nrows();
        let n = self.ncols();
        if m != n {
            return Err(LinalgError::NotSquare { rows: m, cols: n });
        }

        // TODO change accesses to uget and uget_mut
        for j in 0..n {
            let mut d = A::zero();
            for k in 0..j {
                let mut s = A::zero();
                for i in 0..k {
                    s += *self.get((k, i)).unwrap() * *self.get((j, i)).unwrap();
                }
                s = (*self.get((j, k)).unwrap() - s) / self.get((k, k)).unwrap();
                *self.get_mut((j, k)).unwrap() = s;
                d += s * s;
            }
            d = *self.get((j, j)).unwrap() - d;

            if d < A::zero() {
                return Err(LinalgError::NotPositiveDefinite);
            }

            *self.get_mut((j, j)).unwrap() = d.sqrt();
        }

        self.into_lower_triangular()?;
        Ok(self)
    }
}

pub trait CholeskyInto {
    type Output;

    fn cholesky_into(self) -> Result<Self::Output>;
}

impl<A, S> CholeskyInto for ArrayBase<S, Ix2>
where
    A: Real + NumRef + NumAssignOps,
    S: DataMut<Elem = A>,
{
    type Output = Self;

    fn cholesky_into(mut self) -> Result<Self::Output> {
        self.cholesky_inplace()?;
        Ok(self)
    }
}

pub trait Cholesky {
    type Output;

    fn cholesky(&self) -> Result<Self::Output>;
}

impl<A, S> Cholesky for ArrayBase<S, Ix2>
where
    A: Real + NumRef + NumAssignOps,
    S: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn cholesky(&self) -> Result<Self::Output> {
        let arr = self.to_owned();
        arr.cholesky_into()
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
        assert_abs_diff_eq!(chol, lower, epsilon = 1e-4);
        assert_abs_diff_eq!(chol.dot(&chol.t()), arr, epsilon = 1e-4);
    }

    #[test]
    fn bad_matrix() {
        let row = array![[1., 2., 3.], [3., 4., 5.]];
        assert!(matches!(
            row.cholesky(),
            Err(LinalgError::NotSquare { rows: 2, cols: 3 })
        ));

        let non_pd = array![[1., 2.], [2., 1.]];
        let res = non_pd.cholesky_into();
        assert!(matches!(res, Err(LinalgError::NotPositiveDefinite)));
    }

    #[test]
    fn corner_cases() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert_eq!(empty.cholesky().unwrap(), empty);

        let one = array![[1.]];
        assert_eq!(one.cholesky().unwrap(), one);
    }
}
