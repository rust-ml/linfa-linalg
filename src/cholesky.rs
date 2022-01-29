use crate::{LinalgError, Result};
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
