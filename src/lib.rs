use ndarray::{ArrayBase, DataMut, Ix2};
use num_traits::{real::Real, NumAssignOps, NumRef};
use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LinalgError {
    #[error("Matrix with {rows} rows and {cols} cols is not square")]
    NotSquare { rows: usize, cols: usize },
    #[error("Matrix is not positive definite")]
    NotPositiveDefinite,
}

pub type Result<T> = std::result::Result<T, LinalgError>;

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
