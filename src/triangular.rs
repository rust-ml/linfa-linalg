//! Traits for creating and manipulating triangular matrices

use crate::{check_square, index::*, LinalgError, Result};

use ndarray::{s, ArrayBase, Data, DataMut, Ix2, NdFloat, SliceArg};
use num_traits::Zero;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UPLO {
    Upper,
    Lower,
}

/// Transform square matrix into triangular matrix
pub trait IntoTriangular {
    /// Transform square matrix into a strict triangular matrix in place, zeroing out the other
    /// elements.
    fn triangular_inplace(&mut self, uplo: UPLO) -> Result<&mut Self>;

    /// Transform square matrix into a strict triangular matrix, zeroing out the other elements.
    fn into_triangular(self, uplo: UPLO) -> Result<Self>
    where
        Self: Sized;
}

impl<A, S> IntoTriangular for ArrayBase<S, Ix2>
where
    A: Zero,
    S: DataMut<Elem = A>,
{
    fn into_triangular(mut self, uplo: UPLO) -> Result<Self> {
        self.triangular_inplace(uplo)?;
        Ok(self)
    }

    fn triangular_inplace(&mut self, uplo: UPLO) -> Result<&mut Self> {
        let n = check_square(self)?;
        if uplo == UPLO::Upper {
            for i in 0..n {
                for j in 0..i {
                    *self.atm((i, j)) = A::zero();
                }
            }
        } else {
            for i in 0..n {
                for j in i + 1..n {
                    *self.atm((i, j)) = A::zero();
                }
            }
        }
        Ok(self)
    }
}

/// Operations on triangular matrices
pub trait Triangular {
    /// Check if matrix is triangular
    fn is_triangular(&self, uplo: UPLO) -> bool;
}

impl<A, S> Triangular for ArrayBase<S, Ix2>
where
    A: Zero,
    S: Data<Elem = A>,
{
    fn is_triangular(&self, uplo: UPLO) -> bool {
        if let Ok(n) = check_square(self) {
            if uplo == UPLO::Upper {
                for i in 0..n {
                    for j in 0..i {
                        if !self.at((i, j)).is_zero() {
                            return false;
                        }
                    }
                }
            } else {
                for i in 0..n {
                    for j in i + 1..n {
                        if !self.at((i, j)).is_zero() {
                            return false;
                        }
                    }
                }
            }
            true
        } else {
            false
        }
    }
}

#[inline]
/// Generalized implementation for both upper and lower triangular solvers.
/// Ensure that the diagonal of `a` is non-zero, otherwise output will be wrong.
fn solve_triangular_system<A: NdFloat, I: Iterator<Item = usize>, S: SliceArg<Ix2>>(
    a: &ArrayBase<impl Data<Elem = A>, Ix2>,
    b: &mut ArrayBase<impl DataMut<Elem = A>, Ix2>,
    row_iter_fn: impl Fn(usize) -> I,
    row_slice_fn: impl Fn(usize, usize) -> S,
) -> Result<()> {
    let rows = check_square(a)?;
    if b.nrows() != rows {
        return Err(LinalgError::WrongRows {
            expected: rows,
            actual: b.nrows(),
        });
    }
    let cols = b.ncols();

    // XXX Switching the col and row loops might lead to better cache locality for row-major
    // layouts of b
    for k in 0..cols {
        for i in row_iter_fn(rows) {
            let diag = *a.at((i, i));
            let coeff = *b.at((i, k)) / diag;
            *b.atm((i, k)) = coeff;

            b.slice_mut(row_slice_fn(i, k))
                .scaled_add(-coeff, &a.slice(row_slice_fn(i, i)));
        }
    }

    Ok(())
}

pub trait SolveTriangularInplace<B> {
    fn solve_triangular_inplace<'a>(&self, b: &'a mut B, uplo: UPLO) -> Result<&'a mut B>;
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};

    use crate::LinalgError;

    use super::*;

    #[test]
    fn corner_cases() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(empty.is_triangular(UPLO::Lower));
        assert!(empty.is_triangular(UPLO::Upper));
        assert_eq!(empty.clone().into_triangular(UPLO::Lower).unwrap(), empty);

        let one = array![[1]];
        assert!(one.is_triangular(UPLO::Lower));
        assert!(one.is_triangular(UPLO::Upper));
        assert_eq!(one.clone().into_triangular(UPLO::Upper).unwrap(), one);
        assert_eq!(one.clone().into_triangular(UPLO::Lower).unwrap(), one);
    }

    #[test]
    fn non_square() {
        let row = array![[1, 2, 3], [3, 4, 5]];
        assert!(!row.is_triangular(UPLO::Lower));
        assert!(!row.is_triangular(UPLO::Upper));
        assert!(matches!(
            row.into_triangular(UPLO::Lower),
            Err(LinalgError::NotSquare { rows: 2, cols: 3 })
        ));

        let col = array![[1, 2], [3, 5], [6, 8]];
        assert!(!col.is_triangular(UPLO::Lower));
        assert!(!col.is_triangular(UPLO::Upper));
        assert!(matches!(
            col.into_triangular(UPLO::Upper),
            Err(LinalgError::NotSquare { rows: 3, cols: 2 })
        ));
    }

    #[test]
    fn square() {
        let square = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        assert!(!square.is_triangular(UPLO::Lower));
        assert!(!square.is_triangular(UPLO::Upper));

        let upper = square.clone().into_triangular(UPLO::Upper).unwrap();
        assert_eq!(upper, array![[1, 2, 3], [0, 5, 6], [0, 0, 9]]);
        assert!(!upper.is_triangular(UPLO::Lower));
        assert!(upper.is_triangular(UPLO::Upper));

        let lower = square.into_triangular(UPLO::Lower).unwrap();
        assert!(lower.is_triangular(UPLO::Lower));
        assert!(!lower.is_triangular(UPLO::Upper));
        assert_eq!(lower, array![[1, 0, 0], [4, 5, 0], [7, 8, 9]]);
    }
}
