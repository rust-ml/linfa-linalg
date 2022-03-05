//! Traits for creating and manipulating triangular matrices

use crate::{check_square, index::*, LinalgError, Result};

use ndarray::{s, ArrayBase, Data, DataMut, Ix2, NdFloat, SliceArg};
use num_traits::Zero;

/// Transform square matrix into triangular matrix
pub trait IntoTriangular {
    /// Transform square matrix into a strict upper triangular matrix in place, zeroing out the
    /// lower elements.
    fn upper_triangular_inplace(&mut self) -> Result<&mut Self>;
    /// Transform square matrix into a strict lower triangular matrix in place, zeroing out the
    /// upper elements.
    fn lower_triangular_inplace(&mut self) -> Result<&mut Self>;

    /// Transform square matrix into a strict upper triangular matrix, zeroing out the lower
    /// elements.
    fn into_upper_triangular(self) -> Result<Self>
    where
        Self: Sized;
    /// Transform square matrix into a strict lower triangular matrix, zeroing out the upper
    /// elements.
    fn into_lower_triangular(self) -> Result<Self>
    where
        Self: Sized;
}

impl<A, S> IntoTriangular for ArrayBase<S, Ix2>
where
    A: Zero,
    S: DataMut<Elem = A>,
{
    fn into_upper_triangular(mut self) -> Result<Self> {
        self.upper_triangular_inplace()?;
        Ok(self)
    }

    fn into_lower_triangular(mut self) -> Result<Self> {
        self.lower_triangular_inplace()?;
        Ok(self)
    }

    fn upper_triangular_inplace(&mut self) -> Result<&mut Self> {
        let n = check_square(self)?;
        for i in 0..n {
            for j in 0..i {
                *self.atm((i, j)) = A::zero();
            }
        }
        Ok(self)
    }

    fn lower_triangular_inplace(&mut self) -> Result<&mut Self> {
        let n = check_square(self)?;
        for i in 0..n {
            for j in i + 1..n {
                *self.atm((i, j)) = A::zero();
            }
        }
        Ok(self)
    }
}

/// Operations on triangular matrices
pub trait Triangular {
    /// Check if matrix is upper-triangular
    fn is_upper_triangular(&self) -> bool;
    /// Check if matrix is lower-triangular
    fn is_lower_triangular(&self) -> bool;
}

impl<A, S> Triangular for ArrayBase<S, Ix2>
where
    A: Zero,
    S: Data<Elem = A>,
{
    fn is_upper_triangular(&self) -> bool {
        if let Ok(n) = check_square(self) {
            for i in 0..n {
                for j in 0..i {
                    if !self.at((i, j)).is_zero() {
                        return false;
                    }
                }
            }
            true
        } else {
            false
        }
    }

    fn is_lower_triangular(&self) -> bool {
        if let Ok(n) = check_square(self) {
            for i in 0..n {
                for j in i + 1..n {
                    if !self.at((i, j)).is_zero() {
                        return false;
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
    fn solve_upper_triangular_inplace<'a>(&self, b: &'a mut B) -> Result<&'a mut B>;
    fn solve_lower_triangular_inplace<'a>(&self, b: &'a mut B) -> Result<&'a mut B>;
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};

    use crate::LinalgError;

    use super::*;

    #[test]
    fn corner_cases() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(empty.is_lower_triangular());
        assert!(empty.is_upper_triangular());
        assert_eq!(empty.clone().into_lower_triangular().unwrap(), empty);

        let one = array![[1]];
        assert!(one.is_lower_triangular());
        assert!(one.is_upper_triangular());
        assert_eq!(one.clone().into_upper_triangular().unwrap(), one);
        assert_eq!(one.clone().into_lower_triangular().unwrap(), one);
    }

    #[test]
    fn non_square() {
        let row = array![[1, 2, 3], [3, 4, 5]];
        assert!(!row.is_lower_triangular());
        assert!(!row.is_upper_triangular());
        assert!(matches!(
            row.into_lower_triangular(),
            Err(LinalgError::NotSquare { rows: 2, cols: 3 })
        ));

        let col = array![[1, 2], [3, 5], [6, 8]];
        assert!(!col.is_lower_triangular());
        assert!(!col.is_upper_triangular());
        assert!(matches!(
            col.into_upper_triangular(),
            Err(LinalgError::NotSquare { rows: 3, cols: 2 })
        ));
    }

    #[test]
    fn square() {
        let square = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        assert!(!square.is_lower_triangular());
        assert!(!square.is_upper_triangular());

        let upper = square.clone().into_upper_triangular().unwrap();
        assert_eq!(upper, array![[1, 2, 3], [0, 5, 6], [0, 0, 9]]);
        assert!(!upper.is_lower_triangular());
        assert!(upper.is_upper_triangular());

        let lower = square.into_lower_triangular().unwrap();
        assert!(lower.is_lower_triangular());
        assert!(!lower.is_upper_triangular());
        assert_eq!(lower, array![[1, 0, 0], [4, 5, 0], [7, 8, 9]]);
    }
}
