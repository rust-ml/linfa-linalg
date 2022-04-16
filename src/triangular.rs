//! Traits for creating and manipulating triangular matrices

use crate::{check_square, index::*, LinalgError, Result};

use ndarray::{s, Array, ArrayBase, Data, DataMut, Ix2, NdFloat, SliceArg};
use num_traits::Zero;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Denotes an upper-triangular or lower-triangular matrix
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
                    unsafe { *self.atm((i, j)) = A::zero() };
                }
            }
        } else {
            for i in 0..n {
                for j in i + 1..n {
                    unsafe { *self.atm((i, j)) = A::zero() };
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
                        if unsafe { !self.at((i, j)).is_zero() } {
                            return false;
                        }
                    }
                }
            } else {
                for i in 0..n {
                    for j in i + 1..n {
                        if unsafe { !self.at((i, j)).is_zero() } {
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
/// Index passed into `diag_fn` is guaranteed to be within the bounds of `MIN(a.nrows, a.ncols)`.
/// Ensure that the return of `diag_fn` is non-zero, otherwise output will be wrong.
pub(crate) fn solve_triangular_system<A: NdFloat, I: Iterator<Item = usize>, S: SliceArg<Ix2>>(
    a: &ArrayBase<impl Data<Elem = A>, Ix2>,
    b: &mut ArrayBase<impl DataMut<Elem = A>, Ix2>,
    row_iter_fn: impl Fn(usize) -> I,
    row_slice_fn: impl Fn(usize, usize) -> S,
    diag_fn: impl Fn(usize) -> A,
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
            let coeff;
            unsafe {
                let diag = diag_fn(i);
                coeff = *b.at((i, k)) / diag;
                *b.atm((i, k)) = coeff;
            }

            b.slice_mut(row_slice_fn(i, k))
                .scaled_add(-coeff, &a.slice(row_slice_fn(i, i)));
        }
    }

    Ok(())
}

/// Solves a triangular system
pub trait SolveTriangularInplace<B> {
    /// Solves `self * x = b` where `self` is a triangular matrix, modifying `b` into `x` in-place.
    fn solve_triangular_inplace<'a>(&self, b: &'a mut B, uplo: UPLO) -> Result<&'a mut B>;

    /// Solves `self * x = b` where `self` is a triangular matrix, consuming `b`.
    fn solve_triangular_into(&self, mut b: B, uplo: UPLO) -> Result<B> {
        self.solve_triangular_inplace(&mut b, uplo)?;
        Ok(b)
    }
}

impl<A: NdFloat, Si: Data<Elem = A>, So: DataMut<Elem = A>>
    SolveTriangularInplace<ArrayBase<So, Ix2>> for ArrayBase<Si, Ix2>
{
    fn solve_triangular_inplace<'a>(
        &self,
        b: &'a mut ArrayBase<So, Ix2>,
        uplo: UPLO,
    ) -> Result<&'a mut ArrayBase<So, Ix2>> {
        if uplo == UPLO::Upper {
            solve_triangular_system(
                self,
                b,
                |rows| (0..rows).rev(),
                |r, c| s![..r, c],
                |i| unsafe { *self.at((i, i)) },
            )?;
        } else {
            solve_triangular_system(
                self,
                b,
                |rows| (0..rows),
                |r, c| s![r + 1.., c],
                |i| unsafe { *self.at((i, i)) },
            )?;
        }
        Ok(b)
    }
}

/// Solves a triangular system
pub trait SolveTriangular<B> {
    type Output;

    /// Solves `self * x = b` where `self` is a triangular matrix.
    fn solve_triangular(&self, b: &B, uplo: UPLO) -> Result<Self::Output>;
}

impl<A: NdFloat, Si: Data<Elem = A>, So: Data<Elem = A>> SolveTriangular<ArrayBase<So, Ix2>>
    for ArrayBase<Si, Ix2>
{
    type Output = Array<A, Ix2>;

    fn solve_triangular(&self, b: &ArrayBase<So, Ix2>, uplo: UPLO) -> Result<Self::Output> {
        self.solve_triangular_into(b.to_owned(), uplo)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
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

    #[test]
    fn solve_triangular() {
        let lower = array![[1.0, 0.0], [3.0, 4.0]];
        assert!(lower.is_triangular(UPLO::Lower));
        let expected = array![[2.2, 3.1, 2.2], [1.0, 0.0, 5.7]];
        let b = lower.dot(&expected);
        let x = lower.solve_triangular_into(b, UPLO::Lower).unwrap();
        assert_abs_diff_eq!(x, expected, epsilon = 1e-7);

        let upper = array![[4.4, 2.1], [0.0, 4.3]];
        assert!(upper.is_triangular(UPLO::Upper));
        let b = upper.dot(&expected);
        let x = upper.solve_triangular_into(b, UPLO::Upper).unwrap();
        assert_abs_diff_eq!(x, expected, epsilon = 1e-7);
    }

    #[test]
    fn solve_corner_cases() {
        let empty = Array2::<f64>::zeros((0, 0));
        let out = empty.solve_triangular(&empty, UPLO::Upper).unwrap();
        assert_eq!(out.dim(), (0, 0));

        let one = Array2::<f64>::ones((1, 1));
        let out = one.solve_triangular(&one, UPLO::Upper).unwrap();
        assert_abs_diff_eq!(out, one);

        let diag_zero = array![[0., 3.], [2., 0.]];
        let zeros = Array2::<f64>::zeros((2, 2));
        diag_zero.solve_triangular(&zeros, UPLO::Lower).unwrap(); // Just make sure that zeroed diagonals won't crash
    }

    #[test]
    fn solve_error() {
        let non_square = array![[1.2f64, 3.3]];
        assert!(matches!(
            non_square
                .solve_triangular(&non_square, UPLO::Lower)
                .unwrap_err(),
            LinalgError::NotSquare { .. }
        ));

        let square = array![[1.1, 2.2], [3.3, 2.1]];
        assert!(matches!(
            square
                .solve_triangular(&array![[2.2, 3.3]], UPLO::Upper)
                .unwrap_err(),
            LinalgError::WrongRows {
                expected: 2,
                actual: 1
            }
        ));
    }
}
