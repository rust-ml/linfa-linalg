//! Traits for creating and manipulating triangular matrices

use crate::{check_square, index::UncheckedIndexMut, Result};

use ndarray::{ArrayBase, DataMut, Ix2};
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

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};

    use crate::LinalgError;

    use super::*;

    #[test]
    fn corner_cases() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert_eq!(empty.clone().into_lower_triangular().unwrap(), empty);

        let one = array![[1]];
        assert_eq!(one.clone().into_upper_triangular().unwrap(), one);
        assert_eq!(one.clone().into_lower_triangular().unwrap(), one);
    }

    #[test]
    fn non_square() {
        let row = array![[1, 2, 3], [3, 4, 5]];
        assert!(matches!(
            row.into_lower_triangular(),
            Err(LinalgError::NotSquare { rows: 2, cols: 3 })
        ));

        let col = array![[1, 2], [3, 5], [6, 8]];
        assert!(matches!(
            col.into_upper_triangular(),
            Err(LinalgError::NotSquare { rows: 3, cols: 2 })
        ));
    }

    #[test]
    fn square() {
        let square = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        assert_eq!(
            square.clone().into_upper_triangular().unwrap(),
            array![[1, 2, 3], [0, 5, 6], [0, 0, 9]]
        );
        assert_eq!(
            square.into_lower_triangular().unwrap(),
            array![[1, 0, 0], [4, 5, 0], [7, 8, 9]]
        );
    }
}
