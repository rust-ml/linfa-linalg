//! Compact bidiagonal decomposition for matrices

use ndarray::{s, Array1, Array2, ArrayBase, DataMut, Ix2, NdFloat, RawDataClone};

use crate::{
    householder::{assemble_q, clear_column, clear_row},
    LinalgError, Result,
};

/// Compact bidiagonal decomposition of a non-empty matrix
pub trait Bidiagonal {
    type Decomp;

    /// Calculate the compact bidiagonal decomposition of a matrix, consisting of square bidiagonal
    /// matrix `B` and rectangular semi-orthogonal matrices `U` and `Vt`, such that `U * B * Vt`
    /// yields the original matrix.
    fn bidiagonal(self) -> Result<Self::Decomp>;
}

impl<S, A> Bidiagonal for ArrayBase<S, Ix2>
where
    A: NdFloat,
    S: DataMut<Elem = A>,
{
    type Decomp = BidiagonalDecomp<A, S>;

    fn bidiagonal(mut self) -> Result<Self::Decomp> {
        let (nrows, ncols) = self.dim();
        let min_dim = nrows.min(ncols);
        if min_dim == 0 {
            return Err(LinalgError::EmptyMatrix);
        }

        // XXX diagonal and off_diagonal could be uninit
        let mut diagonal = Array1::zeros(min_dim);
        let mut off_diagonal = Array1::zeros(min_dim - 1);

        let upper_diag = nrows >= ncols;
        if upper_diag {
            for i in 0..min_dim - 1 {
                diagonal[i] = clear_column(&mut self, i, 0);
                off_diagonal[i] = clear_row(&mut self, i, 1);
            }
            diagonal[min_dim - 1] = clear_column(&mut self, min_dim - 1, 0);
        } else {
            for i in 0..min_dim - 1 {
                diagonal[i] = clear_row(&mut self, i, 0);
                off_diagonal[i] = clear_column(&mut self, i, 1);
            }
            diagonal[min_dim - 1] = clear_row(&mut self, min_dim - 1, 0);
        }

        Ok(BidiagonalDecomp {
            uv: self,
            diagonal,
            off_diagonal,
            upper_diag,
        })
    }
}

#[derive(Debug)]
/// Full bidiagonal decomposition
pub struct BidiagonalDecomp<A, S: DataMut<Elem = A>> {
    uv: ArrayBase<S, Ix2>,
    off_diagonal: Array1<A>,
    diagonal: Array1<A>,
    upper_diag: bool,
}

impl<A: Clone, S: DataMut<Elem = A> + RawDataClone> Clone for BidiagonalDecomp<A, S> {
    fn clone(&self) -> Self {
        Self {
            uv: self.uv.clone(),
            off_diagonal: self.off_diagonal.clone(),
            diagonal: self.diagonal.clone(),
            upper_diag: self.upper_diag,
        }
    }
}

impl<A: NdFloat, S: DataMut<Elem = A>> BidiagonalDecomp<A, S> {
    /// Whether `B` is upper-bidiagonal or not
    pub fn is_upper_diag(&self) -> bool {
        self.upper_diag
    }

    /// Generates `U` matrix, which is R x min(R, C), where R and C are dimensions of the original
    /// matrix
    pub fn generate_u(&self) -> Array2<A> {
        let shift = !self.upper_diag as usize;
        if self.upper_diag {
            assemble_q(&self.uv, shift, |i| self.diagonal[i])
        } else {
            assemble_q(&self.uv, shift, |i| self.off_diagonal[i])
        }
    }

    /// Generates `Vt` matrix, which is min(R, C) x C, where R and C are dimensions of the original
    /// matrix
    pub fn generate_vt(&self) -> Array2<A> {
        let shift = self.upper_diag as usize;
        if self.upper_diag {
            assemble_q(&self.uv.t(), shift, |i| self.off_diagonal[i])
        } else {
            assemble_q(&self.uv.t(), shift, |i| self.diagonal[i])
        }
        .reversed_axes()
    }

    /// Returns `B` matrix, which is min(R, C) x min(R, C), where R and C are dimensions of the
    /// original matrix
    pub fn into_b(self) -> Array2<A> {
        let d = self.diagonal.len();
        let (r, c) = if self.upper_diag { (0, 1) } else { (1, 0) };
        let (diagonal, off_diagonal) = self.into_diagonals();
        let mut res = Array2::from_diag(&diagonal);

        res.slice_mut(s![r..d, c..d])
            .diag_mut()
            .assign(&off_diagonal);
        res
    }

    /// Returns the diagonal and off-diagonal elements of `B` as 1D arrays
    pub fn into_diagonals(self) -> (Array1<A>, Array1<A>) {
        (
            self.diagonal.mapv_into(A::abs),
            self.off_diagonal.mapv_into(A::abs),
        )
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn bidiagonal_lower() {
        let arr = array![
            [4.0f64, 0., 2., 2.],
            [-2., 6., 3., -2.],
            [2., 7., -3.2, -1.]
        ];
        let decomp = arr.clone().bidiagonal().unwrap();
        let u = decomp.generate_u();
        let vt = decomp.generate_vt();
        let b = decomp.clone().into_b();
        let (diag, offdiag) = decomp.into_diagonals();

        assert_eq!(u.dim(), (3, 3));
        assert_eq!(b.dim(), (3, 3));
        assert_eq!(vt.dim(), (3, 4));
        assert_abs_diff_eq!(u.dot(&u.t()), Array2::eye(3), epsilon = 1e-5);
        assert_abs_diff_eq!(vt.dot(&vt.t()), Array2::eye(3), epsilon = 1e-5);
        assert_abs_diff_eq!(u.dot(&b).dot(&vt), arr, epsilon = 1e-5);

        assert_abs_diff_eq!(diag, b.diag());
        let partial = b.slice(s![1.., 0..]);
        assert_abs_diff_eq!(offdiag, partial.diag());
    }

    #[test]
    fn bidiagonal_upper() {
        let arr = array![
            [4.0f64, 0., 2.],
            [-2., 6., 3.],
            [2., 7., -3.2],
            [4., -3., 0.2]
        ];
        let decomp = arr.clone().bidiagonal().unwrap();
        let u = decomp.generate_u();
        let vt = decomp.generate_vt();
        let b = decomp.clone().into_b();
        let (diag, offdiag) = decomp.into_diagonals();

        assert_eq!(u.dim(), (4, 3));
        assert_eq!(b.dim(), (3, 3));
        assert_eq!(vt.dim(), (3, 3));
        assert_abs_diff_eq!(u.t().dot(&u), Array2::eye(3), epsilon = 1e-5);
        assert_abs_diff_eq!(vt.dot(&vt.t()), Array2::eye(3), epsilon = 1e-5);
        assert_abs_diff_eq!(u.dot(&b).dot(&vt), arr, epsilon = 1e-5);

        assert_abs_diff_eq!(diag, b.diag());
        let partial = b.slice(s![0.., 1..]);
        assert_abs_diff_eq!(offdiag, partial.diag());
    }

    #[test]
    fn bidiagonal_error() {
        assert!(matches!(
            Array2::<f64>::zeros((0, 0)).bidiagonal(),
            Err(LinalgError::EmptyMatrix)
        ));
    }
}
