//! Tridiagonal decomposition of a symmetric matrix

use ndarray::{
    linalg::{general_mat_mul, general_mat_vec_mul},
    s, Array1, Array2, ArrayBase, Axis, DataMut, Ix2, NdFloat, RawDataClone,
};

use crate::{
    check_square, householder,
    triangular::{IntoTriangular, UPLO},
    LinalgError, Result,
};

/// Tridiagonal decomposition of a non-empty symmetric matrix
pub trait SymmetricTridiagonal {
    type Decomp;

    /// Calculate the tridiagonal decomposition of a symmetric matrix, consisting of symmetric
    /// tridiagonal matrix `T` and orthogonal matrix `Q`, such that `Q * T * Q.t` yields the
    /// original matrix.
    fn sym_tridiagonal(self) -> Result<Self::Decomp>;
}

impl<S, A> SymmetricTridiagonal for ArrayBase<S, Ix2>
where
    A: NdFloat,
    S: DataMut<Elem = A>,
{
    type Decomp = TridiagonalDecomp<A, S>;

    fn sym_tridiagonal(mut self) -> Result<Self::Decomp> {
        let n = check_square(&self)?;
        if n < 1 {
            return Err(LinalgError::EmptyMatrix);
        }

        let mut off_diagonal = Array1::zeros(n - 1); // TODO can be uninit
        let mut p = Array1::zeros(n - 1);

        for i in 0..n - 1 {
            let mut m = self.slice_mut(s![i + 1.., ..]);
            let (mut axis, mut m) = m.multi_slice_mut((s![.., i], s![.., i + 1..]));

            let norm = householder::reflection_axis_mut(&mut axis);
            *off_diagonal.get_mut(i).unwrap() = norm.unwrap_or_else(A::zero);

            if norm.is_some() {
                let mut p = p.slice_mut(s![i..]);
                general_mat_vec_mul(A::from(2.0f64).unwrap(), &m, &axis, A::zero(), &mut p);
                let dot = axis.dot(&p);

                let p_row = p.view().insert_axis(Axis(0));
                let p_col = p.view().insert_axis(Axis(1));
                let ax_row = axis.view().insert_axis(Axis(0));
                let ax_col = axis.view().insert_axis(Axis(1));
                general_mat_mul(-A::one(), &p_col, &ax_row, A::one(), &mut m);
                general_mat_mul(-A::one(), &ax_col, &p_row, A::one(), &mut m);
                general_mat_mul(dot + dot, &ax_col, &ax_row, A::one(), &mut m);
            }
        }

        Ok(TridiagonalDecomp {
            diag_matrix: self,
            off_diagonal,
        })
    }
}

/// Full tridiagonal decomposition, containing the symmetric tridiagonal matrix `T`
#[derive(Debug)]
pub struct TridiagonalDecomp<A, S: DataMut<Elem = A>> {
    // This matrix is only useful for its diagonal, which is the diagonal of the tridiagonal matrix
    // Guaranteed to be square matrix
    diag_matrix: ArrayBase<S, Ix2>,
    // The off-diagonal elements of the tridiagonal matrix
    off_diagonal: Array1<A>,
}

impl<A: Clone, S: DataMut<Elem = A> + RawDataClone> Clone for TridiagonalDecomp<A, S> {
    fn clone(&self) -> Self {
        Self {
            diag_matrix: self.diag_matrix.clone(),
            off_diagonal: self.off_diagonal.clone(),
        }
    }
}

impl<A: NdFloat, S: DataMut<Elem = A>> TridiagonalDecomp<A, S> {
    /// Construct the orthogonal matrix `Q`, where `Q * T * Q.t` results in the original matrix
    pub fn generate_q(&self) -> Array2<A> {
        householder::assemble_q(&self.diag_matrix, 1, |i| self.off_diagonal[i].signum())
    }

    /// Return the diagonal elements and off-diagonal elements of the tridiagonal matrix as 1D
    /// arrays
    pub fn into_diagonals(self) -> (Array1<A>, Array1<A>) {
        (
            self.diag_matrix.diag().to_owned(),
            self.off_diagonal.mapv_into(A::abs),
        )
    }

    /// Return the full tridiagonal matrix `T`
    pub fn into_tridiag_matrix(mut self) -> ArrayBase<S, Ix2> {
        self.diag_matrix.triangular_inplace(UPLO::Upper).unwrap();
        self.diag_matrix.triangular_inplace(UPLO::Lower).unwrap();
        for (i, off) in self.off_diagonal.into_iter().enumerate() {
            let off = off.abs();
            self.diag_matrix[(i + 1, i)] = off;
            self.diag_matrix[(i, i + 1)] = off;
        }
        self.diag_matrix
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn sym_tridiagonal() {
        let arr = array![
            [4.0f64, 1., -2., 2.],
            [1., 2., 0., 1.],
            [-2., 0., 3., -2.],
            [2., 1., -2., -1.]
        ];

        let decomp = arr.clone().sym_tridiagonal().unwrap();
        let (diag, offdiag) = decomp.into_diagonals();
        assert_abs_diff_eq!(
            diag,
            array![4., 10. / 3., -33. / 25., 149. / 75.],
            epsilon = 1e-5
        );
        assert_abs_diff_eq!(offdiag, array![3., 5. / 3., 68. / 75.], epsilon = 1e-5);

        let decomp = arr.clone().sym_tridiagonal().unwrap();
        let q = decomp.generate_q();
        let tri = decomp.into_tridiag_matrix();
        assert_abs_diff_eq!(q.dot(&tri).dot(&q.t()), arr, epsilon = 1e-9);
        // Q must be orthogonal
        assert_abs_diff_eq!(q.dot(&q.t()), Array2::eye(4), epsilon = 1e-9);

        let one = array![[1.1f64]].sym_tridiagonal().unwrap();
        let (one_diag, one_offdiag) = one.into_diagonals();
        assert_abs_diff_eq!(one_diag, array![1.1f64]);
        assert!(one_offdiag.is_empty());
    }

    #[test]
    fn sym_tridiag_error() {
        assert!(matches!(
            array![[1., 2., 3.], [5., 4., 3.0f64]].sym_tridiagonal(),
            Err(LinalgError::NotSquare { rows: 2, cols: 3 })
        ));
        assert!(matches!(
            Array2::<f64>::zeros((0, 0)).sym_tridiagonal(),
            Err(LinalgError::EmptyMatrix)
        ));
    }
}
