//! Tridiagonal decomposition of a symmetric matrix

use ndarray::{
    linalg::{general_mat_mul, general_mat_vec_mul},
    s, Array1, Array2, ArrayBase, Axis, DataMut, Ix1, Ix2, NdFloat,
};

use crate::{
    check_square, index::*, reflection::Reflection, triangular::IntoTriangular, LinalgError, Result,
};

/// Performs Householder reflection on a single column
///
/// Returns what would be the first component of column after reflection if a reflection was
/// actually performed.
fn householder_reflection_axis_mut<A: NdFloat, S: DataMut<Elem = A>>(
    col: &mut ArrayBase<S, Ix1>,
) -> Option<A> {
    let reflection_norm_sq = col.dot(col);
    let reflection_norm = reflection_norm_sq.sqrt();

    let first = *col.get(0).unwrap();
    let signed_norm = first.signum() * reflection_norm;
    *col.get_mut(0).unwrap() += signed_norm;
    // Believe it or not, this is equal to `norm(col)^2`
    let new_norm_sq =
        (reflection_norm_sq + first.abs() * reflection_norm) * A::from(2.0f64).unwrap();

    if !new_norm_sq.is_zero() {
        *col /= new_norm_sq.sqrt();
        Some(-signed_norm)
    } else {
        None
    }
}

/// Tridiagonal decomposition of a symmetric matrix
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

            let norm = householder_reflection_axis_mut(&mut axis);
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

impl<A: NdFloat, S: DataMut<Elem = A>> TridiagonalDecomp<A, S> {
    /// Construct the orthogonal matrix `Q`, where `Q * T * Q.t` results in the original matrix
    pub fn generate_q(&self) -> Array2<A> {
        let n = self.diag_matrix.nrows();

        let mut q_matrix = Array2::eye(n);
        for i in (0..n - 1).rev() {
            let axis = self.diag_matrix.slice(s![i + 1.., i]);
            let refl = Reflection::new(axis, A::zero());

            let mut q_rows = q_matrix.slice_mut(s![i + 1.., i..]);
            refl.reflect_col(&mut q_rows);
            q_rows *= self.off_diagonal.at(i).signum();
        }

        q_matrix
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
        self.diag_matrix.upper_triangular_inplace().unwrap();
        self.diag_matrix.lower_triangular_inplace().unwrap();
        for (i, off) in self.off_diagonal.into_iter().enumerate() {
            let off = off.abs();
            let off1 = self.diag_matrix.atm((i + 1, i));
            *off1 = off;
            let off2 = self.diag_matrix.atm((i, i + 1));
            *off2 = off;
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
    fn householder() {
        let mut arr = array![1.5f64, 2., 3.];
        let ret = householder_reflection_axis_mut(&mut arr).unwrap();
        assert_abs_diff_eq!(ret, -3.90512, epsilon = 1e-4);
        assert_abs_diff_eq!(arr, array![0.8319, 0.3078, 0.4617], epsilon = 1e-4);
        assert_abs_diff_eq!(arr.dot(&arr), 1.0, epsilon = 1e-4);

        let mut arr = array![-3., 0., 0., 0.];
        let ret = householder_reflection_axis_mut(&mut arr).unwrap();
        assert_abs_diff_eq!(ret, 3., epsilon = 1e-4);
        assert_abs_diff_eq!(arr, array![-1., 0., 0., 0.], epsilon = 1e-4);

        let mut arr = array![0., 0.];
        assert_eq!(householder_reflection_axis_mut(&mut arr), None);
        assert_abs_diff_eq!(arr, array![0., 0.]);
    }

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
