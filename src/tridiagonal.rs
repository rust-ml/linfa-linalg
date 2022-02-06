use ndarray::{
    linalg::{general_mat_mul, general_mat_vec_mul},
    s, Array1, Array2, ArrayBase, Data, DataMut, Ix1, Ix2,
};

use crate::reflection::Reflection;
use crate::{check_square, Float, LinalgError, Result};

/// Performs Householder reflection on a single column
///
/// Returns what would be the first component of column after reflection if a reflection was
/// actually performed.
fn householder_reflection_axis_mut<A: Float, S: DataMut<Elem = A>>(
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

pub trait SymmetricTridiagonal<T> {
    fn sym_tridiagonal_inplace(&mut self) -> Result<T>;

    fn sym_tridiagonal_into(mut self) -> Result<(Self, T)>
    where
        Self: Sized,
    {
        let out = self.sym_tridiagonal_inplace()?;
        Ok((self, out))
    }
}

impl<S, A> SymmetricTridiagonal<Array1<A>> for ArrayBase<S, Ix2>
where
    A: Float,
    S: DataMut<Elem = A>,
{
    fn sym_tridiagonal_inplace(&mut self) -> Result<Array1<A>> {
        let n = check_square(self)?;
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

                let mlen = m.nrows();
                let p_row = p.view().into_shape((1, mlen))?;
                let p_col = p.view().into_shape((mlen, 1))?;
                let ax_row = axis.view().into_shape((1, mlen))?;
                let ax_col = axis.view().into_shape((mlen, 1))?;
                general_mat_mul(-A::one(), &p_col, &ax_row, A::one(), &mut m);
                general_mat_mul(-A::one(), &ax_col, &p_row, A::one(), &mut m);
                general_mat_mul(dot + dot, &ax_col, &ax_row, A::one(), &mut m);
            }
        }

        Ok(off_diagonal)
    }
}

/// Full tridiagonal decomposition, including the reconstructed Q matrix
pub struct TridiagonalDecomp<A> {
    pub off_diagonal: Array1<A>,
    pub q_matrix: Array2<A>,
}

impl<A: Float> TridiagonalDecomp<A> {
    fn from_off_diagonal<D: Data<Elem = A>>(
        m: &ArrayBase<D, Ix2>,
        off_diagonal: Array1<A>,
    ) -> Result<Self> {
        let n = check_square(m)?;

        let mut q_matrix = Array2::eye(n);
        for i in (0..n - 1).rev() {
            let axis = m.slice(s![i + 1.., i]);
            let refl = Reflection::new(axis, A::zero());

            let mut q_rows = q_matrix.slice_mut(s![i + 1.., i..]);
            refl.reflect_col(&mut q_rows);
            q_rows *= off_diagonal[i].signum();
        }

        Ok(Self {
            off_diagonal,
            q_matrix,
        })
    }
}

impl<S, A> SymmetricTridiagonal<TridiagonalDecomp<A>> for ArrayBase<S, Ix2>
where
    A: Float,
    S: DataMut<Elem = A>,
{
    fn sym_tridiagonal_inplace(&mut self) -> Result<TridiagonalDecomp<A>> {
        let off_diagonal = self.sym_tridiagonal_inplace()?;
        let decomp = TridiagonalDecomp::from_off_diagonal(self, off_diagonal)?;
        Ok(decomp)
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
}
