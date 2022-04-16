use std::ops::AddAssign;

use ndarray::{s, Array2, ArrayBase, Data, DataMut, Ix1, Ix2, NdFloat};

use crate::reflection::Reflection;

/// Performs Householder reflection on a single column
///
/// Returns what would be the first component of column after reflection if a reflection was
/// actually performed.
pub fn reflection_axis_mut<A: NdFloat, S: DataMut<Elem = A>>(
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

/// Uses an householder reflection to zero out the `icol`-th column, starting with the `shift + 1`-th
/// subdiagonal element.
///
/// Returns the signed norm of the column.
pub fn clear_column<A: NdFloat, S: DataMut<Elem = A>>(
    matrix: &mut ArrayBase<S, Ix2>,
    icol: usize,
    shift: usize,
) -> A {
    let (mut left, mut right) = matrix.multi_slice_mut((s![.., icol], s![.., icol + 1..]));
    let mut axis = left.slice_mut(s![icol + shift..]);
    let refl_norm = reflection_axis_mut(&mut axis);

    if let Some(refl_norm) = refl_norm {
        let refl = Reflection::new(axis, A::zero());
        let sign = refl_norm.signum();
        let mut refl_rows = right.slice_mut(s![icol + shift.., ..]);
        refl.reflect_cols(&mut refl_rows);
        refl_rows *= sign;
    }
    refl_norm.unwrap_or_else(A::zero)
}

/// Uses an householder reflection to zero out the `irow`-th row, ending before the `shift + 1`-th
/// superdiagonal element.
///
/// Returns the signed norm of the column.
pub fn clear_row<A: NdFloat>(
    matrix: &mut ArrayBase<impl DataMut<Elem = A>, Ix2>,
    irow: usize,
    shift: usize,
) -> A {
    clear_column(&mut matrix.view_mut().reversed_axes(), irow, shift)
}

/// Used to assemble `Q` for tridiagonal decompositions and `U` for bidiagonal decompositions.
///
/// Panics if `shift` exceeds either dimension of `matrix`.
pub fn assemble_q<A: NdFloat, S: Data<Elem = A>>(
    matrix: &ArrayBase<S, Ix2>,
    shift: usize,
    sign_fn: impl Fn(usize) -> A,
) -> Array2<A> {
    let (nrows, ncols) = matrix.dim();
    let dim = nrows.min(ncols);
    let mut res = if nrows == ncols {
        Array2::eye(nrows)
    } else {
        let mut a = Array2::zeros((nrows, dim));
        a.diag_mut().fill(A::one());
        a
    };

    for i in (0..dim - shift).rev() {
        let axis = matrix.slice(s![i + shift.., i]);
        let refl = Reflection::new(axis, A::zero());

        let mut res_rows = res.slice_mut(s![i + shift.., i..]);
        refl.reflect_cols(&mut res_rows);
        res_rows *= sign_fn(i);
    }

    res
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn householder() {
        let mut arr = array![1.5f64, 2., 3.];
        let ret = reflection_axis_mut(&mut arr).unwrap();
        assert_abs_diff_eq!(ret, -3.90512, epsilon = 1e-4);
        assert_abs_diff_eq!(arr, array![0.8319, 0.3078, 0.4617], epsilon = 1e-4);
        assert_abs_diff_eq!(arr.dot(&arr), 1.0, epsilon = 1e-4);

        let mut arr = array![-3., 0., 0., 0.];
        let ret = reflection_axis_mut(&mut arr).unwrap();
        assert_abs_diff_eq!(ret, 3., epsilon = 1e-4);
        assert_abs_diff_eq!(arr, array![-1., 0., 0., 0.], epsilon = 1e-4);

        let mut arr = array![0., 0.];
        assert_eq!(reflection_axis_mut(&mut arr), None);
        assert_abs_diff_eq!(arr, array![0., 0.]);
    }
}
