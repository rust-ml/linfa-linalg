use ndarray::{ArrayViewMut1, ScalarOperand};
use num_traits::{Float, NumAssignOps};

/// Performs Householder reflection on a single column
///
/// Returns what would be the first component of column after reflection if a reflection was
/// actually performed.
fn householder_reflection_axis_mut<F: 'static + Float + NumAssignOps + ScalarOperand>(
    mut col: ArrayViewMut1<F>,
) -> Option<F> {
    let reflection_norm_sq = col.dot(&col);
    let reflection_norm = reflection_norm_sq.sqrt();

    let first = *col.get(0).unwrap();
    let signed_norm = first.signum() * reflection_norm;
    *col.get_mut(0).unwrap() += signed_norm;
    // Believe it or not, this is equal to `norm(col)^2`
    let new_norm_sq =
        (reflection_norm_sq + first.abs() * reflection_norm) * F::from(2.0f64).unwrap();

    if !new_norm_sq.is_zero() {
        col /= new_norm_sq.sqrt();
        Some(-signed_norm)
    } else {
        None
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
        let ret = householder_reflection_axis_mut(arr.view_mut()).unwrap();
        assert_abs_diff_eq!(ret, -3.90512, epsilon = 1e-4);
        assert_abs_diff_eq!(arr, array![0.8319, 0.3078, 0.4617], epsilon = 1e-4);
        assert_abs_diff_eq!(arr.dot(&arr), 1.0, epsilon = 1e-4);

        let mut arr = array![-3., 0., 0., 0.];
        let ret = householder_reflection_axis_mut(arr.view_mut()).unwrap();
        assert_abs_diff_eq!(ret, 3., epsilon = 1e-4);
        assert_abs_diff_eq!(arr, array![-1., 0., 0., 0.], epsilon = 1e-4);

        let mut arr = array![0., 0.];
        assert_eq!(householder_reflection_axis_mut(arr.view_mut()), None);
        assert_abs_diff_eq!(arr, array![0., 0.]);
    }
}
