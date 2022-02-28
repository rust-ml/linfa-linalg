use approx::assert_abs_diff_eq;
use ndarray::Array2;
use proptest::prelude::*;

use ndarray_linalg_rs::{cholesky::*, triangular::*};

mod common;

prop_compose! {
    fn hpd_arr()
        (arr in common::square_arr()) -> Array2<f64> {
        let dim = arr.nrows();
        let mut mul = arr.t().dot(&arr);
        for i in 0..dim {
            mul[(i, i)] += 1.0;
        }
        mul
    }
}

fn run_cholesky_test(orig: Array2<f64>) {
    let chol = orig.cholesky().unwrap();
    assert_abs_diff_eq!(chol.dot(&chol.t()), orig, epsilon = 1e-7);
    let dirty = orig.cholesky_dirty().unwrap();
    assert!(chol.is_lower_triangular());
    assert_abs_diff_eq!(chol, dirty.into_lower_triangular().unwrap(), epsilon = 1e-7);

    let chol = orig.clone().cholesky_into().unwrap();
    assert_abs_diff_eq!(chol.dot(&chol.t()), orig, epsilon = 1e-7);
    let dirty = orig.clone().cholesky_into_dirty().unwrap();
    assert!(chol.is_lower_triangular());
    assert_abs_diff_eq!(chol, dirty.into_lower_triangular().unwrap(), epsilon = 1e-7);

    let mut a = orig.clone();
    let chol = a.cholesky_inplace().unwrap();
    assert_abs_diff_eq!(chol.dot(&chol.t()), orig, epsilon = 1e-7);
    assert_abs_diff_eq!(a.dot(&a.t()), orig, epsilon = 1e-7);
    let mut b = orig;
    let dirty = b.cholesky_inplace_dirty().unwrap();
    assert!(a.is_lower_triangular());
    assert_abs_diff_eq!(a, dirty.lower_triangular_inplace().unwrap(), epsilon = 1e-7);
}

proptest! {
    #[test]
    fn cholesky_test(arr in hpd_arr()) {
        run_cholesky_test(arr)
    }
}
