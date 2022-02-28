use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
use ndarray::Array2;
use proptest::prelude::*;

use ndarray_linalg_rs::{cholesky::*, triangular::*};

const FLOAT_LIMIT: f64 = 1000.0;

prop_compose! {
    fn square_arr(dim: usize)
        (data in prop::collection::vec(-FLOAT_LIMIT..FLOAT_LIMIT, dim*dim)) -> Array2<f64> {
        Array2::from_shape_vec((dim, dim), data).unwrap()
    }
}

prop_compose! {
    fn hpd_arr(dim: usize)
        (arr in square_arr(dim)) -> Array2<f64> {
        let mut mul = arr.t().dot(&arr);
        for i in 0..dim {
            mul[(i, i)] += 1.0;
        }
        mul
    }
}

fn cholesky_test(orig: Array2<f64>) {
    let chol = orig.cholesky().unwrap();
    assert_abs_diff_eq!(chol.dot(&chol.t()), orig, epsilon = 1e-7);
    let dirty = orig.cholesky_dirty().unwrap();
    assert_abs_diff_ne!(chol, dirty, epsilon = 1e-7);
    assert_abs_diff_eq!(chol, dirty.into_lower_triangular().unwrap(), epsilon = 1e-7);

    let chol = orig.clone().cholesky_into().unwrap();
    assert_abs_diff_eq!(chol.dot(&chol.t()), orig, epsilon = 1e-7);
    let dirty = orig.clone().cholesky_into_dirty().unwrap();
    assert_abs_diff_ne!(chol, dirty, epsilon = 1e-7);
    assert_abs_diff_eq!(chol, dirty.into_lower_triangular().unwrap(), epsilon = 1e-7);

    let mut a = orig.clone();
    let chol = a.cholesky_inplace().unwrap();
    assert_abs_diff_eq!(chol.dot(&chol.t()), orig, epsilon = 1e-7);
    assert_abs_diff_eq!(a.dot(&a.t()), orig, epsilon = 1e-7);
    let mut b = orig;
    let dirty = b.cholesky_inplace_dirty().unwrap();
    assert_abs_diff_ne!(a, dirty, epsilon = 1e-7);
    assert_abs_diff_eq!(a, dirty.lower_triangular_inplace().unwrap(), epsilon = 1e-7);
}

proptest! {
    #[test]
    fn cholesky_test3(arr in hpd_arr(3)) {
        cholesky_test(arr)
    }

    #[test]
    fn cholesky_test4(arr in hpd_arr(4)) {
        cholesky_test(arr)
    }

    #[test]
    fn cholesky_test5(arr in hpd_arr(5)) {
        cholesky_test(arr)
    }
}
