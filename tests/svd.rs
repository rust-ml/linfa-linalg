use approx::assert_abs_diff_eq;
use ndarray::prelude::*;
use proptest::prelude::*;

use ndarray_linalg_rs::svd::*;

mod common;

fn run_svd_test(arr: Array2<f64>) {
    let (nrows, ncols) = arr.dim();
    let (u, s, vt) = arr.svd(true, true).unwrap();
    let (u, vt) = (u.unwrap(), vt.unwrap());
    assert!(s.iter().copied().all(f64::is_sign_positive));

    // U and Vt should be semi-orthogonal
    if nrows > ncols {
        assert_abs_diff_eq!(u.t().dot(&u), Array2::eye(s.len()), epsilon = 1e-7);
    } else {
        assert_abs_diff_eq!(u.dot(&u.t()), Array2::eye(s.len()), epsilon = 1e-7);
    }
    assert_abs_diff_eq!(vt.dot(&vt.t()), Array2::eye(s.len()), epsilon = 1e-7);

    // U * S * Vt should equal original array
    assert_abs_diff_eq!(u.dot(&Array2::from_diag(&s)).dot(&vt), arr, epsilon = 1e-7);

    let (u2, s2, vt2) = arr.svd(false, true).unwrap();
    assert!(u2.is_none());
    assert_abs_diff_eq!(s2, s, epsilon = 1e-9);
    assert_abs_diff_eq!(vt2.unwrap(), vt, epsilon = 1e-9);

    let (u3, s3, vt3) = arr.svd(true, false).unwrap();
    assert!(vt3.is_none());
    assert_abs_diff_eq!(s3, s, epsilon = 1e-9);
    assert_abs_diff_eq!(u3.unwrap(), u, epsilon = 1e-9);

    let (u4, s4, vt4) = arr.svd(false, false).unwrap();
    assert!(vt4.is_none());
    assert!(u4.is_none());
    assert_abs_diff_eq!(s4, s, epsilon = 1e-9);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    #[test]
    fn bidiagonal_test(arr in common::rect_arr()) {
        run_svd_test(arr);
    }
}
