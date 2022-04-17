use approx::assert_abs_diff_eq;
use ndarray::prelude::*;
use proptest::prelude::*;

use ndarray_linalg_rs::{qr::*, triangular::*};

mod common;

fn run_qr_test(orig: Array2<f64>) {
    let (q, r) = orig.qr().unwrap().into_decomp();
    assert_abs_diff_eq!(q.t().dot(&q), Array2::eye(q.ncols()), epsilon = 1e-7);
    assert!(r.is_triangular(UPLO::Upper));
    assert_abs_diff_eq!(q.dot(&r), orig, epsilon = 1e-7);
}

fn run_inv_test(orig: Array2<f64>) {
    let qr = orig.qr().unwrap();
    if qr.is_invertible() {
        let inv = qr.inverse().unwrap();
        assert_abs_diff_eq!(orig.dot(&inv), Array2::eye(orig.nrows()), epsilon = 1e-7);
        assert_abs_diff_eq!(inv.dot(&orig), Array2::eye(orig.nrows()), epsilon = 1e-7);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    #[test]
    fn qr_test(arr in common::thin_arr()) {
        run_qr_test(arr)
    }

    #[test]
    fn inv_qr_test(arr in common::square_arr()) {
        run_inv_test(arr)
    }
}
