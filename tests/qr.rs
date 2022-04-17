use approx::assert_abs_diff_eq;
use ndarray::prelude::*;
use proptest::prelude::*;

use ndarray_linalg_rs::{qr::*, triangular::*, LinalgError};

mod common;

fn run_qr_test(orig: Array2<f64>) {
    let (q, r) = orig.qr().unwrap().into_decomp();
    assert_abs_diff_eq!(q.t().dot(&q), Array2::eye(q.ncols()), epsilon = 1e-7);
    assert!(r.is_triangular(UPLO::Upper));
    assert_abs_diff_eq!(q.dot(&r), orig, epsilon = 1e-7);
}

fn run_inv_test(orig: Array2<f64>) {
    let qr = orig.qr().unwrap();
    let inv = match qr.inverse() {
        Ok(inv) => inv,
        Err(LinalgError::NonInvertible) => return,
        Err(err) => panic!("Unexpected error: {}", err),
    };
    assert_abs_diff_eq!(orig.dot(&inv), Array2::eye(orig.nrows()), epsilon = 1e-7);
    assert_abs_diff_eq!(inv.dot(&orig), Array2::eye(orig.nrows()), epsilon = 1e-7);
}

fn run_least_sq_test(a: Array2<f64>, x: Array2<f64>) {
    let b = a.dot(&x);
    let sol = match a.clone().least_squares(&b) {
        Ok(inv) => inv,
        Err(LinalgError::NonInvertible) => return,
        Err(err) => panic!("Unexpected error: {}", err),
    };
    assert_abs_diff_eq!(a.dot(&sol), b, epsilon = 1e-7);
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

    #[test]
    fn least_squares_qr_test((a, x) in common::system_of_arr(common::rect_arr())) {
        run_least_sq_test(a, x);
    }
}

#[test]
fn inverse_scaled_identity() {
    // A perfectly invertible matrix with
    // very small coefficients
    let a = array!(
        [1.0e-20, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0e-20, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0e-20, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0e-20, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0e-20],
    );
    let expected_inverse = array!(
        [1.0e+20, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0e+20, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0e+20, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0e+20, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0e+20],
    );
    let a_inv = a.qr_into().unwrap().inverse().unwrap();
    assert_abs_diff_eq!(a_inv, expected_inverse, epsilon = 1e-3);
}
