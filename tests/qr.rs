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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    #[test]
    fn qr_test(arr in common::thin_arr()) {
        run_qr_test(arr)
    }
}
