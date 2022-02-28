use approx::assert_abs_diff_eq;
use ndarray::Array2;
use num_traits::Zero;
use proptest::prelude::*;

use ndarray_linalg_rs::tridiagonal::*;

mod common;

// Assume arr is symmetric
fn run_tridiagonal_test(arr: &Array2<f64>) {
    let n = arr.nrows();
    let decomp = arr.clone().sym_tridiagonal().unwrap();
    let q = decomp.generate_q();
    let tri = decomp.into_tridiag_matrix();
    // Ensure it's actually tridiagonal
    for ((i, j), e) in tri.indexed_iter() {
        if (i as i64 - j as i64).abs() > 1 {
            assert!(e.is_zero());
        }
    }

    // Q * T * Q.t must equal arr
    assert_abs_diff_eq!(q.dot(&tri).dot(&q.t()), arr, epsilon = 1e-7);
    // Q must be orthogonal
    assert_abs_diff_eq!(q.dot(&q.t()), Array2::eye(n), epsilon = 1e-7);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    #[test]
    fn tridiagonal_test(arr in common::symm_arr()) {
        run_tridiagonal_test(&arr);
    }

    #[test]
    // Just make sure it doesn't crash, we know the answer will be wrong
    fn tridiagonal_non_symm(arr in common::square_arr()) {
        let decomp = arr.sym_tridiagonal().unwrap();
        decomp.generate_q();
        decomp.into_tridiag_matrix();
    }
}
