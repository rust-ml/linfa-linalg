use approx::assert_abs_diff_eq;
use ndarray::prelude::*;
use proptest::prelude::*;

use linfa_linalg::{cholesky::*, triangular::*};

mod common;

fn run_cholesky_test(orig: Array2<f64>) {
    let chol = orig.cholesky().unwrap();
    assert_abs_diff_eq!(chol.dot(&chol.t()), orig, epsilon = 1e-7);
    let dirty = orig.cholesky_dirty().unwrap();
    assert!(chol.is_triangular(UPLO::Lower));
    assert_abs_diff_eq!(
        chol,
        dirty.into_triangular(UPLO::Lower).unwrap(),
        epsilon = 1e-7
    );

    let chol = orig.clone().cholesky_into().unwrap();
    assert_abs_diff_eq!(chol.dot(&chol.t()), orig, epsilon = 1e-7);
    let dirty = orig.clone().cholesky_into_dirty().unwrap();
    assert!(chol.is_triangular(UPLO::Lower));
    assert_abs_diff_eq!(
        chol,
        dirty.into_triangular(UPLO::Lower).unwrap(),
        epsilon = 1e-7
    );

    let mut a = orig.clone();
    let chol = a.cholesky_inplace().unwrap();
    assert_abs_diff_eq!(chol.dot(&chol.t()), orig, epsilon = 1e-7);
    assert_abs_diff_eq!(a.dot(&a.t()), orig, epsilon = 1e-7);
    let mut b = orig;
    let dirty = b.cholesky_inplace_dirty().unwrap();
    assert!(a.is_triangular(UPLO::Lower));
    assert_abs_diff_eq!(
        a,
        dirty.triangular_inplace(UPLO::Lower).unwrap(),
        epsilon = 1e-7
    );
}

fn run_solvec_test(mut a: Array2<f64>, x: Array2<f64>) {
    let mut b = a.dot(&x);

    assert_abs_diff_eq!(a.clone().solvec(&b).unwrap(), x, epsilon = 1e-5);
    assert_abs_diff_eq!(a.clone().solvec_into(b.clone()).unwrap(), x, epsilon = 1e-5);
    assert_abs_diff_eq!(*a.solvec_inplace(&mut b).unwrap(), x, epsilon = 1e-5);
}

fn run_invc_test(a: Array2<f64>) {
    let inv = a.invc().unwrap();
    assert_abs_diff_eq!(a.dot(&inv), Array2::eye(a.nrows()), epsilon = 1e-7);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    #[test]
    fn cholesky_test(arr in common::hpd_arr()) {
        run_cholesky_test(arr)
    }

    #[test]
    fn solvec_test((a, x) in common::system_of_arr(common::hpd_arr())) {
        run_solvec_test(a, x)
    }

    #[test]
    fn invc_test(arr in common::hpd_arr()) {
        run_invc_test(arr)
    }
}

#[test]
fn cholesky_f32() {
    let arr = array![[25f32, 15., -5.], [15., 18., 0.], [-5., 0., 11.]];
    let lower = array![[5.0, 0.0, 0.0], [3.0, 3.0, 0.0], [-1., 1., 3.]];

    let chol = arr.cholesky().unwrap();
    assert_abs_diff_eq!(chol, lower, epsilon = 1e-7);
    assert_abs_diff_eq!(chol.dot(&chol.t()), arr, epsilon = 1e-7);
}
