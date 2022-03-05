use approx::{abs_diff_eq, assert_abs_diff_eq};
use ndarray::Array2;
use proptest::prelude::*;

use ndarray_linalg_rs::triangular::*;

mod common;

prop_compose! {
    fn tri_system(uplo: UPLO)(a in common::square_arr(), cols in common::DIM_RANGE)
        (x in common::matrix(a.nrows(), cols), a in Just(a)) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let mut a = a.into_triangular(uplo).unwrap();
        for e in a.diag_mut() {
            if abs_diff_eq!(*e, 0.0, epsilon = 1e-7) {
                *e = 1.0;
            }
        }
        let b = a.dot(&x);
        (a, x, b)
    }
}

fn run_solve_triangular_test(a: Array2<f64>, x: Array2<f64>, mut b: Array2<f64>, uplo: UPLO) {
    let out = a.solve_triangular(&b, uplo).unwrap();
    assert_abs_diff_eq!(out, x, epsilon = 1e-5);

    let out = a.solve_triangular_into(b.clone(), uplo).unwrap();
    assert_abs_diff_eq!(out, x, epsilon = 1e-5);

    let out = a.solve_triangular_inplace(&mut b, uplo).unwrap();
    assert_abs_diff_eq!(*out, x, epsilon = 1e-5);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    #[test]
    fn solve_triangular_lower(system in tri_system(UPLO::Lower)) {
        let (a, x, b) = system;
        run_solve_triangular_test(a, x, b, UPLO::Lower);
    }

    #[test]
    fn solve_triangular_upper(system in tri_system(UPLO::Upper)) {
        let (a, x, b) = system;
        run_solve_triangular_test(a, x, b, UPLO::Upper);
    }
}
