use approx::assert_abs_diff_eq;
use ndarray::{array, Array2};
use proptest::prelude::*;

use ndarray_linalg_rs::triangular::*;

mod common;

fn tri_system(uplo: UPLO) -> impl Strategy<Value = (Array2<f64>, Array2<f64>)> {
    let squares = common::square_arr().prop_map(move |a| {
        let mut a = a.into_triangular(uplo).unwrap();
        for e in a.diag_mut() {
            if e.abs() < 1.0 {
                *e = 1.0;
            }
        }
        a
    });
    common::system_of_arr(squares)
}

fn run_solve_triangular_test(a: Array2<f64>, x: Array2<f64>, uplo: UPLO) {
    let mut b = a.dot(&x);
    let out = a.solve_triangular(&b, uplo).unwrap();
    assert_abs_diff_eq!(out, x, epsilon = 1e-4);

    let out = a.solve_triangular_into(b.clone(), uplo).unwrap();
    assert_abs_diff_eq!(out, x, epsilon = 1e-4);

    let out = a.solve_triangular_inplace(&mut b, uplo).unwrap();
    assert_abs_diff_eq!(*out, x, epsilon = 1e-4);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    #[test]
    fn solve_triangular_lower(system in tri_system(UPLO::Lower)) {
        let (a, x) = system;
        run_solve_triangular_test(a, x, UPLO::Lower);
    }

    #[test]
    fn solve_triangular_upper(system in tri_system(UPLO::Upper)) {
        let (a, x) = system;
        run_solve_triangular_test(a, x, UPLO::Upper);
    }
}

#[test]
fn known_failure() {
    let a = array![
        [
            3.3562218754086643,
            816.8378593548371,
            -470.72612882136764,
            336.4740568255552,
            654.2571917815051,
            795.9197872262403,
            687.6149593664059,
            -997.6505563244662,
            681.854510815619
        ],
        [
            0.0,
            131.15884945733683,
            -896.9056656026227,
            -73.40632816520974,
            611.318676608028,
            -790.7729067903583,
            995.0019153426838,
            444.63027937639754,
            -396.09160479446655
        ],
        [
            0.0,
            0.0,
            -565.1685049775538,
            588.4959814213651,
            511.3727699624353,
            595.7728287283007,
            924.8460315485909,
            170.01500035862023,
            661.2113982885169
        ],
        [
            0.0,
            0.0,
            0.0,
            -593.2837512804098,
            -887.4556343125483,
            -242.79784588272446,
            968.5909681725007,
            721.4017188483001,
            493.7638484101958
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            43.15642142769889,
            -731.963766509625,
            5.590841737202595,
            209.75382244557431,
            -894.7267077467912
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -961.7702985603449,
            -127.20656830334167,
            -623.3087923506572
        ],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ];

    let x = array![
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            826.3385166487137,
            0.0,
            0.0,
            0.0
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -942.7258981612537,
            0.0,
            0.0,
            0.0
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            548.7103157449269,
            0.0,
            0.0,
            0.0
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -352.50397129849307,
            0.0,
            0.0,
            0.0
        ]
    ];

    run_solve_triangular_test(a, x, UPLO::Upper);
}
