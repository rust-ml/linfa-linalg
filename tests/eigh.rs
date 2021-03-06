use approx::assert_abs_diff_eq;
use ndarray::prelude::*;
use proptest::prelude::*;

use linfa_linalg::eigh::*;

mod common;

fn check_eigh(arr: &Array2<f64>, vals: &Array1<f64>, vecs: &Array2<f64>) {
    // Original array multiplied with eigenvec should equal eigenval times eigenvec
    for (i, v) in vecs.axis_iter(Axis(1)).enumerate() {
        let av = arr.dot(&v);
        let ev = v.mapv(|x| vals[i] * x);
        assert_abs_diff_eq!(av, ev, epsilon = 1e-5);
    }
}

fn run_eigh_test(arr: Array2<f64>) {
    let n = arr.nrows();
    let d = arr.eigh().unwrap();
    let (vals, vecs) = d.clone();
    assert_abs_diff_eq!(arr.eigvalsh().unwrap(), vals, epsilon = 1e-5);
    // Eigenvecs should be orthogonal
    let s = vecs.t().dot(&vecs);
    assert_abs_diff_eq!(s, Array2::eye(n), epsilon = 1e-5);
    check_eigh(&arr, &vals, &vecs);

    let (evals, evecs) = arr.clone().eigh_into().unwrap();
    assert_abs_diff_eq!(evals, vals);
    assert_abs_diff_eq!(evecs, vecs);
    let evals = arr.clone().eigvalsh_into().unwrap();
    assert_abs_diff_eq!(evals, vals);

    // Check if ascending eigen is actually sorted and valid
    let (vals, vecs) = d.clone().sort_eig_asc();
    check_eigh(&arr, &vals, &vecs);
    assert!(vals.windows(2).into_iter().all(|w| w[0] <= w[1]));

    // Check if descending eigen is actually sorted and valid
    let (vals, vecs) = d.sort_eig_desc();
    check_eigh(&arr, &vals, &vecs);
    assert!(vals.windows(2).into_iter().all(|w| w[0] >= w[1]));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    #[test]
    fn eigh_test(arr in common::symm_arr()) {
        run_eigh_test(arr);
    }

    #[test]
    // Make sure this doesn't crash on non-symmetric matrices
    fn eigh_no_symm(arr in common::square_arr()) {
        arr.eigh_into().unwrap();
    }
}

#[test]
fn eigh_f32() {
    let vals = array![[1f32, -5., 7.], [-5., 2., -9.], [7., -9., 3.]]
        .eigvalsh()
        .unwrap();
    assert_abs_diff_eq!(vals, array![16.28378, -3.41558, -6.86819], epsilon = 1e-5);
}
