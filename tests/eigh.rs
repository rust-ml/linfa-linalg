use approx::assert_abs_diff_eq;
use ndarray::prelude::*;
use proptest::prelude::*;

use ndarray_linalg_rs::eigh::*;

mod common;

fn run_eigh_test(arr: Array2<f64>) {
    let n = arr.nrows();
    let (vals, vecs) = arr.eigh().unwrap();
    assert_abs_diff_eq!(arr.eigvalsh().unwrap(), vals, epsilon = 1e-5);
    // Eigenvecs should be orthogonal
    let s = vecs.t().dot(&vecs);
    assert_abs_diff_eq!(s, Array2::eye(n), epsilon = 1e-5);
    // Original array multiplied with eigenvec should equal eigenval times eigenvec
    for (i, v) in vecs.axis_iter(Axis(1)).enumerate() {
        let av = arr.dot(&v);
        let ev = v.mapv(|x| vals[i] * x);
        assert_abs_diff_eq!(av, ev, epsilon = 1e-5);
    }

    let (evals, evecs) = arr.clone().eigh_into().unwrap();
    assert_abs_diff_eq!(evals, vals);
    assert_abs_diff_eq!(evecs, vecs);
    let evals = arr.eigvalsh_into().unwrap();
    assert_abs_diff_eq!(evals, vals);
}

proptest! {
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
