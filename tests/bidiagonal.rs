use approx::assert_abs_diff_eq;
use ndarray::prelude::*;
use proptest::prelude::*;

use linfa_linalg::bidiagonal::*;

mod common;

fn run_bidiagonal_test(arr: Array2<f64>) {
    let (nrows, ncols) = arr.dim();
    let decomp = arr.clone().bidiagonal().unwrap();
    let u = decomp.generate_u();
    let vt = decomp.generate_vt();
    let upper_diag = decomp.is_upper_diag();
    let b = decomp.clone().into_b();
    let (diag, offdiag) = decomp.into_diagonals();

    assert_eq!(b.nrows(), b.ncols());
    // U and Vt should be semi-orthogonal
    if nrows > ncols {
        assert_abs_diff_eq!(u.t().dot(&u), Array2::eye(b.nrows()), epsilon = 1e-7);
    } else {
        assert_abs_diff_eq!(u.dot(&u.t()), Array2::eye(b.nrows()), epsilon = 1e-7);
    }
    assert_abs_diff_eq!(vt.dot(&vt.t()), Array2::eye(b.nrows()), epsilon = 1e-7);

    // U * B * Vt should equal original array
    assert_abs_diff_eq!(u.dot(&b).dot(&vt), arr, epsilon = 1e-5);

    // Diagonal and off-diagonal should correspond to B
    assert_abs_diff_eq!(diag, b.diag());
    let partial = if upper_diag {
        b.slice(s![0.., 1..])
    } else {
        b.slice(s![1.., 0..])
    };
    assert_abs_diff_eq!(offdiag, partial.diag());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    #[test]
    fn bidiagonal_test(arr in common::rect_arr()) {
        run_bidiagonal_test(arr);
    }
}
