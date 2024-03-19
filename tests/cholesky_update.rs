use approx::assert_abs_diff_eq;
use ndarray::prelude::*;
use proptest::prelude::*;
use linfa_linalg::{cholesky::*, cholesky_update::*};
mod common;

prop_compose! {
    fn gram_arr()
        (arr in common::square_arr()) -> (Array2<f64>,Array1<f64>){
        let dim = arr.nrows();
        let mut mul = arr.t().dot(&arr);
        for i in 0..dim {
            mul[(i, i)] += 1.0;
        }

        (mul,arr.slice(s![0,..]).to_owned())
    }
}

fn run_cholesky_update_test(orig: (Array2<f64>, Array1<f64>)) {
    let (arr, x) = orig;
    let mut l_tri = arr.cholesky().unwrap();
    l_tri.cholesky_update_inplace(&x);

    let vt=x.clone().into_shape((1,x.shape()[0])).unwrap();
    let v=x.clone().into_shape((x.shape()[0],1)).unwrap();

    let restore = l_tri.dot(&l_tri.t());
    let expected = arr + v.dot(&vt);
    assert_abs_diff_eq!(restore, expected, epsilon = 1e-7);
}

