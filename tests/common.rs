#![allow(unused)]

use std::ops::RangeInclusive;

use ndarray::prelude::*;
use proptest::prelude::*;

const FLOAT_RANGE: RangeInclusive<f64> = -1000.0..=1000.0;
const DIM_RANGE: RangeInclusive<usize> = 1..=10;

prop_compose! {
    pub fn square_arr()(dim in DIM_RANGE)
        (data in prop::collection::vec(FLOAT_RANGE, dim*dim), dim in Just(dim)) -> Array2<f64> {
        Array2::from_shape_vec((dim, dim), data).unwrap()
    }
}

// TODO offer this in the main crate API
fn to_symm(arr: &mut Array2<f64>) {
    let n = arr.nrows();
    for i in 0..n {
        for j in 0..i {
            arr[(i, j)] = arr[(j, i)];
        }
    }
}

prop_compose! {
    pub fn symm_arr()(mut arr in square_arr()) -> Array2<f64> {
        to_symm(&mut arr);
        arr
    }
}
