#![allow(unused)]

use std::ops::RangeInclusive;

use ndarray::prelude::*;
use proptest::prelude::*;
use proptest_derive::Arbitrary;

const FLOAT_RANGE: RangeInclusive<f64> = -1000.0..=1000.0;
const DIM_RANGE: RangeInclusive<usize> = 1..=10;

#[derive(Debug, Arbitrary)]
struct Layout {
    invert_rows: bool,
    invert_cols: bool,
    transpose: bool,
}

impl Layout {
    fn apply(&self, mut arr: Array2<f64>) -> Array2<f64> {
        if self.invert_rows {
            arr.invert_axis(Axis(0));
        }
        if self.invert_cols {
            arr.invert_axis(Axis(1));
        }
        if self.transpose {
            arr.reversed_axes()
        } else {
            arr
        }
    }
}

prop_compose! {
    pub fn square_arr()(dim in DIM_RANGE)
        (data in prop::collection::vec(FLOAT_RANGE, dim*dim), dim in Just(dim), layout in any::<Layout>()) -> Array2<f64> {
        layout.apply(Array2::from_shape_vec((dim, dim), data).unwrap())
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
