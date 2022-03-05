#![allow(unused)]

use std::ops::RangeInclusive;

use ndarray::prelude::*;
use proptest::prelude::*;
use proptest_derive::Arbitrary;

pub const FLOAT_RANGE: RangeInclusive<f64> = -1000.0..=1000.0;
pub const DIM_RANGE: RangeInclusive<usize> = 1..=10;

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
    pub fn matrix(rows: usize, cols: usize)
        (data in prop::collection::vec(FLOAT_RANGE, rows*cols), mut layout in any::<Layout>()) -> Array2<f64> {
        if rows != cols {
            layout.transpose = false; // Transpose reverse rows and cols, which we don't want on non-square matrices
        }
        layout.apply(Array2::from_shape_vec((rows, cols), data).unwrap())
    }
}

prop_compose! {
    pub fn square_arr()(dim in DIM_RANGE)
        (arr in matrix(dim, dim)) -> Array2<f64> {
        arr
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
