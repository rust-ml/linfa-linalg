#![allow(unused)]

use std::ops::RangeInclusive;

use ndarray::prelude::*;
use proptest::prelude::*;
use proptest_derive::Arbitrary;

pub const FLOAT_RANGE: RangeInclusive<f64> = -100.0..=100.0;
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

prop_compose! {
    pub fn rect_arr()(rows in DIM_RANGE, cols in DIM_RANGE)
        (arr in matrix(rows, cols)) -> Array2<f64> {
        arr
    }
}

/// Rect array where rows >= cols
pub fn thin_arr() -> impl Strategy<Value = Array2<f64>> {
    DIM_RANGE
        .prop_flat_map(|cols| (cols..=10).prop_map(move |rows| (rows, cols)))
        .prop_flat_map(|(r, c)| matrix(r, c))
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

/// Given a strategy that produces square arrays `a`, output strategy producing the arrays `a`
/// and `x`, where `a * x = b` (`b` needs to be computed inside the test).
pub fn system_of_arr(
    squares: impl Strategy<Value = Array2<f64>>,
) -> impl Strategy<Value = (Array2<f64>, Array2<f64>)> {
    squares.prop_flat_map(|a| {
        let rows = a.nrows();
        (
            Just(a),
            DIM_RANGE.prop_flat_map(move |col| matrix(rows, col)),
        )
    })
}
