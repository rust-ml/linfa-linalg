use std::ops::RangeInclusive;

use ndarray::prelude::*;
use proptest::prelude::*;

#[allow(unused)]
const FLOAT_RANGE: RangeInclusive<f64> = -1000.0..=1000.0;
#[allow(unused)]
const DIM_RANGE: RangeInclusive<usize> = 1..=10;

prop_compose! {
    pub fn square_arr()(dim in DIM_RANGE)
        (data in prop::collection::vec(FLOAT_RANGE, dim*dim), dim in Just(dim)) -> Array2<f64> {
        Array2::from_shape_vec((dim, dim), data).unwrap()
    }
}
