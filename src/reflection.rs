use ndarray::{ArrayBase, Data, DataMut, Ix1, Ix2};

use crate::Float;

/// Reflection with respect to a plane
pub struct Reflection<A, D: Data<Elem = A>> {
    axis: ArrayBase<D, Ix1>,
    bias: A,
}

impl<A, D: Data<Elem = A>> Reflection<A, D> {
    /// Create a new reflection with respect to the plane orthogonal to the given axis and bias
    ///
    /// `axis` must be a unit vector
    /// `bias` is the position of the plane on the axis from the origin
    pub fn new(axis: ArrayBase<D, Ix1>, bias: A) -> Self {
        Self { axis, bias }
    }
}

impl<A: Float, D: Data<Elem = A>> Reflection<A, D> {
    /// Apply reflection to the columns of `rhs`
    pub fn reflect_col<M: DataMut<Elem = A>>(&self, rhs: &mut ArrayBase<M, Ix2>) {
        for i in 0..rhs.ncols() {
            let m_two = A::from(-2.0f64).unwrap();
            let factor = (self.axis.dot(&rhs.column(i)) - self.bias) * m_two;
            rhs.column_mut(i).scaled_add(factor, &self.axis);
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn reflect_plane() {
        let y_axis = array![0., 1., 0.];
        let refl = Reflection::new(y_axis.view(), 0.0);

        let mut v = array![[1., 2., 3.], [3., 4., 5.]].reversed_axes();
        refl.reflect_col(&mut v);
        assert_abs_diff_eq!(v, array![[1., -2., 3.], [3., -4., 5.]].reversed_axes());
        refl.reflect_col(&mut v);
        assert_abs_diff_eq!(v, array![[1., 2., 3.], [3., 4., 5.]].reversed_axes());

        let refl = Reflection::new(y_axis.view(), 3.0);
        let mut v = array![[1., 2., 3.], [3., 4., 5.]].reversed_axes();
        refl.reflect_col(&mut v);
        assert_abs_diff_eq!(v, array![[1., 4., 3.], [3., 2., 5.]].reversed_axes());
    }
}
