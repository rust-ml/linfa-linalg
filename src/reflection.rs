use ndarray::{
    linalg::{general_mat_mul, general_mat_vec_mul},
    ArrayBase, Axis, Data, DataMut, Ix1, Ix2, NdFloat,
};

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

    pub fn axis(&self) -> &ArrayBase<D, Ix1> {
        &self.axis
    }
}

impl<A: NdFloat, D: Data<Elem = A>> Reflection<A, D> {
    /// Apply reflection to the columns of `rhs`
    pub fn reflect_cols<M: DataMut<Elem = A>>(&self, rhs: &mut ArrayBase<M, Ix2>) {
        for i in 0..rhs.ncols() {
            let m_two = A::from(-2.0f64).unwrap();
            let factor = (self.axis.dot(&rhs.column(i)) - self.bias) * m_two;
            rhs.column_mut(i).scaled_add(factor, &self.axis);
        }
    }

    /// Apply reflection to the rows of `lhs`
    ///
    /// Assume that length of `work` equals rows of `lhs` and length of `self.axis` equals columns
    /// of `lhs`.
    pub fn reflect_rows<M1: DataMut<Elem = A>, M2: DataMut<Elem = A>>(
        &self,
        lhs: &mut ArrayBase<M1, Ix2>,
        work: &mut ArrayBase<M2, Ix1>,
    ) {
        // work = lhs * axis
        general_mat_vec_mul(A::one(), lhs, &self.axis, A::zero(), work);
        *work -= self.bias;
        let m_two = A::from(-2.0).unwrap();
        // lhs += work * axis
        general_mat_mul(
            m_two,
            &work.view().insert_axis(Axis(1)),
            &self.axis.view().insert_axis(Axis(0)),
            A::one(),
            lhs,
        );
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    use super::*;

    #[test]
    fn reflect_plane_col() {
        let y_axis = array![0., 1., 0.];
        let refl = Reflection::new(y_axis.view(), 0.0);

        let mut v = array![[1., 2., 3.], [3., 4., 5.]].reversed_axes();
        refl.reflect_cols(&mut v);
        assert_abs_diff_eq!(v, array![[1., -2., 3.], [3., -4., 5.]].reversed_axes());
        refl.reflect_cols(&mut v);
        assert_abs_diff_eq!(v, array![[1., 2., 3.], [3., 4., 5.]].reversed_axes());

        let refl = Reflection::new(y_axis.view(), 3.0);
        let mut v = array![[1., 2., 3.], [3., 4., 5.]].reversed_axes();
        refl.reflect_cols(&mut v);
        assert_abs_diff_eq!(v, array![[1., 4., 3.], [3., 2., 5.]].reversed_axes());
    }

    #[test]
    fn reflect_plane_row() {
        let y_axis = array![0., 1., 0.];
        let refl = Reflection::new(y_axis.view(), 0.0);

        let mut v = array![[1., 2., 3.], [3., 4., 5.]];
        let mut work = Array1::zeros(2);
        refl.reflect_rows(&mut v, &mut work);
        assert_abs_diff_eq!(v, array![[1., -2., 3.], [3., -4., 5.]]);
        refl.reflect_rows(&mut v, &mut work);
        assert_abs_diff_eq!(v, array![[1., 2., 3.], [3., 4., 5.]]);

        let refl = Reflection::new(y_axis.view(), 3.0);
        let mut v = array![[1., 2., 3.], [3., 4., 5.]];
        refl.reflect_rows(&mut v, &mut work);
        assert_abs_diff_eq!(v, array![[1., 4., 3.], [3., 2., 5.]]);
    }
}
