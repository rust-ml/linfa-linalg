use crate::{check_square, Result};

use ndarray::{ArrayBase, DataMut, Ix2};
use num_traits::Zero;

pub trait IntoTriangular: Sized {
    fn into_upper_triangular(self) -> Result<Self>;
    fn into_lower_triangular(self) -> Result<Self>;
}

// TODO change to uget_mut
impl<'a, A, S> IntoTriangular for &'a mut ArrayBase<S, Ix2>
where
    A: Zero,
    S: DataMut<Elem = A>,
{
    fn into_upper_triangular(self) -> Result<Self> {
        let n = check_square(self)?;
        for i in 0..n {
            for j in 0..i {
                *self.get_mut((i, j)).unwrap() = A::zero();
            }
        }
        Ok(self)
    }

    fn into_lower_triangular(self) -> Result<Self> {
        let n = check_square(self)?;
        for i in 0..n {
            for j in i + 1..n {
                *self.get_mut((i, j)).unwrap() = A::zero();
            }
        }
        Ok(self)
    }
}

impl<A, S> IntoTriangular for ArrayBase<S, Ix2>
where
    A: Zero,
    S: DataMut<Elem = A>,
{
    fn into_upper_triangular(mut self) -> Result<Self> {
        (&mut self).into_upper_triangular()?;
        Ok(self)
    }

    fn into_lower_triangular(mut self) -> Result<Self> {
        (&mut self).into_lower_triangular()?;
        Ok(self)
    }
}
