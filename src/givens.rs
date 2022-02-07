use ndarray::{ArrayBase, DataMut, Ix2};

use crate::{Float, LinalgError, Result};

/// A Givens Rotation
#[derive(Debug, Clone)]
pub struct GivensRotation<A> {
    c: A,
    s: A,
}

impl<A: Float> GivensRotation<A> {
    /// Computes rotation `R` such that the `y` component of `R * [x, y].t` is 0
    ///
    /// Returns `None` if `y` is 0 (no rotation needed), otherwise return the rotation and the norm
    /// of vector `[x, y]`.
    pub fn cancel_y(x: A, y: A) -> Option<(Self, A)> {
        // Not equivalent to nalgebra impl
        if !y.is_zero() {
            let r = (x * x + y * y).sqrt();
            let c = x / r;
            let s = -y / r;
            Some((Self { c, s }, r))
        } else {
            None
        }
    }

    /// Constructs Givens rotation without checking whether `c` and `s` are valid
    pub fn new_unchecked(c: A, s: A) -> Self {
        Self { c, s }
    }

    pub fn c(&self) -> A {
        self.c
    }
    pub fn s(&self) -> A {
        self.s
    }

    /// The inverse Givens rotation
    pub fn inverse(self) -> Self {
        Self {
            c: self.c,
            s: -self.s,
        }
    }

    /// Performs the multiplication `lhs = lhs * self` in-place.
    pub fn rotate_rows<S: DataMut<Elem = A>>(&self, lhs: &mut ArrayBase<S, Ix2>) -> Result<()> {
        let cols = lhs.ncols();
        if cols != 2 {
            return Err(LinalgError::WrongColumns {
                expected: 2,
                actual: cols,
            });
        }
        let c = self.c;
        let s = self.s;

        for j in 0..lhs.nrows() {
            let a = *lhs.get((j, 0)).unwrap();
            let b = *lhs.get((j, 1)).unwrap();
            *lhs.get_mut((j, 0)).unwrap() = a * c + s * b;
            *lhs.get_mut((j, 1)).unwrap() = -s * a + b * c;
        }

        Ok(())
    }
}
