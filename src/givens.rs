use ndarray::{ArrayBase, DataMut, Ix2, NdFloat};

use crate::{index::*, LinalgError, Result};

/// A Givens Rotation
#[derive(Debug, Clone)]
pub struct GivensRotation<A> {
    c: A,
    s: A,
}

impl<A: NdFloat> GivensRotation<A> {
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

    pub fn try_new(c: A, s: A, eps: A) -> Option<(Self, A)> {
        let norm = (c * c + s * s).sqrt();
        if norm > eps {
            let c = c / norm;
            let s = s / norm;
            Some((Self { c, s }, norm))
        } else {
            None
        }
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
            unsafe {
                let a = *lhs.at((j, 0));
                let b = *lhs.at((j, 1));
                *lhs.atm((j, 0)) = a * c + s * b;
                *lhs.atm((j, 1)) = -s * a + b * c;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn cancel_y() {
        let (rot, r) = GivensRotation::cancel_y(1.0f64, 2.0).unwrap();
        assert_abs_diff_eq!(r, 5.0_f64.sqrt());
        assert_abs_diff_eq!(rot.c, 0.4472136, epsilon = 1e-5);
        assert_abs_diff_eq!(rot.s, -0.8944272, epsilon = 1e-5);
        assert_abs_diff_eq!(
            array![[rot.c, -rot.s], [rot.s, rot.c]].dot(&array![1., 2.]),
            array![r, 0.]
        );

        assert!(GivensRotation::cancel_y(3.0f64, 0.).is_none());
    }

    #[test]
    fn rotate_rows() {
        let (rot, _) = GivensRotation::cancel_y(1.0f64, 2.0).unwrap();
        let rows = array![[2., 3.], [4., 5.], [1., 2.], [3., 4.]];
        let mut out = rows.clone();
        rot.rotate_rows(&mut out).unwrap();
        assert_abs_diff_eq!(
            rows.dot(&array![[rot.c, -rot.s], [rot.s, rot.c]]),
            out,
            epsilon = 1e-5
        );

        assert!(matches!(
            rot.rotate_rows(&mut array![[1., 2., 3.]]),
            Err(LinalgError::WrongColumns {
                expected: 2,
                actual: 3
            })
        ));
    }
}
