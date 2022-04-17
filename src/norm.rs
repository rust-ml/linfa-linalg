//! Norm of vectors

use ndarray::{prelude::*, Data};

/// Define norm as a metric linear space, treating the whole matrix as one big vector.
pub trait Norm {
    type Output;

    /// L-1 norm
    fn norm_l1(&self) -> Self::Output;
    /// L-2 norm
    fn norm_l2(&self) -> Self::Output;
    /// Maximum norm (L-infinite)
    fn norm_max(&self) -> Self::Output;
}

impl<A, S, D> Norm for ArrayBase<S, D>
where
    A: NdFloat + std::iter::Sum,
    S: Data<Elem = A>,
    D: Dimension,
{
    type Output = A;

    fn norm_l1(&self) -> Self::Output {
        self.iter().map(|x| x.abs()).sum()
    }

    fn norm_l2(&self) -> Self::Output {
        self.iter().map(|&x| x * x).sum::<A>().sqrt()
    }

    fn norm_max(&self) -> Self::Output {
        self.iter().fold(A::zero(), |f, &val| val.abs().max(f))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn norms() {
        let a = array![[1.0f64, -3.], [2., -8.]];
        assert_abs_diff_eq!(a.norm_l1(), 14.);
        assert_abs_diff_eq!(a.norm_l2(), 78.0f64.sqrt());
        assert_abs_diff_eq!(a.norm_max(), 8.);
    }
}
