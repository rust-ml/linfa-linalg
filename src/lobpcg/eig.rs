//! Truncated eigenvalue decomposition
//!
use super::random;
use crate::lobpcg::{lobpcg, LobpcgResult, TruncatedOrder as Order};

use ndarray::prelude::*;
use ndarray::{stack, NdFloat};
use num_traits::{Float, NumCast};
use std::iter::Sum;

use rand_xoshiro::Xoshiro256Plus;
use rand::{SeedableRng, Rng};

/// Truncated eigenproblem solver
///
/// This struct wraps the LOBPCG algorithm and provides convenient builder-pattern access to
/// parameter like maximal iteration, precision and constraint matrix. Furthermore it allows
/// conversion into a iterative solver where each iteration step yields a new eigenvalue/vector
/// pair.
///
/// # Example
///
/// ```rust
/// use ndarray::{arr1, Array2};
/// use ndarray_linalg_rs::lobpcg::{TruncatedEig, TruncatedOrder};
///
/// let diag = arr1(&[1., 2., 3., 4., 5.]);
/// let a = Array2::from_diag(&diag);
///
/// let eig = TruncatedEig::new(a, TruncatedOrder::Largest)
///    .precision(1e-5)
///    .maxiter(500);
///
/// let res = eig.decompose(3);
/// ```
pub struct TruncatedEig<A: NdFloat, R: Rng> {
    order: Order,
    problem: Array2<A>,
    pub constraints: Option<Array2<A>>,
    preconditioner: Option<Array2<A>>,
    precision: f32,
    maxiter: usize,
    rng: R
}

impl<A: Float + NdFloat + PartialOrd + Default + Sum> TruncatedEig<A, Xoshiro256Plus> {
    /// Create a new truncated eigenproblem solver
    ///
    /// # Properties
    /// * `problem`: problem matrix
    /// * `order`: ordering of the eigenvalues with [TruncatedOrder](crate::TruncatedOrder)
    pub fn new(problem: Array2<A>, order: Order) -> TruncatedEig<A, Xoshiro256Plus> {
        TruncatedEig {
            precision: 1e-5,
            maxiter: problem.len_of(Axis(0)) * 2,
            preconditioner: None,
            constraints: None,
            order,
            problem,
            rng: Xoshiro256Plus::seed_from_u64(42),
        }
    }
}

impl<A: Float + NdFloat + PartialOrd + Default + Sum, R: Rng> TruncatedEig<A, R> {
    /// Set desired precision
    ///
    /// This argument specifies the desired precision, which is passed to the LOBPCG solver. It
    /// controls at which point the opimization of each eigenvalue is stopped. The precision is
    /// global and applied to all eigenvalues with respect to their L2 norm.
    ///
    /// If the precision can't be reached and the maximum number of iteration is reached, then an
    /// error is returned in [LobpcgResult](crate::lobpcg::LobpcgResult).
    pub fn precision(mut self, precision: f32) -> Self {
        self.precision = precision;

        self
    }

    /// Set the maximal number of iterations
    ///
    /// The LOBPCG is an iterative approach to eigenproblems and stops when this maximum
    /// number of iterations are reached.
    pub fn maxiter(mut self, maxiter: usize) -> Self {
        self.maxiter = maxiter;

        self
    }

    /// Construct a solution, which is orthogonal to this
    ///
    /// If a number of eigenvectors are already known, then this function can be used to construct
    /// a orthogonal subspace. Also used with an iterative approach.
    pub fn orthogonal_to(mut self, constraints: Array2<A>) -> Self {
        self.constraints = Some(constraints);

        self
    }

    /// Apply a preconditioner
    ///
    /// A preconditioning matrix can speed up the solving process by improving the spectral
    /// distribution of the eigenvalues. It requires prior knowledge of the problem.
    pub fn precondition_with(mut self, preconditioner: Array2<A>) -> Self {
        self.preconditioner = Some(preconditioner);

        self
    }

    /// Calculate the eigenvalue decomposition
    ///
    /// # Parameters
    ///
    ///  * `num`: number of eigenvalues ordered by magnitude
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::{arr1, Array2};
    /// use ndarray_linalg_rs::lobpcg::{TruncatedEig, TruncatedOrder};
    ///
    /// let diag = arr1(&[1., 2., 3., 4., 5.]);
    /// let a = Array2::from_diag(&diag);
    ///
    /// let eig = TruncatedEig::new(a, TruncatedOrder::Largest)
    ///    .precision(1e-5)
    ///    .maxiter(500);
    ///
    /// let res = eig.decompose(3);
    /// ```
    pub fn decompose(&mut self, num: usize) -> LobpcgResult<A> {
        let x: Array2<f64> = random((self.problem.len_of(Axis(0)), num), &mut self.rng);
        let x = x.mapv(|x| NumCast::from(x).unwrap());

        if let Some(ref preconditioner) = self.preconditioner {
            lobpcg(
                |y| self.problem.dot(&y),
                x,
                |mut y| y.assign(&preconditioner.dot(&y)),
                self.constraints.clone(),
                self.precision,
                self.maxiter,
                self.order.clone(),
            )
        } else {
            lobpcg(
                |y| self.problem.dot(&y),
                x,
                |_| {},
                self.constraints.clone(),
                self.precision,
                self.maxiter,
                self.order.clone(),
            )
        }
    }
}

impl<A: Float + NdFloat + PartialOrd + Default + Sum, R: Rng> IntoIterator for TruncatedEig<A, R> {
    type Item = (Array1<A>, Array2<A>);
    type IntoIter = TruncatedEigIterator<A, R>;

    fn into_iter(self) -> TruncatedEigIterator<A, R> {
        TruncatedEigIterator {
            step_size: 1,
            remaining: self.problem.len_of(Axis(0)),
            eig: self,
        }
    }
}

/// Truncated eigenproblem iterator
///
/// This wraps a truncated eigenproblem and provides an iterator where each step yields a new
/// eigenvalue/vector pair. Useful for generating pairs until a certain condition is met.
///
/// # Example
///
/// ```rust
/// use ndarray::{arr1, Array2};
/// use ndarray_linalg_rs::lobpcg::{TruncatedEig, TruncatedOrder};
///
/// let diag = arr1(&[1., 2., 3., 4., 5.]);
/// let a = Array2::from_diag(&diag);
///
/// let teig = TruncatedEig::new(a, TruncatedOrder::Largest)
///     .precision(1e-5)
///     .maxiter(500);
///
/// // solve eigenproblem until eigenvalues get smaller than 0.5
/// let res = teig.into_iter()
///     .take_while(|x| x.0[0] > 0.5)
///     .flat_map(|x| x.0.to_vec())
///     .collect::<Vec<_>>();
/// ```
pub struct TruncatedEigIterator<A: NdFloat, R: Rng> {
    step_size: usize,
    remaining: usize,
    eig: TruncatedEig<A, R>,
}

impl<A: Float + NdFloat + PartialOrd + Default + Sum, R: Rng> Iterator for TruncatedEigIterator<A, R> {
    type Item = (Array1<A>, Array2<A>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let step_size = usize::min(self.step_size, self.remaining);
        let res = self.eig.decompose(step_size);

        match res {
            LobpcgResult::Ok(vals, vecs, norms) | LobpcgResult::Err(vals, vecs, norms, _) => {
                // abort if any eigenproblem did not converge
                for r_norm in norms {
                    if r_norm > NumCast::from(0.1).unwrap() {
                        return None;
                    }
                }

                // add the new eigenvector to the internal constrain matrix
                let new_constraints = if let Some(ref constraints) = self.eig.constraints {
                    let eigvecs_arr: Vec<_> = constraints
                        .columns()
                        .into_iter()
                        .chain(vecs.columns().into_iter())
                        .collect();

                    stack(Axis(1), &eigvecs_arr).unwrap()
                } else {
                    vecs.clone()
                };

                self.eig.constraints = Some(new_constraints);
                self.remaining -= step_size;

                Some((vals, vecs))
            }
            LobpcgResult::NoResult(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Order;
    use super::TruncatedEig;
    use ndarray::{arr1, Array2};

    #[test]
    fn test_truncated_eig() {
        let diag = arr1(&[
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20.,
        ]);
        let a = Array2::from_diag(&diag);

        let teig = TruncatedEig::new(a, Order::Largest)
            .precision(1e-5)
            .maxiter(500);

        let res = teig.into_iter().take(3).flat_map(|x| x.0.to_vec());
        let ground_truth = vec![20., 19., 18.];

        assert!(
            ground_truth
                .into_iter()
                .zip(res)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                < 0.01
        );
    }
}