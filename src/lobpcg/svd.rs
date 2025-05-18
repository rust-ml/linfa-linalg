//! Truncated singular value decomposition
//!
//! This module computes the k largest/smallest singular values/vectors for a dense matrix.
use crate::{
    eigh::{EigSort, Eigh},
    lobpcg::{lobpcg, random, Lobpcg},
    Order, Result,
};
use ndarray::prelude::*;
use num_traits::NumCast;
use std::iter::Sum;

use rand::Rng;

/// The result of a eigenvalue decomposition, not yet transformed into singular values/vectors
///
/// Provides methods for either calculating just the singular values with reduced cost or the
/// vectors with additional cost of matrix multiplication.
#[derive(Debug, Clone)]
pub struct TruncatedSvdResult<A> {
    eigvals: Array1<A>,
    eigvecs: Array2<A>,
    problem: Array2<A>,
    order: Order,
    ngm: bool,
}

impl<A: NdFloat + 'static + MagnitudeCorrection> TruncatedSvdResult<A> {
    /// Returns singular values ordered by magnitude with indices.
    fn singular_values_with_indices(&self) -> (Array1<A>, Vec<usize>) {
        // numerate eigenvalues
        let mut a = self.eigvals.iter().enumerate().collect::<Vec<_>>();

        let (values, indices) = if self.order == Order::Largest {
            // sort by magnitude
            a.sort_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap().reverse());

            // calculate cut-off magnitude (borrowed from scipy)
            let cutoff = A::epsilon() * // float precision
                         A::correction() * // correction term (see trait below)
                         *a[0].1; // max eigenvalue

            // filter low singular values away
            let (values, indices): (Vec<A>, Vec<usize>) = a
                .into_iter()
                .filter(|(_, x)| *x > &cutoff)
                .map(|(a, b)| (b.sqrt(), a))
                .unzip();

            (values, indices)
        } else {
            a.sort_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap());

            let (values, indices) = a.into_iter().map(|(a, b)| (b.sqrt(), a)).unzip();

            (values, indices)
        };

        (Array1::from(values), indices)
    }

    /// Returns singular values ordered by magnitude
    pub fn values(&self) -> Array1<A> {
        let (values, _) = self.singular_values_with_indices();

        values
    }

    /// Returns singular values, left-singular vectors and right-singular vectors
    pub fn values_vectors(&self) -> (Array2<A>, Array1<A>, Array2<A>) {
        let (values, indices) = self.singular_values_with_indices();

        // branch n > m (for A is [n x m])
        #[allow(clippy::branches_sharing_code)]
        let (u, v) = if self.ngm {
            let vlarge = self.eigvecs.select(Axis(1), &indices);
            let mut ularge = self.problem.dot(&vlarge);

            ularge
                .columns_mut()
                .into_iter()
                .zip(values.iter())
                .for_each(|(mut a, b)| a.mapv_inplace(|x| x / *b));

            (ularge, vlarge)
        } else {
            let ularge = self.eigvecs.select(Axis(1), &indices);

            let mut vlarge = self.problem.t().dot(&ularge);
            vlarge
                .columns_mut()
                .into_iter()
                .zip(values.iter())
                .for_each(|(mut a, b)| a.mapv_inplace(|x| x / *b));

            (ularge, vlarge)
        };

        (u, values, v.reversed_axes())
    }
}

#[derive(Debug, Clone)]
/// Truncated singular value decomposition
///
/// Wraps the LOBPCG algorithm and provides convenient builder-pattern access to
/// parameter like maximal iteration, precision and constrain matrix.
pub struct TruncatedSvd<A: NdFloat, R: Rng> {
    order: Order,
    problem: Array2<A>,
    precision: f32,
    maxiter: usize,
    rng: R,
}

impl<A: NdFloat + Sum, R: Rng> TruncatedSvd<A, R> {
    /// Create a new truncated SVD problem
    ///
    /// # Parameters
    ///  * `problem`: rectangular matrix which is decomposed
    ///  * `order`: whether to return large or small (close to zero) singular values
    ///  * `rng`: random number generator
    pub fn new_with_rng(problem: Array2<A>, order: Order, rng: R) -> TruncatedSvd<A, R> {
        TruncatedSvd {
            precision: 1e-5,
            maxiter: problem.len_of(Axis(0)) * 2,
            order,
            problem,
            rng,
        }
    }
}

impl<A: NdFloat + Sum, R: Rng> TruncatedSvd<A, R> {
    /// Set the required precision of the solution
    ///
    /// The precision is, in the context of SVD, the square-root precision of the underlying
    /// eigenproblem solution. The eigenproblem-precision is used to check the L2 error of each
    /// eigenvector and stops its optimization when the required precision is reached.
    pub fn precision(mut self, precision: f32) -> Self {
        self.precision = precision;

        self
    }

    /// Set the maximal number of iterations
    ///
    /// The LOBPCG is an iterative approach to eigenproblems and stops when this maximum
    /// number of iterations are reached
    pub fn maxiter(mut self, maxiter: usize) -> Self {
        self.maxiter = maxiter;

        self
    }

    /// Calculate the singular value decomposition
    ///
    /// # Parameters
    ///
    ///  * `num`: number of singular-value/vector pairs, ordered by magnitude
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::{arr1, Array2};
    /// use linfa_linalg::{Order, lobpcg::TruncatedSvd};
    /// use rand::SeedableRng;
    /// use rand_xoshiro::Xoshiro256Plus;
    ///
    /// let diag = arr1(&[1., 2., 3., 4., 5.]);
    /// let a = Array2::from_diag(&diag);
    ///
    /// let eig = TruncatedSvd::new_with_rng(a, Order::Largest, Xoshiro256Plus::seed_from_u64(42))
    ///    .precision(1e-4)
    ///    .maxiter(500);
    ///
    /// let res = eig.decompose(3);
    /// ```
    pub fn decompose(mut self, num: usize) -> Result<TruncatedSvdResult<A>> {
        if num == 0 {
            // return empty solution if requested eigenvalue number is zero
            return Ok(TruncatedSvdResult {
                eigvals: Array1::zeros(0),
                eigvecs: Array2::zeros((0, 0)),
                problem: Array2::zeros((0, 0)),
                order: self.order,
                ngm: false,
            });
        }

        let (n, m) = (self.problem.nrows(), self.problem.ncols());
        let ngm = n > m;

        // use dense eigenproblem solver if more than 1/5 eigenvalues requested
        if num * 5 > n.min(m) {
            let problem = if ngm {
                self.problem.t().dot(&self.problem)
            } else {
                self.problem.dot(&self.problem.t())
            };

            let (eigvals, eigvecs) = problem.eigh()?.sort_eig(self.order);

            let (eigvals, eigvecs) = (
                eigvals.slice_move(s![..num]),
                eigvecs.slice_move(s![..num, ..]),
            );

            return Ok(TruncatedSvdResult {
                eigvals,
                eigvecs,
                problem: self.problem,
                order: self.order,
                ngm,
            });
        }

        // generate initial matrix
        let x: Array2<f32> = random((usize::min(n, m), num), &mut self.rng);
        let x = x.mapv(|x| NumCast::from(x).unwrap());

        // square precision because the SVD squares the eigenvalue as well
        let precision = self.precision * self.precision;

        // use problem definition with less operations required
        let res = if n > m {
            lobpcg(
                |y| self.problem.t().dot(&self.problem.dot(&y)),
                x,
                |_| {},
                None,
                precision,
                self.maxiter,
                self.order,
            )
        } else {
            lobpcg(
                |y| self.problem.dot(&self.problem.t().dot(&y)),
                x,
                |_| {},
                None,
                precision,
                self.maxiter,
                self.order,
            )
        };

        // convert into TruncatedSvdResult
        match res {
            Ok(Lobpcg {
                eigvals, eigvecs, ..
            })
            | Err((
                _,
                Some(Lobpcg {
                    eigvals, eigvecs, ..
                }),
            )) => Ok(TruncatedSvdResult {
                problem: self.problem,
                eigvals,
                eigvecs,
                order: self.order,
                ngm,
            }),
            Err((err, None)) => Err(err),
        }
    }
}

/// Magnitude Correction
///
/// The magnitude correction changes the cut-off point at which an eigenvector belongs to the
/// null-space and its eigenvalue is therefore zero. The correction is multiplied by the floating
/// point epsilon and therefore dependent on the floating type.
pub trait MagnitudeCorrection {
    fn correction() -> Self;
}

impl MagnitudeCorrection for f32 {
    fn correction() -> Self {
        1.0e3
    }
}

impl MagnitudeCorrection for f64 {
    fn correction() -> Self {
        1.0e6
    }
}

#[cfg(test)]
mod tests {
    use super::Order;
    use super::TruncatedSvd;

    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2, Array2, NdFloat};
    use rand::distributions::{Distribution, Standard};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    /// Generate random array
    fn random<A>(sh: (usize, usize)) -> Array2<A>
    where
        A: NdFloat,
        Standard: Distribution<A>,
    {
        let rng = Xoshiro256Plus::seed_from_u64(3);
        super::random(sh, rng)
    }

    #[test]
    fn test_truncated_svd() {
        let a = arr2(&[[3., 2., 2.], [2., 3., -2.]]);

        let res = TruncatedSvd::new_with_rng(a, Order::Largest, Xoshiro256Plus::seed_from_u64(42))
            .precision(1e-5)
            .maxiter(10)
            .decompose(2)
            .unwrap();

        let (_, sigma, _) = res.values_vectors();

        assert_abs_diff_eq!(&sigma, &arr1(&[5.0, 3.0]), epsilon = 1e-5);
    }

    #[test]
    fn test_truncated_svd_random() {
        let a: Array2<f64> = random((50, 10));

        let res = TruncatedSvd::new_with_rng(
            a.clone(),
            Order::Largest,
            Xoshiro256Plus::seed_from_u64(42),
        )
        .precision(1e-5)
        .maxiter(10)
        .decompose(10)
        .unwrap();

        let (u, sigma, v_t) = res.values_vectors();
        let reconstructed = u.dot(&Array2::from_diag(&sigma).dot(&v_t));

        assert_abs_diff_eq!(&a, &reconstructed, epsilon = 1e-5);
    }
}
