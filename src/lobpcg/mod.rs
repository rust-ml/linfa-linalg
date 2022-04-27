//!
//! Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG) is a matrix-free method for
//! finding the large (or smallest) eigenvalues and the corresponding eigenvectors of a symmetric
//! eigenvalue problem
//! ```text
//! A x = lambda x
//! ```
//! where A is symmetric and (x, lambda) the solution. It has the following advantages:
//! * matrix free: does not require storing the coefficient matrix explicitely and only evaluates
//! matrix-vector products.
//! * factorization-free: does not require any matrix decomposition
//! * linear-convergence: theoretically guaranteed and practically observed
//!
//! See also the wikipedia article at [LOBPCG](https://en.wikipedia.org/wiki/LOBPCG)
//!
mod algorithm;
mod eig;
mod svd;

use ndarray::prelude::*;
use ndarray::OwnedRepr;
use rand::distributions::Standard;
use rand::prelude::*;

pub use crate::{LinalgError, Order};
pub use algorithm::lobpcg;
pub use eig::{TruncatedEig, TruncatedEigIterator};
pub use svd::{MagnitudeCorrection, TruncatedSvd};

/// Generate random array
pub(crate) fn random<A, Sh, D, R: Rng>(sh: Sh, mut rng: R) -> ArrayBase<OwnedRepr<A>, D>
where
    A: NdFloat,
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
    Standard: Distribution<A>,
{
    ArrayBase::from_shape_fn(sh, |_| rng.gen::<A>())
}

/// The result of the eigensolver
///
/// In the best case the eigensolver has converged with a result better than the given threshold,
/// then a `LobpcgResult::Ok` gives the eigenvalues, eigenvectors and norms. If an error ocurred
/// during the process, it is returned in `LobpcgResult::Err`, but the best result is still returned,
/// as it could be usable. If there is no result at all, then `LobpcgResult::NoResult` is returned.
/// This happens if the algorithm fails in an early stage, for example if the matrix `A` is not SPD
pub type LobpcgResult<A> = std::result::Result<Lobpcg<A>, (LinalgError, Option<Lobpcg<A>>)>;
pub struct Lobpcg<A> {
    evals: Array1<A>,
    evecs: Array2<A>,
    rnorm: Vec<A>,
}
