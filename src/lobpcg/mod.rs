//! Decomposition with LOBPCG
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
use ndarray::DataOwned;
use rand::distributions::Standard;
use rand::prelude::*;

pub use algorithm::{lobpcg, LobpcgResult, Order as TruncatedOrder};
pub use eig::{TruncatedEig, TruncatedEigIterator};
pub use svd::{MagnitudeCorrection, TruncatedSvd};

/// Generate random array
pub fn random<A, S, Sh, D>(sh: Sh) -> ArrayBase<S, D>
where
    A: NdFloat,
    S: DataOwned<Elem = A>,
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
    Standard: Distribution<A>,
{
    let mut rng = thread_rng();
    ArrayBase::from_shape_fn(sh, |_| rng.gen::<A>())
}
