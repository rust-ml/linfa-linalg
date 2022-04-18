mod eig;
mod lobpcg;
mod svd;

use ndarray::prelude::*;
use ndarray::DataOwned;
use rand::distributions::Standard;
use rand::prelude::*;

pub use eig::TruncatedEig;
pub use lobpcg::{lobpcg, LobpcgResult, Order as TruncatedOrder};
pub use svd::TruncatedSvd;

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
