use ndarray::{s, Array1, Array2, ArrayBase, Data, DataMut, Ix2, NdFloat};

use crate::{
    givens::GivensRotation, index::*, tridiagonal::SymmetricTridiagonal, LinalgError, Result,
};

fn svd<A: NdFloat, S: DataMut<Elem = A>>(
    mut matrix: ArrayBase<S, Ix2>,
    compute_u: bool,
    compute_v: bool,
    eps: A,
) -> Result<()> {
    if matrix.is_empty() {
        return Err(LinalgError::EmptyMatrix);
    }
    let (nrows, ncols) = matrix.dim();
    let min_dim = nrows.min(ncols);

    let amax = matrix
        .iter()
        .map(|f| f.abs())
        .fold(A::neg_infinity(), |a, b| a.max(b));

    if amax != A::zero() {
        matrix /= amax;
    }

    Ok(())
}
