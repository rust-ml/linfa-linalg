//! Provides pure-Rust implementations of linear algebra routines for `ndarray` without depending
//! on external LAPACK/BLAS libraries.
//!
//! ## Eliminating BLAS dependencies
//!
//! If this crate is being used as a BLAS-less replacement for `ndarray-linalg`, make sure to
//! remove `ndarray-linalg` from the entire dependency tree of your crate. This is because
//! `ndarray-linalg`, even as a transitive dependency, forces `ndarray` to be built with the `blas`
//! feature, which forces all matrix multiplications to rely on a BLAS backend. This leads to
//! linker errors if no BLAS backend is specified.

pub mod cholesky;
pub mod triangular;

use ndarray::{ArrayBase, Ix2, RawData};
use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LinalgError {
    /// Non-square matrix encountered in operations requiring square matrices
    #[error("Matrix with {rows} rows and {cols} cols is not square")]
    NotSquare { rows: usize, cols: usize },
    /// Non-positive definite matrix encountered when expecting a positive definite matrix
    #[error("Matrix is not positive definite")]
    NotPositiveDefinite,
}

pub type Result<T> = std::result::Result<T, LinalgError>;

pub(crate) fn check_square<S: RawData>(arr: &ArrayBase<S, Ix2>) -> Result<usize> {
    let (n, m) = (arr.nrows(), arr.ncols());
    if n != m {
        Err(LinalgError::NotSquare { rows: n, cols: m })
    } else {
        Ok(n)
    }
}
