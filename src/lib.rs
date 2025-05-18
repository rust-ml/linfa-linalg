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

#![allow(clippy::many_single_char_names)]
#![allow(clippy::result_large_err)]

pub mod bidiagonal;
pub mod cholesky;
pub mod eigh;
mod givens;
mod householder;
mod index;
#[cfg(feature = "iterative")]
pub mod lobpcg;
pub mod norm;
pub mod qr;
pub mod reflection;
pub mod svd;
pub mod triangular;
pub mod tridiagonal;
pub mod cholesky_update;

use ndarray::{ArrayBase, Ix2, RawData, ShapeError};
use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LinalgError {
    /// Non-square matrix
    #[error("Matrix of ({rows}, {cols}) is not square")]
    NotSquare { rows: usize, cols: usize },
    /// Matrix rows less than columns
    #[error("Expected matrix rows({rows}) >= cols({cols})")]
    NotThin { rows: usize, cols: usize },
    /// Non-positive definite matrix
    #[error("Matrix is not positive definite")]
    NotPositiveDefinite,
    /// Non-invertible matrix
    #[error("Matrix is non-invertible")]
    NonInvertible,
    /// Unexpected empty matrix
    #[error("Matrix is empty")]
    EmptyMatrix,
    /// Wrong number of columns in matrix
    #[error("Matrix must have {expected} columns, not {actual}")]
    WrongColumns { expected: usize, actual: usize },
    /// Wrong number of rows in matrix
    #[error("Matrix must have {expected} rows, not {actual}")]
    WrongRows { expected: usize, actual: usize },
    /// ShapeError from `ndarray`
    #[error(transparent)]
    Shape(#[from] ShapeError),
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

/// Find largest or smallest eigenvalues
///
/// Corresponds to descending and ascending order
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Order {
    Largest,
    Smallest,
}
