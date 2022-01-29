pub mod cholesky;

use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LinalgError {
    #[error("Matrix with {rows} rows and {cols} cols is not square")]
    NotSquare { rows: usize, cols: usize },
    #[error("Matrix is not positive definite")]
    NotPositiveDefinite,
}

pub type Result<T> = std::result::Result<T, LinalgError>;
