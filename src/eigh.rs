use ndarray::{Array1, ArrayBase, DataMut, Ix2};

use crate::{check_square, tridiagonal::SymmetricTridiagonal, Float, Result};

fn symmetric_eig<A: Float, S: DataMut<Elem = A>>(
    matrix: ArrayBase<S, Ix2>,
    eigenvectors: bool,
    eps: A,
) -> Result<()> {
    let dim = check_square(&matrix)?;

    let tridiag_decomp = matrix.sym_tridiagonal()?;
    let q_mat = if eigenvectors {
        Some(tridiag_decomp.generate_q())
    } else {
        None
    };
    let (mut diag, mut off_diag) = tridiag_decomp.into_diagonals();

    //if dim == 1 {
    //return Ok((diag, q_mat));
    //}

    let (mut start, mut end) = delimit_subproblem(&diag, &mut off_diag, dim - 1, eps);

    while end != start {
        let subdim = end - start + 1;

        if subdim > 2 {
            let m = end - 1;
            let n = end;

            let x = diag[start] - wilkinson_shift(diag[m], diag[n], off_diag[m]);
            let y = off_diag[start];

            for i in start..n {
                let j = i + 1;
            }
        }
    }

    Ok(())
}

fn delimit_subproblem<A: Float>(
    diag: &Array1<A>,
    off_diag: &mut Array1<A>,
    end: usize,
    eps: A,
) -> (usize, usize) {
    let mut n = end;

    while n > 0 {
        let m = n - 1;
        if off_diag[m].abs() > eps * diag[n].abs() + diag[m].abs() {
            break;
        }
        n -= 1;
    }

    if n == 0 {
        return (0, 0);
    }

    let mut new_start = n - 1;
    while new_start > 0 {
        let m = new_start - 1;
        if off_diag[m].is_zero()
            || off_diag[m].abs() <= eps * (diag[new_start].abs() + diag[m].abs())
        {
            off_diag[m] = A::zero();
        }
        new_start -= 1;
    }

    (new_start, n)
}

/// Computes the wilkinson shift, i.e., the 2x2 symmetric matrix eigenvalue to its tailing
/// component `tnn`.
///
/// The inputs are interpreted as the 2x2 matrix:
///     tmm  tmn
///     tmn  tnn
fn wilkinson_shift<A: Float>(tmm: A, tnn: A, tmn: A) -> A {
    let tmn_sq = tmn * tmn;
    if !tmn_sq.is_zero() {
        let d = (tmm - tnn) * A::from(0.5).unwrap();
        tnn - tmn_sq / (d + d.signum() * (d * d + tmn_sq).sqrt())
    } else {
        tnn
    }
}
