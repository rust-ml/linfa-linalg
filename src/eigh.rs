use ndarray::{s, Array1, Array2, ArrayBase, DataMut, Ix2};

use crate::{
    check_square, givens::GivensRotation, tridiagonal::SymmetricTridiagonal, Float, Result,
};

fn symmetric_eig<A: Float, S: DataMut<Elem = A>>(
    matrix: ArrayBase<S, Ix2>,
    eigenvectors: bool,
    eps: A,
) -> Result<(Array1<A>, Option<Array2<A>>)> {
    let dim = check_square(&matrix)?;

    let tridiag_decomp = matrix.sym_tridiagonal()?;
    let mut q_mat = if eigenvectors {
        Some(tridiag_decomp.generate_q())
    } else {
        None
    };
    let (mut diag, mut off_diag) = tridiag_decomp.into_diagonals();

    if dim == 1 {
        return Ok((diag, q_mat));
    }

    let (mut start, mut end) = delimit_subproblem(&diag, &mut off_diag, dim - 1, eps);

    while end != start {
        let subdim = end - start + 1;

        #[allow(clippy::comparison_chain)]
        if subdim > 2 {
            let m = end - 1;
            let n = end;

            let mut x = diag[start] - wilkinson_shift(diag[m], diag[n], off_diag[m]);
            let mut y = off_diag[start];

            for i in start..n {
                let j = i + 1;

                if let Some((rot, norm)) = GivensRotation::cancel_y(x, y) {
                    if i > start {
                        off_diag[i - 1] = norm;
                    }

                    let mii = diag[i];
                    let mjj = diag[j];
                    let mij = off_diag[i];
                    let cc = rot.c() * rot.c();
                    let ss = rot.s() * rot.s();
                    let cs = rot.c() * rot.s();
                    let b = cs * mij * A::from(2.0f64).unwrap();

                    diag[i] = cc * mii + ss * mjj - b;
                    diag[j] = ss * mii + cc * mjj + b;
                    off_diag[i] = cs * (mii - mjj) + mij * (cc - ss);

                    if i != n - 1 {
                        x = off_diag[i];
                        y = -rot.s() * off_diag[i + 1];
                        off_diag[i + 1] *= rot.c();
                    }

                    if let Some(q) = &mut q_mat {
                        rot.clone()
                            .inverse()
                            .rotate_rows(&mut q.slice_mut(s![.., i..i + 2]))
                            .unwrap();
                    }
                } else {
                    break;
                }
            }

            if off_diag[m].abs() <= eps * (diag[m].abs() + diag[n].abs()) {
                end -= 1;
            }
        } else if subdim == 2 {
            let eigvals = compute_2x2_eigvals(
                diag[start],
                off_diag[start],
                off_diag[start],
                diag[start + 1],
            )
            .unwrap();
            let basis = (eigvals.0 - diag[start + 1], off_diag[start]);

            diag[start] = eigvals.0;
            diag[start + 1] = eigvals.1;

            if let (Some(q), Some((rot, _))) =
                (&mut q_mat, GivensRotation::try_new(basis.0, basis.1, eps))
            {
                rot.rotate_rows(&mut q.slice_mut(s![.., start..start + 2]))
                    .unwrap();
            }
            end -= 1;
        }

        let sub = delimit_subproblem(&diag, &mut off_diag, end, eps);
        start = sub.0;
        end = sub.1;
    }

    Ok((diag, q_mat))
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

fn compute_2x2_eigvals<A: Float>(h00: A, h10: A, h01: A, h11: A) -> Option<(A, A)> {
    let val = (h00 - h11) * A::from(0.5f64).unwrap();
    let discr = h10 * h01 + val * val;
    if discr >= A::zero() {
        let sqrt_discr = discr.sqrt();
        let half_tra = (h00 + h11) * A::from(0.5f64).unwrap();
        Some((half_tra + sqrt_discr, half_tra - sqrt_discr))
    } else {
        None
    }
}
