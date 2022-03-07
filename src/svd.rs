use ndarray::{s, Array1, Array2, ArrayBase, Data, DataMut, Ix2, NdFloat};

use crate::{
    bidiagonal::Bidiagonal, givens::GivensRotation, index::*, tridiagonal::SymmetricTridiagonal,
    LinalgError, Result,
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

    let bidiag = matrix.bidiagonal()?;
    let mut u = compute_u.then(|| bidiag.generate_u());
    let mut vt = compute_v.then(|| bidiag.generate_vt());
    let (mut diag, mut off_diag) = bidiag.into_diagonals();

    // TODO delimit subproblem

    Ok(())
}

fn delimit_subproblem<A: NdFloat>(
    diag: &mut Array1<A>,
    off_diag: &mut Array1<A>,
    u: &mut Option<Array2<A>>,
    v_t: &mut Option<Array2<A>>,
    is_upper_diag: bool,
    end: usize,
    eps: A,
) -> (usize, usize) {
    let mut n = end;
    while n > 0 {
        let m = n - 1;
        unsafe {
            if *off_diag.at(m) <= eps * (*diag.at(n) + *diag.at(m)) {
                *off_diag.atm(m) = A::zero();
                continue;
            }
        }

        if unsafe { *diag.at(m) } <= eps {
            unsafe { *diag.atm(m) = A::zero() };
            cancel_horizontal_off_diagonal_elt(diag, off_diag, u, v_t, is_upper_diag, m, m + 1);
            if m != 0 {
                cancel_vertical_off_diagonal_elt(diag, off_diag, u, v_t, is_upper_diag, m - 1);
            }
        } else if unsafe { *diag.at(n) } <= eps {
            unsafe { *diag.atm(m) = A::zero() };
            cancel_vertical_off_diagonal_elt(diag, off_diag, u, v_t, is_upper_diag, m);
        } else {
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

        unsafe {
            if *off_diag.at(m) <= eps * (*diag.at(new_start) + *diag.at(m)) {
                *off_diag.atm(m) = A::zero();
                break;
            }
        }

        if unsafe { *diag.at(m) } <= eps {
            unsafe { *diag.atm(m) = A::zero() };
            cancel_horizontal_off_diagonal_elt(diag, off_diag, u, v_t, is_upper_diag, m, n);
            if m != 0 {
                cancel_vertical_off_diagonal_elt(diag, off_diag, u, v_t, is_upper_diag, m - 1);
            }
            break;
        }
        new_start -= 1;
    }

    (new_start, n)
}

fn cancel_horizontal_off_diagonal_elt<A: NdFloat>(
    diag: &mut Array1<A>,
    off_diag: &mut Array1<A>,
    u: &mut Option<Array2<A>>,
    v_t: &mut Option<Array2<A>>,
    is_upper_diag: bool,
    i: usize,
    end: usize,
) {
    let mut v = (off_diag[i], diag[i + 1]);
    off_diag[i] = A::zero();

    for k in i..end {
        if let Some((rot, norm)) = GivensRotation::cancel_x(v.0, v.1) {
            unsafe { *diag.atm(k + 1) = norm };

            if is_upper_diag {
                if let Some(u) = u {
                    rot.inverse()
                        .rotate_rows(&mut u.slice_mut(s![.., i..=k;k-i]))
                        .unwrap()
                }
            } else if let Some(v_t) = v_t {
                rot.rotate_cols(&mut v_t.slice_mut(s![i..=k;k-i, ..]))
                    .unwrap();
            }

            if k + 1 != end {
                unsafe {
                    v.0 = -rot.s() * *off_diag.at(k + 1);
                    v.1 = *diag.at(k + 2); // XXX is this supposed to be +1?
                    *off_diag.atm(k + 1) *= rot.c();
                }
            }
        } else {
            break;
        }
    }
}

fn cancel_vertical_off_diagonal_elt<A: NdFloat>(
    diag: &mut Array1<A>,
    off_diag: &mut Array1<A>,
    u: &mut Option<Array2<A>>,
    v_t: &mut Option<Array2<A>>,
    is_upper_diag: bool,
    i: usize,
) {
    let mut v = (diag[i], off_diag[i]);
    off_diag[i] = A::zero();

    for k in (0..i + 1).rev() {
        if let Some((rot, norm)) = GivensRotation::cancel_y(v.0, v.1) {
            unsafe { *diag.atm(k) = norm };

            if is_upper_diag {
                if let Some(v_t) = v_t {
                    rot.rotate_cols(&mut v_t.slice_mut(s![k..=i;i-k, ..]))
                        .unwrap();
                }
            } else if let Some(u) = u {
                rot.inverse()
                    .rotate_rows(&mut u.slice_mut(s![.., k..=i;i-k]))
                    .unwrap()
            }

            if k > 0 {
                unsafe {
                    v.0 = *diag.at(k - 1);
                    v.1 = rot.s() * *off_diag.at(k - 1);
                    *off_diag.atm(k - 1) *= rot.c();
                }
            }
        } else {
            break;
        }
    }
}
