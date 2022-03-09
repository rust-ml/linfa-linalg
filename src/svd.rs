//! Compact singular-value decomposition of matrices

#![allow(clippy::type_complexity)]

use std::ops::MulAssign;

use ndarray::{s, Array1, Array2, ArrayBase, Data, DataMut, Ix2, NdFloat};

use crate::{
    bidiagonal::Bidiagonal, eigh::wilkinson_shift, givens::GivensRotation, index::*, LinalgError,
    Result,
};

fn svd<A: NdFloat, S: DataMut<Elem = A>>(
    mut matrix: ArrayBase<S, Ix2>,
    compute_u: bool,
    compute_v: bool,
    eps: A,
) -> Result<(Option<Array2<A>>, Array1<A>, Option<Array2<A>>)> {
    if matrix.is_empty() {
        return Err(LinalgError::EmptyMatrix);
    }
    let (nrows, ncols) = matrix.dim();
    let dim = nrows.min(ncols);

    let amax = matrix
        .iter()
        .map(|f| f.abs())
        .fold(A::neg_infinity(), |a, b| a.max(b));

    if amax != A::zero() {
        matrix /= amax;
    }

    let bidiag = matrix.bidiagonal()?;
    let is_upper_diag = bidiag.is_upper_diag();
    let mut u = compute_u.then(|| bidiag.generate_u());
    let mut vt = compute_v.then(|| bidiag.generate_vt());
    let (mut diag, mut off_diag) = bidiag.into_diagonals();

    let (mut start, mut end) = delimit_subproblem(
        &mut diag,
        &mut off_diag,
        &mut u,
        &mut vt,
        is_upper_diag,
        dim - 1,
        eps,
    );

    #[allow(clippy::comparison_chain)]
    while end != start {
        let subdim = end - start + 1;

        if subdim > 2 {
            let m = end - 1;
            let n = end;

            let mut vec = unsafe {
                let dm = *diag.at(m);
                let dn = *diag.at(n);
                let fm = *off_diag.at(m);
                let fm1 = *off_diag.at(m - 1);

                let tmm = dm * dm + fm1 * fm1;
                let tmn = dm * fm;
                let tnn = dn * dn + fm * fm;
                let shift = wilkinson_shift(tmm, tnn, tmn);

                let ds = *diag.at(start);
                (ds * ds - shift, ds * *off_diag.at(start))
            };

            for k in start..n {
                let mut subm = unsafe {
                    let m12 = if k == n - 1 {
                        A::zero()
                    } else {
                        *off_diag.at(k + 1)
                    };
                    Array2::from_shape_vec(
                        (2, 3),
                        vec![
                            *diag.at(k),
                            *off_diag.at(k),
                            A::zero(),
                            A::zero(),
                            *diag.at(k + 1),
                            m12,
                        ],
                    )
                    .unwrap()
                };

                if let Some((rot1, norm1)) = GivensRotation::cancel_y(vec.0, vec.1) {
                    rot1.inverse()
                        .rotate_rows(&mut subm.slice_mut(s![.., 0..=1]))
                        .unwrap();

                    let (rot2, norm2);
                    unsafe {
                        if k > start {
                            *off_diag.atm(k - 1) = norm1;
                        }

                        let (v1, v2) = (*subm.at((0, 0)), *subm.at((1, 0)));
                        if let Some((rot, norm)) = GivensRotation::cancel_y(v1, v2) {
                            rot.rotate_cols(&mut subm.slice_mut(s![.., 1..=2])).unwrap();
                            rot2 = Some(rot);
                            norm2 = norm;
                        } else {
                            rot2 = None;
                            norm2 = v1;
                        };
                        *subm.atm((0, 0)) = norm2;
                    }

                    if let Some(ref mut vt) = vt {
                        if is_upper_diag {
                            rot1.rotate_cols(&mut vt.slice_mut(s![k..k + 2, ..]))
                                .unwrap();
                        } else if let Some(rot2) = &rot2 {
                            rot2.rotate_cols(&mut vt.slice_mut(s![k..k + 2, ..]))
                                .unwrap();
                        }
                    }

                    if let Some(ref mut u) = u {
                        if !is_upper_diag {
                            rot1.inverse()
                                .rotate_rows(&mut u.slice_mut(s![.., k..k + 2]))
                                .unwrap();
                        } else if let Some(rot2) = &rot2 {
                            rot2.inverse()
                                .rotate_rows(&mut u.slice_mut(s![.., k..k + 2]))
                                .unwrap();
                        }
                    }

                    unsafe {
                        *diag.atm(k) = *subm.at((0, 0));
                        *diag.atm(k + 1) = *subm.at((1, 1));
                        *off_diag.atm(k) = *subm.at((0, 1));
                        if k != n - 1 {
                            *off_diag.atm(k + 1) = *subm.at((1, 2));
                        }
                        vec.0 = *subm.at((0, 1));
                        vec.1 = *subm.at((0, 2));
                    }
                } else {
                    break;
                }
            }
        } else if subdim == 2 {
            // Solve 2x2 subproblem
            let (rot_u, rot_v) = unsafe {
                let (s1, s2, u2, v2) = compute_2x2_uptrig_svd(
                    *diag.at(start),
                    *off_diag.at(start),
                    *diag.at(start + 1),
                    compute_u && is_upper_diag || compute_v && !is_upper_diag,
                    compute_v && is_upper_diag || compute_u && !is_upper_diag,
                );
                *diag.atm(start) = s1;
                *diag.atm(start + 1) = s2;
                *off_diag.atm(start) = A::zero();

                if is_upper_diag {
                    (u2, v2)
                } else {
                    (v2, u2)
                }
            };

            if let Some(ref mut u) = u {
                rot_u
                    .unwrap()
                    .rotate_rows(&mut u.slice_mut(s![.., start..start + 2]))
                    .unwrap();
            }

            if let Some(ref mut vt) = vt {
                rot_v
                    .unwrap()
                    .inverse()
                    .rotate_cols(&mut vt.slice_mut(s![start..start + 2, ..]))
                    .unwrap();
            }

            end -= 1;
        }

        // Re-delimit the subproblem in case some decoupling occurred.
        let sub = delimit_subproblem(
            &mut diag,
            &mut off_diag,
            &mut u,
            &mut vt,
            is_upper_diag,
            end,
            eps,
        );
        start = sub.0;
        end = sub.1;
    }

    diag *= amax;

    // Ensure singular values are positive
    for i in 0..dim {
        let val = diag[i];
        if val.is_sign_negative() {
            diag[i] = -val;
            if let Some(u) = &mut u {
                u.column_mut(i).mul_assign(-A::zero());
            }
        }
    }

    Ok((u, diag, vt))
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
            if off_diag.at(m).is_zero()
                || off_diag.at(m).abs() <= eps * (diag.at(n).abs() + diag.at(m).abs())
            {
                *off_diag.atm(m) = A::zero();
            } else if diag.at(m).abs() <= eps {
                *diag.atm(m) = A::zero();
                cancel_horizontal_off_diagonal_elt(diag, off_diag, u, v_t, is_upper_diag, m, m + 1);
                if m != 0 {
                    cancel_vertical_off_diagonal_elt(diag, off_diag, u, v_t, is_upper_diag, m - 1);
                }
            } else if diag.at(n).abs() <= eps {
                *diag.atm(n) = A::zero();
                cancel_vertical_off_diagonal_elt(diag, off_diag, u, v_t, is_upper_diag, m);
            } else {
                break;
            }
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
            if off_diag.at(m).abs() <= eps * (diag.at(new_start).abs() + diag.at(m).abs()) {
                *off_diag.atm(m) = A::zero();
                break;
            }
        }

        if unsafe { diag.at(m).abs() } <= eps {
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
                        .rotate_rows(&mut u.slice_mut(s![.., i..=k+1;k-i+1]))
                        .unwrap()
                }
            } else if let Some(v_t) = v_t {
                rot.rotate_cols(&mut v_t.slice_mut(s![i..=k+1;k-i+1, ..]))
                    .unwrap();
            }

            if k + 1 != end {
                unsafe {
                    v.0 = -rot.s() * *off_diag.at(k + 1);
                    v.1 = *diag.at(k + 2);
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
                    rot.rotate_cols(&mut v_t.slice_mut(s![k..=i+1;i-k+1, ..]))
                        .unwrap();
                }
            } else if let Some(u) = u {
                rot.inverse()
                    .rotate_rows(&mut u.slice_mut(s![.., k..=i+1;i-k+1]))
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

// Explicit formulae inspired from the paper "Computing the Singular Values of 2-by-2 Complex
// Matrices", Sanzheng Qiao and Xiaohong Wang.
// http://www.cas.mcmaster.ca/sqrl/papers/sqrl5.pdf
fn compute_2x2_uptrig_svd<A: NdFloat>(
    m11: A,
    m12: A,
    m22: A,
    compute_u: bool,
    compute_v: bool,
) -> (A, A, Option<GivensRotation<A>>, Option<GivensRotation<A>>) {
    let two = A::from(2.0).unwrap();
    let denom = (m11 + m22).hypot(m12) + (m11 - m22).hypot(m12);

    // NOTE: v1 is the singular value that is the closest to m22.
    // This prevents cancellation issues when constructing the vector `csv` below. If we chose
    // otherwise, we would have v1 ~= m11 when m12 is small. This would cause catastrophic
    // cancellation on `v1 * v1 - m11 * m11` below.
    let mut v1 = m11 * m22 * two / denom;
    let mut v2 = denom / two;

    let mut u = None;
    let mut v_t = None;

    if compute_v || compute_u {
        let cv = m11 * m12;
        let sv = v1 * v1 - m11 * m11;
        let (csv, sgn_v) = GivensRotation::new(cv, sv);
        v1 *= sgn_v;
        v2 *= sgn_v;
        if compute_v {
            v_t = Some(csv.clone());
        }

        let cu = (m11 * csv.c() + m12 * csv.s()) / v1;
        let su = (m22 * csv.s()) / v1;
        let (csu, sgn_u) = GivensRotation::new(cu, su);
        v1 *= sgn_u;
        v2 *= sgn_u;
        if compute_u {
            u = Some(csu);
        }
    }

    (v1, v2, u, v_t)
}

/// Compact singular-value decomposition of a non-empty matrix
pub trait SVDInto {
    type U;
    type Vt;
    type Sigma;

    /// Calculates the compact SVD of a matrix, consisting of a square non-negative diagonal
    /// matrix `S` and rectangular semi-orthogonal matrices `U` and `Vt`, such that `U * S * Vt`
    /// yields the original matrix. Only the diagonal elements of `S` is returned.
    fn svd_into(
        self,
        calc_u: bool,
        calc_vt: bool,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::Vt>)>;
}

impl<A: NdFloat, S: DataMut<Elem = A>> SVDInto for ArrayBase<S, Ix2> {
    type U = Array2<A>;
    type Vt = Array2<A>;
    type Sigma = Array1<A>;

    fn svd_into(
        self,
        calc_u: bool,
        calc_vt: bool,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::Vt>)> {
        svd(self, calc_u, calc_vt, A::from(1e-15).unwrap())
    }
}

/// Compact singular-value decomposition of a non-empty matrix
pub trait SVD {
    type U;
    type Vt;
    type Sigma;

    /// Calculates the compact SVD of a matrix, consisting of a square non-negative diagonal
    /// matrix `S` and rectangular semi-orthogonal matrices `U` and `Vt`, such that `U * S * Vt`
    /// yields the original matrix. Only the diagonal elements of `S` is returned.
    fn svd(
        &self,
        calc_u: bool,
        calc_vt: bool,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::Vt>)>;
}

impl<A: NdFloat, S: Data<Elem = A>> SVD for ArrayBase<S, Ix2> {
    type U = Array2<A>;
    type Vt = Array2<A>;
    type Sigma = Array1<A>;

    fn svd(
        &self,
        calc_u: bool,
        calc_vt: bool,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::Vt>)> {
        svd(self.to_owned(), calc_u, calc_vt, A::from(1e-15).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn svd_test() {
        let (u, s, vt) = svd(array![[3.0, 0.], [0., -2.]], true, true, 1e-15).unwrap();
        assert_abs_diff_eq!(s, array![3., 2.], epsilon = 1e-7);
        assert_abs_diff_eq!(u.unwrap(), array![[1., 0.], [0., -1.]], epsilon = 1e-7);
        assert_abs_diff_eq!(vt.unwrap(), array![[1., 0.], [0., 1.]], epsilon = 1e-7);

        let (u, s, vt) = svd(array![[1., 0., -1.], [-2., 1., 4.]], false, false, 1e-15).unwrap();
        assert_abs_diff_eq!(s, array![0.51371, 4.76824], epsilon = 1e-5);
        assert!(u.is_none());
        assert!(vt.is_none());
    }

    fn test_svd_props(a: Array2<f64>, exp_s: Array1<f64>) {
        let (u, s, vt) = svd(a.clone(), true, true, 1e-15).unwrap();
        let (u, vt) = (u.unwrap(), vt.unwrap());
        assert_abs_diff_eq!(s, exp_s, epsilon = 1e-5);
        assert!(s.iter().copied().all(f64::is_sign_positive));
        assert_abs_diff_eq!(u.dot(&Array2::from_diag(&s)).dot(&vt), a, epsilon = 1e-5);

        let (u2, s2, vt2) = svd(a.clone(), false, true, 1e-15).unwrap();
        assert!(u2.is_none());
        assert_abs_diff_eq!(s2, s, epsilon = 1e-9);
        assert_abs_diff_eq!(vt2.unwrap(), vt, epsilon = 1e-9);

        let (u3, s3, vt3) = svd(a.clone(), true, false, 1e-15).unwrap();
        assert!(vt3.is_none());
        assert_abs_diff_eq!(s3, s, epsilon = 1e-9);
        assert_abs_diff_eq!(u3.unwrap(), u, epsilon = 1e-9);

        let (u4, s4, vt4) = svd(a, false, false, 1e-15).unwrap();
        assert!(vt4.is_none());
        assert!(u4.is_none());
        assert_abs_diff_eq!(s4, s, epsilon = 1e-9);
    }

    #[test]
    fn svd_props() {
        test_svd_props(array![[-2., 1., 4.]], array![21f64.sqrt()]);
        test_svd_props(array![[1., 1.], [1., 1.]], array![2., 0.]);
        test_svd_props(
            array![[-3., 4.], [4.3, 2.1], [6.6, 8.7]],
            array![5.2633658, 11.80876],
        );
        test_svd_props(
            array![
                [10.74785316637712, -5.994983325167452, -6.064492921857296],
                [-4.149751381521569, 20.654504205822462, -4.470436210703133],
                [-22.772715014220207, -1.4554372570788008, 18.108113992170573]
            ],
            array![2.23811978e+01, 3.16188022e+01, 0.],
        );
    }
}
