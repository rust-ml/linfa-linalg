//! Eigendecomposition for symmetric square matrices

use ndarray::{s, Array1, Array2, ArrayBase, Data, DataMut, Ix2, NdFloat};

use crate::{
    check_square, givens::GivensRotation, index::*, tridiagonal::SymmetricTridiagonal, Result,
};

fn symmetric_eig<A: NdFloat, S: DataMut<Elem = A>>(
    mut matrix: ArrayBase<S, Ix2>,
    eigenvectors: bool,
    eps: A,
) -> Result<(Array1<A>, Option<Array2<A>>)> {
    let dim = check_square(&matrix)?;
    if dim < 1 {
        return Ok((
            Array1::zeros(0),
            if eigenvectors {
                Some(Array2::zeros((0, 0)))
            } else {
                None
            },
        ));
    }

    let amax = matrix
        .iter()
        .map(|f| f.abs())
        .fold(A::neg_infinity(), |a, b| a.max(b));

    if amax != A::zero() {
        matrix /= amax;
    }

    let tridiag_decomp = matrix.sym_tridiagonal()?;
    let mut q_mat = if eigenvectors {
        Some(tridiag_decomp.generate_q())
    } else {
        None
    };
    let (mut diag, mut off_diag) = tridiag_decomp.into_diagonals();

    if dim == 1 {
        diag *= amax;
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
                        unsafe { *off_diag.atm(i - 1) = norm };
                    }

                    let cc = rot.c() * rot.c();
                    let ss = rot.s() * rot.s();
                    let cs = rot.c() * rot.s();
                    unsafe {
                        let mii = *diag.at(i);
                        let mjj = *diag.at(j);
                        let mij = *off_diag.at(i);
                        let b = cs * mij * A::from(2.0f64).unwrap();
                        *diag.atm(i) = cc * mii + ss * mjj - b;
                        *diag.atm(j) = ss * mii + cc * mjj + b;
                        *off_diag.atm(i) = cs * (mii - mjj) + mij * (cc - ss);

                        if i != n - 1 {
                            x = *off_diag.at(i);
                            y = -rot.s() * *off_diag.at(i + 1);
                            *off_diag.atm(i + 1) *= rot.c();
                        }
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
            .unwrap(); // XXX not sure when this unwrap panics
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

    diag *= amax;
    Ok((diag, q_mat))
}

fn delimit_subproblem<A: NdFloat>(
    diag: &Array1<A>,
    off_diag: &mut Array1<A>,
    end: usize,
    eps: A,
) -> (usize, usize) {
    let mut n = end;

    while n > 0 {
        let m = n - 1;
        unsafe {
            if off_diag.at(m).abs() > eps * (diag.at(n).abs() + diag.at(m).abs()) {
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
            if off_diag.at(m).is_zero()
                || off_diag.at(m).abs() <= eps * (diag.at(new_start).abs() + diag.at(m).abs())
            {
                *off_diag.atm(m) = A::zero();
                break;
            }
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
pub(crate) fn wilkinson_shift<A: NdFloat>(tmm: A, tnn: A, tmn: A) -> A {
    if !tmn.is_zero() {
        let tmn_sq = tmn * tmn;
        let d = (tmm - tnn) * A::from(0.5).unwrap();
        tnn - tmn_sq / (d + d.signum() * (d * d + tmn_sq).sqrt())
    } else {
        tnn
    }
}

fn compute_2x2_eigvals<A: NdFloat>(h00: A, h10: A, h01: A, h11: A) -> Option<(A, A)> {
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

/// Eigendecomposition of symmetric matrices
pub trait EighInto: Sized {
    type EigVal;
    type EigVec;

    /// Calculate eigenvalues and eigenvectors of symmetric matrices, consuming the original
    fn eigh_into(self) -> Result<(Self::EigVal, Self::EigVec)>;
}

impl<A: NdFloat, S: DataMut<Elem = A>> EighInto for ArrayBase<S, Ix2> {
    type EigVal = Array1<A>;
    type EigVec = Array2<A>;

    fn eigh_into(self) -> Result<(Self::EigVal, Self::EigVec)> {
        let (val, vecs) = symmetric_eig(self, true, A::epsilon())?;
        Ok((val, vecs.unwrap()))
    }
}

/// Eigendecomposition of symmetric matrices
pub trait Eigh {
    type EigVal;
    type EigVec;

    /// Calculate eigenvalues and eigenvectors of symmetric matrices
    fn eigh(&self) -> Result<(Self::EigVal, Self::EigVec)>;
}

impl<A: NdFloat, S: Data<Elem = A>> Eigh for ArrayBase<S, Ix2> {
    type EigVal = Array1<A>;
    type EigVec = Array2<A>;

    fn eigh(&self) -> Result<(Self::EigVal, Self::EigVec)> {
        self.to_owned().eigh_into()
    }
}

/// Eigenvalues of symmetric matrices
pub trait EigValshInto {
    type EigVal;

    /// Calculate eigenvalues of symmetric matrices without eigenvectors, consuming the original
    fn eigvalsh_into(self) -> Result<Self::EigVal>;
}

impl<A: NdFloat, S: DataMut<Elem = A>> EigValshInto for ArrayBase<S, Ix2> {
    type EigVal = Array1<A>;

    fn eigvalsh_into(self) -> Result<Self::EigVal> {
        symmetric_eig(self, false, A::epsilon()).map(|(vals, _)| vals)
    }
}

/// Eigenvalues of symmetric matrices
pub trait EigValsh {
    type EigVal;

    /// Calculate eigenvalues of symmetric matrices without eigenvectors
    fn eigvalsh(&self) -> Result<Self::EigVal>;
}

impl<A: NdFloat, S: Data<Elem = A>> EigValsh for ArrayBase<S, Ix2> {
    type EigVal = Array1<A>;

    fn eigvalsh(&self) -> Result<Self::EigVal> {
        self.to_owned().eigvalsh_into()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use ndarray::Axis;

    use crate::LinalgError;

    use super::*;

    #[test]
    fn eigvals_2x2() {
        let (e1, e2) = compute_2x2_eigvals(5., 4., 3., 2.).unwrap();
        assert_abs_diff_eq!(e1, 7.2749172, epsilon = 1e-5);
        assert_abs_diff_eq!(e2, -0.2749172, epsilon = 1e-5);

        let (e1, e2) = compute_2x2_eigvals(6., 2., -1., 3.).unwrap();
        assert_abs_diff_eq!(e1, 5., epsilon = 1e-5);
        assert_abs_diff_eq!(e2, 4., epsilon = 1e-5);

        let (e1, e2) = compute_2x2_eigvals(6., 2., 2., 6.).unwrap();
        assert_abs_diff_eq!(e1, 8., epsilon = 1e-5);
        assert_abs_diff_eq!(e2, 4., epsilon = 1e-5);

        assert_eq!(compute_2x2_eigvals(-2., 3., -3., -2.), None);
    }

    #[test]
    fn symm_eigvals() {
        let (vals, vecs) = symmetric_eig(array![[6., 2.], [2., 6.]], false, f64::EPSILON).unwrap();
        assert_abs_diff_eq!(vals, array![8., 4.]);
        assert_eq!(vecs, None);

        let (vals, vecs) = symmetric_eig(
            array![[1., -5., 7.], [-5., 2., -9.], [7., -9., 3.]],
            false,
            f64::EPSILON,
        )
        .unwrap();
        assert_abs_diff_eq!(vals, array![16.28378, -3.41558, -6.86819], epsilon = 1e-5);
        assert_eq!(vecs, None);
    }

    fn test_eigvecs(a: Array2<f64>, exp_vals: Array1<f64>) {
        let n = a.nrows();
        let (vals, vecs) = symmetric_eig(a.clone(), true, f64::EPSILON).unwrap();
        let vecs = vecs.unwrap();
        assert_abs_diff_eq!(vals, exp_vals, epsilon = 1e-5);

        let s = vecs.t().dot(&vecs);
        assert_abs_diff_eq!(s, Array2::eye(n), epsilon = 1e-5);

        for (i, v) in vecs.axis_iter(Axis(1)).enumerate() {
            let av = a.dot(&v);
            let ev = v.mapv(|x| vals[i] * x);
            assert_abs_diff_eq!(av, ev, epsilon = 1e-5);
        }
    }

    #[test]
    fn sym_eigvecs1() {
        test_eigvecs(
            array![[3., 1., 1.], [1., 3., 1.], [1., 1., 3.]],
            array![5., 2., 2.],
        );
    }

    #[test]
    fn sym_eigvecs2() {
        test_eigvecs(array![[6., 2.], [2., 6.]], array![8., 4.]);
    }

    #[test]
    fn sym_eigvecs3() {
        test_eigvecs(
            array![[1., -5., 7.], [-5., 2., -9.], [7., -9., 3.]],
            array![16.28378, -3.41558, -6.86819],
        );
    }

    #[test]
    fn corner() {
        assert_eq!(
            symmetric_eig(Array2::zeros((0, 0)), false, f64::EPSILON).unwrap(),
            (Array1::zeros(0), None)
        );
        assert_eq!(
            symmetric_eig(Array2::zeros((0, 0)), true, f64::EPSILON).unwrap(),
            (Array1::zeros(0), Some(Array2::zeros((0, 0))))
        );

        symmetric_eig(Array2::zeros((1, 1)), true, f64::EPSILON).unwrap();
        symmetric_eig(Array2::zeros((4, 4)), true, f64::EPSILON).unwrap();
        assert!(matches!(
            symmetric_eig(Array2::zeros((3, 1)), true, f64::EPSILON),
            Err(LinalgError::NotSquare { rows: 3, cols: 1 })
        ));
        // Non-symmetric cases
        symmetric_eig(array![[5., 4.], [3., 2.]], true, f64::EPSILON).unwrap();
        symmetric_eig(array![[-2., 3.], [-3., -2.]], true, f64::EPSILON).unwrap();
    }
}
