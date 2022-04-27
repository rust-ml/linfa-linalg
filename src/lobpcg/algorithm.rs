use ndarray::prelude::*;
use ndarray::{concatenate, ScalarOperand};
use num_traits::{Float, NumCast};
///! Locally Optimal Block Preconditioned Conjugated
///!
///! This module implements the Locally Optimal Block Preconditioned Conjugated (LOBPCG) algorithm,
///which can be used as a solver for large symmetric eigenproblems.
use std::iter::Sum;

use crate::{cholesky::*, eigh::*, norm::*, triangular::*};
use crate::{LinalgError, Order, Result};

use super::{Lobpcg, LobpcgResult};

/// Solve the generalized eigenvalue problem with pencil (A, B)
fn generalized_eig<A: NdFloat>(a: Array2<A>, b: Array2<A>) -> Result<(Array1<A>, Array2<A>)> {
    let (vals_b, vecs_b) = b.eigh_into()?;
    let vals_b_recip = vals_b.mapv_into(|x| (x.max(A::from(1e-10f32).unwrap())).sqrt().recip());
    let vecs_b_tilde = vecs_b * vals_b_recip;
    let a_tilde = vecs_b_tilde.t().dot(&a.dot(&vecs_b_tilde));
    let (vals_a, vecs_a) = a_tilde.eigh_into()?;
    let vecs = vecs_b_tilde.dot(&vecs_a);

    Ok((vals_a, vecs))
}

/// Solve full eigenvalue problem, sort by `order` and truncate to `size`
fn sorted_eig<A: NdFloat>(
    a: Array2<A>,
    b: Option<Array2<A>>,
    size: usize,
    order: Order,
) -> Result<(Array1<A>, Array2<A>)> {
    let n = a.len_of(Axis(0));

    let res = match b {
        Some(b) => generalized_eig(a, b)?,
        _ => a.eigh_into()?,
    };

    // sort and ensure that signs are deterministic
    let (vals, vecs) = res.sort_eig(false);
    let s = vecs.row(0).mapv(|x| x.signum());
    let vecs = vecs * s;

    Ok(match order {
        Order::Largest => (
            vals.slice_move(s![n-size..; -1]),
            vecs.slice_move(s![.., n-size..; -1]),
        ),
        Order::Smallest => (vals.slice_move(s![..size]), vecs.slice_move(s![.., ..size])),
    })
}

/// Masks a matrix with the given `matrix`
fn ndarray_mask<A: NdFloat>(matrix: ArrayView2<A>, mask: &[bool]) -> Array2<A> {
    assert_eq!(mask.len(), matrix.ncols());

    let indices = mask
        .iter()
        .enumerate()
        .filter(|(_, b)| **b)
        .map(|(a, _)| a)
        .collect::<Vec<usize>>();

    matrix.select(Axis(1), &indices)
}

/// Applies constraints ensuring that a matrix is orthogonal to it
///
/// This functions takes a matrix `v` and constraint-matrix `y` and orthogonalize `v` to `y`.
fn apply_constraints<A: NdFloat>(
    mut v: ArrayViewMut<A, Ix2>,
    cholesky_yy: &Array2<A>,
    y: ArrayView2<A>,
) {
    let gram_yv = y.t().dot(&v);

    let u = cholesky_yy
        .solve_triangular_into(gram_yv, UPLO::Lower)
        .unwrap();

    // performs `v = -1 y . u + 1 v`, therefore `v -= y.u`
    ndarray::linalg::general_mat_mul(-A::one(), &y, &u, A::one(), &mut v);
}

/// Orthonormalize `V` with Cholesky factorization
///
/// This also returns the matrix `R` of the `QR` problem
fn orthonormalize<T: NdFloat>(v: Array2<T>) -> Result<(Array2<T>, Array2<T>)> {
    let gram_vv = v.t().dot(&v);
    let gram_vv_fac = gram_vv.cholesky_into()?;

    //assert_abs_diff_eq!(
    //    &gram_vv,
    //    &gram_vv_fac.dot(&gram_vv_fac.t()),
    //    epsilon=NumCast::from(1e-5).unwrap(),
    //);

    let v_t = v.reversed_axes();
    let u = gram_vv_fac
        .solve_triangular_into(v_t, UPLO::Lower)?
        .reversed_axes();

    Ok((u, gram_vv_fac))
}

/// Eigenvalue solver for large symmetric positive definite (SPD) eigenproblems
///
/// # Arguments
/// * `a` - An operator defining the problem, usually a sparse (sometimes also dense) matrix
/// multiplication. Also called the "stiffness matrix".
/// * `x` - Initial approximation of the k eigenvectors. If `a` has shape=(n,n), then `x` should
/// have shape=(n,k).
/// * `m` - Preconditioner to `a`, by default the identity matrix. Should approximate the inverse
/// of `a`.
/// * `y` - Constraints of (n,size_y), iterations are performed in the orthogonal complement of the
/// column-space of `y`. It must be full rank.
/// * `tol` - The tolerance values defines at which point the solver stops the optimization. The approximation
/// of a eigenvalue stops when then l2-norm of the residual is below this threshold.
/// * `maxiter` - The maximal number of iterations
/// * `order` - Whether to solve for the largest or lowest eigenvalues
///
/// The function returns an `LobpcgResult` with the eigenvalue/eigenvector and achieved residual norm
/// for it. All iterations are tracked and the optimal solution returned. In case of an error a
/// special variant `LobpcgResult::NotConverged` additionally carries the error. This can happen when
/// the precision of the matrix is too low (switch then from `f32` to `f64` for example).
pub fn lobpcg<
    A: Float + NdFloat + Sum + ScalarOperand + PartialOrd + Default,
    F: Fn(ArrayView2<A>) -> Array2<A>,
    G: Fn(ArrayViewMut2<A>),
>(
    a: F,
    mut x: Array2<A>,
    m: G,
    y: Option<Array2<A>>,
    tol: f32,
    maxiter: usize,
    order: Order,
) -> LobpcgResult<A> {
    // the initital approximation should be maximal square
    // n is the dimensionality of the problem
    let (n, size_x) = (x.nrows(), x.ncols());
    if size_x > n {
        return Err((
            LinalgError::NotThin {
                rows: size_x,
                cols: n,
            },
            None,
        ));
    }

    /*let size_y = match y {
        Some(ref y) => y.ncols(),
        _ => 0,
    };

    if (n - size_y) < 5 * size_x {
        panic!("Please use a different approach, the LOBPCG method only supports the calculation of a couple of eigenvectors!");
    }*/

    // cap the number of iteration
    let mut iter = usize::min(n * 10, maxiter);
    let tol = NumCast::from(tol).unwrap();

    // calculate cholesky factorization of YY' and apply constraints to initial guess
    let cholesky_yy = y.as_ref().map(|y| {
        let cholesky_yy = y.t().dot(y).cholesky_into().unwrap();
        apply_constraints(x.view_mut(), &cholesky_yy, y.view());
        cholesky_yy
    });

    // orthonormalize the initial guess
    let (x, _) = match orthonormalize(x) {
        Ok(x) => x,
        Err(err) => return Err((err, None)),
    };

    // calculate AX and XAX for Rayleigh quotient
    let ax = a(x.view());
    let xax = x.t().dot(&ax);

    // perform eigenvalue decomposition of XAX
    let (mut lambda, eig_block) =
        sorted_eig(xax, None, size_x, order).map_err(|err| (err, None))?;

    // initiate approximation of the eigenvector
    let mut x = x.dot(&eig_block);
    let mut ax = ax.dot(&eig_block);

    // track residual below threshold
    let mut activemask = vec![true; size_x];

    // track residuals and best result
    let mut residual_norms_history = Vec::new();
    let mut best_result = None;

    let mut previous_block_size = size_x;

    let mut ident: Array2<A> = Array2::eye(size_x);
    let ident0: Array2<A> = Array2::eye(size_x);
    //let two: A = NumCast::from(2.0).unwrap();
    let two = A::from(2.0).unwrap();

    let mut previous_p_ap: Option<(Array2<A>, Array2<A>)> = None;
    let mut explicit_gram_flag = true;

    let final_norm = loop {
        // calculate residual
        let lambda_diag = Array2::from_diag(&lambda);
        let lambda_x = x.dot(&lambda_diag);

        // calculate residual AX - lambdaX
        let r = &ax - &lambda_x;

        // calculate L2 norm of error for every eigenvalue
        let residual_norms = r
            .columns()
            .into_iter()
            .map(|x| x.norm_l2())
            .collect::<Vec<A>>();
        residual_norms_history.push(residual_norms.clone());

        // compare best result and update if we improved
        let sum_rnorm = residual_norms.iter().cloned().sum();
        if best_result
            .as_ref()
            .map(|x: &(_, _, Vec<A>)| x.2.iter().cloned().sum::<A>() > sum_rnorm)
            .unwrap_or(true)
        {
            best_result = Some((lambda.clone(), x.clone(), residual_norms.clone()));
        }

        // disable eigenvalues which are below the tolerance threshold
        activemask = residual_norms
            .iter()
            .zip(activemask.iter())
            .map(|(x, a)| *x > tol && *a)
            .collect();

        // resize identity block if necessary
        let current_block_size = activemask.iter().filter(|x| **x).count();
        if current_block_size != previous_block_size {
            previous_block_size = current_block_size;
            ident = Array2::eye(current_block_size);
        }

        // if we are below the threshold for all eigenvalue or exceeded the number of iteration,
        // abort
        if current_block_size == 0 || iter == 0 {
            break Ok(residual_norms);
        }

        // select active eigenvalues, apply pre-conditioner, orthogonalize to Y and orthonormalize
        let mut active_block_r = ndarray_mask(r.view(), &activemask);
        // apply preconditioner
        m(active_block_r.view_mut());
        // apply constraints to the preconditioned residuals
        if let (Some(ref y), Some(ref cholesky_yy)) = (&y, &cholesky_yy) {
            apply_constraints(active_block_r.view_mut(), cholesky_yy, y.view());
        }
        // orthogonalize the preconditioned residual to x
        // performs `v = -1 y . u + 1 v`, therefore `v -= y.u`
        ndarray::linalg::general_mat_mul(
            -A::one(),
            &x,
            &x.t().dot(&active_block_r),
            A::one(),
            &mut active_block_r,
        );

        let (r, _) = match orthonormalize(active_block_r) {
            Ok(x) => x,
            Err(err) => break Err(err),
        };

        let ar = a(r.view());

        // check whether `A` is of type `f32` or `f64`
        let max_rnorm_float = if A::epsilon() > NumCast::from(1e-8).unwrap() {
            NumCast::from(1.0).unwrap()
        } else {
            NumCast::from(1.0e-8).unwrap()
        };

        // if we are once below the max_rnorm, enable explicit gram flag
        let max_norm = residual_norms.into_iter().fold(A::neg_infinity(), A::max);
        explicit_gram_flag = max_norm <= max_rnorm_float || explicit_gram_flag;

        // perform the Rayleigh Ritz procedure
        let xar = x.t().dot(&ar);
        let mut rar = r.t().dot(&ar);

        // for small residuals calculate covariance matrices explicitely, otherwise approximate
        // them such that X is orthogonal and uncorrelated to the residual R and use eigenvalues of
        // previous decomposition
        let (xax, xx, rr, xr) = if explicit_gram_flag {
            rar = (&rar + &rar.t()) / two;
            let xax = x.t().dot(&ax);

            (
                (&xax + &xax.t()) / two,
                x.t().dot(&x),
                r.t().dot(&r),
                x.t().dot(&r),
            )
        } else {
            (
                lambda_diag,
                ident0.clone(),
                ident.clone(),
                Array2::zeros((size_x, current_block_size)),
            )
        };

        // mask and orthonormalize P and AP
        let mut p_ap = previous_p_ap
            .as_ref()
            .and_then(|(p, ap)| {
                let active_p = ndarray_mask(p.view(), &activemask);
                let active_ap = ndarray_mask(ap.view(), &activemask);

                orthonormalize(active_p).map(|x| (active_ap, x)).ok()
            })
            .and_then(|(active_ap, (active_p, p_r))| {
                // orthonormalize AP with R^{-1} of A
                let active_ap = active_ap.reversed_axes();
                p_r.solve_triangular(&active_ap, UPLO::Lower)
                    .map(|active_ap| (active_p, active_ap.reversed_axes()))
                    .ok()
            });

        // compute symmetric gram matrices and calculate solution of eigenproblem
        //
        // first try to compute the eigenvalue decomposition of the span{R, X, P},
        // if this fails (or the algorithm was restarted), then just use span{R, X}
        let result = p_ap
            .as_ref()
            .ok_or(LinalgError::NonInvertible)
            .and_then(|(active_p, active_ap)| {
                let xap = x.t().dot(active_ap);
                let rap = r.t().dot(active_ap);
                let pap = active_p.t().dot(active_ap);
                let xp = x.t().dot(active_p);
                let rp = r.t().dot(active_p);
                let (pap, pp) = if explicit_gram_flag {
                    ((&pap + &pap.t()) / two, active_p.t().dot(active_p))
                } else {
                    (pap, ident.clone())
                };

                sorted_eig(
                    concatenate![
                        Axis(0),
                        concatenate![Axis(1), xax, xar, xap],
                        concatenate![Axis(1), xar.t(), rar, rap],
                        concatenate![Axis(1), xap.t(), rap.t(), pap]
                    ],
                    Some(concatenate![
                        Axis(0),
                        concatenate![Axis(1), xx, xr, xp],
                        concatenate![Axis(1), xr.t(), rr, rp],
                        concatenate![Axis(1), xp.t(), rp.t(), pp]
                    ]),
                    size_x,
                    order,
                )
            })
            .or_else(|_| {
                p_ap = None;

                sorted_eig(
                    concatenate![
                        Axis(0),
                        concatenate![Axis(1), xax, xar],
                        concatenate![Axis(1), xar.t(), rar]
                    ],
                    Some(concatenate![
                        Axis(0),
                        concatenate![Axis(1), xx, xr],
                        concatenate![Axis(1), xr.t(), rr]
                    ]),
                    size_x,
                    order,
                )
            });

        // update eigenvalues and eigenvectors (lambda is also used in the next iteration)
        let eig_vecs;
        match result {
            Ok((x, y)) => {
                lambda = x;
                eig_vecs = y;
            }
            Err(x) => break Err(x),
        }

        // approximate eigenvector X and conjugate vectors P with solution of eigenproblem
        let (p, ap, tau) = if let Some((active_p, active_ap)) = p_ap {
            // tau are eigenvalues to basis of X
            let tau = eig_vecs.slice(s![..size_x, ..]);
            // alpha are eigenvalues to basis of R
            let alpha = eig_vecs.slice(s![size_x..size_x + current_block_size, ..]);
            // gamma are eigenvalues to basis of P
            let gamma = eig_vecs.slice(s![size_x + current_block_size.., ..]);

            // update AP and P in span{R, P} as linear combination
            let updated_p = r.dot(&alpha) + active_p.dot(&gamma);
            let updated_ap = ar.dot(&alpha) + active_ap.dot(&gamma);

            (updated_p, updated_ap, tau)
        } else {
            // tau are eigenvalues to basis of X
            let tau = eig_vecs.slice(s![..size_x, ..]);
            // alpha are eigenvalues to basis of R
            let alpha = eig_vecs.slice(s![size_x.., ..]);

            // update AP and P as linear combination of the residual matrix R
            let updated_p = r.dot(&alpha);
            let updated_ap = ar.dot(&alpha);

            (updated_p, updated_ap, tau)
        };

        // update approximation of X as linear combinations of span{X, P, R}
        x = x.dot(&tau) + &p;
        ax = ax.dot(&tau) + &ap;

        previous_p_ap = Some((p, ap));

        iter -= 1;
    };

    // retrieve best result and convert norm into `A`
    let (vals, vecs, rnorm) = best_result.unwrap();
    let res = Lobpcg {
        evals: vals,
        evecs: vecs,
        rnorm,
    };

    match final_norm {
        Ok(_) => Ok(res),
        Err(err) => Err((err, Some(res))),
    }
}

#[cfg(test)]
mod tests {
    use super::ndarray_mask;
    use super::orthonormalize;
    use super::sorted_eig;
    use super::Order;
    use super::{lobpcg, Lobpcg};
    use crate::qr::*;
    use approx::assert_abs_diff_eq;
    use ndarray::prelude::*;
    use rand::distributions::{Distribution, Standard};
    use rand::Rng;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    /// Generate random array
    fn random<A>(sh: (usize, usize)) -> Array2<A>
    where
        A: NdFloat,
        Standard: Distribution<A>,
    {
        let mut rng = Xoshiro256Plus::seed_from_u64(3);
        ArrayBase::from_shape_fn(sh, |_| rng.gen::<A>())
    }

    /// Test the `sorted_eigen` function
    #[test]
    fn test_sorted_eigen() {
        let matrix: Array2<f64> = random((10, 10)) * 10.0;
        let matrix = matrix.t().dot(&matrix);

        // return all eigenvectors with largest first
        let (vals, vecs) = sorted_eig(matrix.clone(), None, 10, Order::Largest).unwrap();

        // calculate V * A * V' and compare to original matrix
        let diag = Array2::from_diag(&vals);
        let rec = (vecs.dot(&diag)).dot(&vecs.t());

        assert_abs_diff_eq!(&matrix, &rec, epsilon = 1e-5);
    }

    /// Test the masking function
    #[test]
    fn test_masking() {
        let matrix: Array2<f64> = random((10, 5)) * 10.0;
        let masked_matrix = ndarray_mask(matrix.view(), &[true, true, false, true, false]);
        assert_abs_diff_eq!(
            &masked_matrix.slice(s![.., 2]),
            &matrix.slice(s![.., 3]),
            epsilon = 1e-12,
        );
    }

    /// Test orthonormalization of a random matrix
    #[test]
    fn test_orthonormalize() {
        let matrix: Array2<f64> = random((10, 10)) * 10.0;

        let (n, l) = orthonormalize(matrix.clone()).unwrap();

        // check for orthogonality
        let identity = n.dot(&n.t());
        assert_abs_diff_eq!(&identity, &Array2::eye(10), epsilon = 1e-2);

        // compare returned factorization with QR decomposition
        let qr = matrix.qr().unwrap();
        assert_abs_diff_eq!(
            &qr.into_r().mapv(|x| x.abs()),
            &l.t().mapv(|x| x.abs()),
            epsilon = 1e-2
        );
    }

    #[test]
    fn test_generalized_eigenvalue() {
        let matrix: Array2<f64> = random((10, 10)) * 1.;
        let matrix = matrix.t().dot(&matrix);
        let identity = Array2::eye(10);
        let matrix_inv = matrix.qr().unwrap().inverse().unwrap();

        // check that for the same matrix all eigenvalues are one
        let (vals, _) =
            sorted_eig(matrix.clone(), Some(matrix.clone()), 10, Order::Largest).unwrap();

        assert_abs_diff_eq!(vals, Array1::from_elem(10, 1.0), epsilon = 1e-4);

        let (vals1, _) = sorted_eig(matrix, Some(identity.clone()), 10, Order::Largest).unwrap();
        let (vals2, _) = sorted_eig(identity, Some(matrix_inv), 10, Order::Largest).unwrap();

        assert_abs_diff_eq!(vals1, vals2, epsilon = 1e-5);
        //assert_abs_diff_eq!(vecs1, vecs2, epsilon=1e-5);
    }

    fn assert_symmetric(a: &Array2<f64>) {
        assert_abs_diff_eq!(a.view(), &a.t(), epsilon = 1e-5);
    }

    fn check_eigenvalues(a: &Array2<f64>, order: Order, num: usize, ground_truth_eigvals: &[f64]) {
        assert_symmetric(a);

        let n = a.len_of(Axis(0));
        let x: Array2<f64> = random((n, num));

        let result = lobpcg(|y| a.dot(&y), x, |_| {}, None, 1e-6, n * 3, order);
        match result {
            Ok(Lobpcg { evals, rnorm, .. }) | Err((_, Some(Lobpcg { evals, rnorm, .. }))) => {
                // check convergence
                for (i, norm) in rnorm.into_iter().enumerate() {
                    if norm > 1e-5 {
                        println!("==== Assertion Failed ====");
                        println!("The {}th eigenvalue estimation did not converge!", i);
                        panic!("Too large deviation of residual norm: {} > 0.01", norm);
                    }
                }

                // check correct order of eigenvalues
                if ground_truth_eigvals.len() == num {
                    assert_abs_diff_eq!(
                        &Array1::from(ground_truth_eigvals.to_vec()),
                        &evals,
                        epsilon = num as f64 * 5e-5,
                    )
                }
            }
            Err((err, None)) => panic!("Did not converge: {:?}", err),
        }
    }

    /// Test the eigensolver with a identity matrix problem and a random initial solution
    #[test]
    fn test_eigsolver_diag() {
        let diag = arr1(&[
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20.,
        ]);
        let a = Array2::from_diag(&diag);

        check_eigenvalues(&a, Order::Largest, 3, &[20., 19., 18.]);
        check_eigenvalues(&a, Order::Smallest, 3, &[1., 2., 3.]);
    }

    /// Test the eigensolver with matrix of constructed eigenvalues
    #[test]
    fn test_eigsolver_constructed() {
        let n = 50;
        let tmp = random((n, n));
        //let (v, _) = tmp.qr_square().unwrap();
        let (v, _) = orthonormalize(tmp).unwrap();

        // set eigenvalues in decreasing order
        let t = Array2::from_diag(&Array1::linspace(n as f64, -(n as f64) + 2., n));
        let a = v.dot(&t.dot(&v.t()));

        // find five largest eigenvalues
        check_eigenvalues(&a, Order::Largest, 5, &[50.0, 48.0, 46.0, 44.0, 42.0]);
        check_eigenvalues(&a, Order::Smallest, 5, &[-48.0, -46.0, -44.0, -42.0, -40.0]);
    }

    #[test]
    fn test_eigsolver_constrained() {
        let diag = arr1(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
        let a = Array2::from_diag(&diag);
        let x: Array2<f64> = random((10, 1));
        let y: Array2<f64> = arr2(&[
            [1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0.],
        ])
        .reversed_axes();

        let result = lobpcg(
            |y| a.dot(&y),
            x,
            |_| {},
            Some(y),
            1e-10,
            50,
            Order::Smallest,
        );
        match result {
            Ok(Lobpcg {
                evals,
                evecs,
                rnorm,
            })
            | Err((
                _,
                Some(Lobpcg {
                    evals,
                    evecs,
                    rnorm,
                }),
            )) => {
                // check convergence
                for (i, norm) in rnorm.into_iter().enumerate() {
                    if norm > 0.01 {
                        println!("==== Assertion Failed ====");
                        println!("The {}th eigenvalue estimation did not converge!", i);
                        panic!("Too large deviation of residual norm: {} > 0.01", norm);
                    }
                }

                // should be the third eigenvalue
                assert_abs_diff_eq!(&evals, &Array1::from(vec![3.0]), epsilon = 1e-6);
                assert_abs_diff_eq!(
                    &evecs.column(0).mapv(|x| x.abs()),
                    &arr1(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    epsilon = 1e-5,
                );
            }
            Err((err, None)) => panic!("Did not converge: {:?}", err),
        }
    }
}
