use approx::assert_abs_diff_eq;
use ndarray::Array2;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use num_traits::Float;
use rand::{Rng, SeedableRng};
use rand_isaac::IsaacRng;

use ndarray_linalg_rs::cholesky::*;

fn random_hpd<F: 'static + Float + SampleUniform>(rng: &mut impl Rng, n: usize) -> Array2<F> {
    let arr = Array2::random_using(
        (n, n),
        Uniform::new(F::zero(), F::from(100.0).unwrap()),
        rng,
    );
    let mul = &arr.t().dot(&arr);
    Array2::eye(n) + mul
}

macro_rules! cholesky_test {
    ($elem:ident, $rtol:expr) => {
        #[test]
        fn $elem() {
            let mut rng = IsaacRng::seed_from_u64(64);
            let orig: Array2<$elem> = random_hpd(&mut rng, 3);

            let chol = orig.cholesky().unwrap();
            assert_abs_diff_eq!(chol.dot(&chol.t()), orig, epsilon = $rtol);

            let chol = orig.clone().cholesky_into().unwrap();
            assert_abs_diff_eq!(chol.dot(&chol.t()), orig, epsilon = $rtol);

            let mut a = orig.clone();
            let chol = a.cholesky_inplace().unwrap();
            assert_abs_diff_eq!(chol.dot(&chol.t()), orig, epsilon = $rtol);
            assert_abs_diff_eq!(a.dot(&a.t()), orig, epsilon = $rtol);
        }
    };
}

cholesky_test!(f32, 1e-3);
cholesky_test!(f64, 1e-9);
