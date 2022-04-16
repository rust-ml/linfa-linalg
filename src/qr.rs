use crate::{
    householder,
    index::UncheckedIndex,
    reflection::Reflection,
    triangular::{self, IntoTriangular, UPLO},
    LinalgError, Result,
};

use ndarray::{prelude::*, Data, DataMut, OwnedRepr, RawDataClone};

pub trait QRInto {
    type Decomp;

    fn qr_into(self) -> Result<Self::Decomp>;
}

impl<A: NdFloat, S: DataMut<Elem = A>> QRInto for ArrayBase<S, Ix2> {
    type Decomp = QRDecomp<A, S>;

    fn qr_into(mut self) -> Result<Self::Decomp> {
        let (rows, cols) = self.dim();
        if self.nrows() < self.ncols() {
            return Err(LinalgError::NotTall { rows, cols });
        }

        let mut diag = Array::zeros(cols);
        for i in 0..cols {
            diag[i] = householder::clear_column(&mut self, i, 0);
        }

        Ok(QRDecomp { qr: self, diag })
    }
}

pub trait QR {
    type Decomp;

    fn qr(&self) -> Result<Self::Decomp>;
}

impl<A: NdFloat, S: Data<Elem = A>> QR for ArrayBase<S, Ix2> {
    type Decomp = QRDecomp<A, OwnedRepr<A>>;

    fn qr(&self) -> Result<Self::Decomp> {
        self.to_owned().qr_into()
    }
}

#[derive(Debug)]
pub struct QRDecomp<A, S: DataMut<Elem = A>> {
    // qr must be a "tall" matrix (rows >= cols)
    qr: ArrayBase<S, Ix2>,
    // diag length must be equal to qr.ncols
    diag: Array1<A>,
}

impl<A: Clone, S: DataMut<Elem = A> + RawDataClone> Clone for QRDecomp<A, S> {
    fn clone(&self) -> Self {
        Self {
            qr: self.qr.clone(),
            diag: self.diag.clone(),
        }
    }
}

impl<A: NdFloat, S: DataMut<Elem = A>> QRDecomp<A, S> {
    pub fn q(&self) -> Array2<A> {
        householder::assemble_q(&self.qr, 0, |i| self.diag[i])
    }

    pub fn into_r(self) -> ArrayBase<S, Ix2> {
        let ncols = self.qr.ncols();
        let mut r = self.qr.slice_move(s![..ncols, ..ncols]);
        // Should zero out the lower-triangular portion (not the diagonal)
        r.triangular_inplace(UPLO::Upper).unwrap();
        r.diag_mut().assign(&self.diag.mapv_into(A::abs));
        r
    }

    pub fn into_decomp(self) -> (Array2<A>, ArrayBase<S, Ix2>) {
        let q = self.q();
        (q, self.into_r())
    }

    /// Multiplies `b` by transpose of `Q` matrix.
    /// Panics of `b` has wrong shape
    fn qt_mul<Si: DataMut<Elem = A>>(&self, b: &mut ArrayBase<Si, Ix2>) {
        let dim = self.diag.len();
        for i in 0..dim {
            let axis = self.qr.slice(s![i.., i]);
            let refl = Reflection::new(axis, A::zero());

            let mut rows = b.slice_mut(s![i.., ..]);
            refl.reflect_cols(&mut rows);
            rows *= self.diag[i];
        }
    }

    pub fn solve_inplace<Si: DataMut<Elem = A>>(&self, b: &mut ArrayBase<Si, Ix2>) -> Result<()> {
        self.qt_mul(b);
        triangular::solve_triangular_system(
            &self.qr,
            b,
            |rows| (0..rows).rev(),
            |r, c| s![..r, c],
            |i| unsafe { *self.diag.at(i) },
        )
    }

    pub fn solve<Si: Data<Elem = A>>(&self, b: &ArrayBase<Si, Ix2>) -> Result<Array2<A>> {
        let mut b = b.to_owned();
        self.solve_inplace(&mut b)?;
        Ok(b)
    }

    pub fn inverse(&self) -> Result<Array2<A>> {
        let mut res = Array2::eye(self.diag.len());
        self.solve_inplace(&mut res)?;
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn qr() {
        let arr = array![[3.2, 1.3], [4.4, 5.2], [1.3, 6.7]];
        let (q, r) = arr.qr().unwrap().into_decomp();

        assert_abs_diff_eq!(
            q,
            array![
                [0.5720674, -0.4115578],
                [0.7865927, 0.0301901],
                [0.2324024, 0.9108835]
            ],
            epsilon = 1e-5
        );
        assert_abs_diff_eq!(r, array![[5.594, 6.391], [0., 5.725]], epsilon = 1e-3);
    }
}
