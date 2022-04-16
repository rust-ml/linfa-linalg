use crate::{
    householder,
    index::UncheckedIndex,
    triangular::{self, IntoTriangular},
    Result,
};

use ndarray::{prelude::*, Data, DataMut, OwnedRepr, RawDataClone};

pub trait QRInto {
    type Decomp;

    fn qr_into(self) -> Self::Decomp;
}

impl<A: NdFloat, S: DataMut<Elem = A>> QRInto for ArrayBase<S, Ix2> {
    type Decomp = QRDecomp<A, S>;

    fn qr_into(mut self) -> Self::Decomp {
        let dims = self.nrows().min(self.ncols());

        let mut diag = Array::zeros(dims);
        for i in 0..dims {
            diag[i] = householder::clear_column(&mut self, i, 0);
        }

        QRDecomp { qr: self, diag }
    }
}

pub trait QR {
    type Decomp;

    fn qr(&self) -> Self::Decomp;
}

impl<A: NdFloat, S: Data<Elem = A>> QR for ArrayBase<S, Ix2> {
    type Decomp = QRDecomp<A, OwnedRepr<A>>;

    fn qr(&self) -> Self::Decomp {
        self.to_owned().qr_into()
    }
}

#[derive(Debug)]
pub struct QRDecomp<A, S: DataMut<Elem = A>> {
    // diag should have same length as MIN(qr.nrows, qr.ncols)
    qr: ArrayBase<S, Ix2>,
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
        let dims = self.diag.len();
        let ncols = self.qr.ncols();
        let mut r = self.qr.slice_move(s![..dims, ..ncols]);
        // Should zero out the lower-triangular portion (not the diagonal)
        r.slice_mut(s![..dims, ..dims])
            .triangular_inplace(crate::triangular::UPLO::Upper)
            .unwrap();
        r.diag_mut().assign(&self.diag);
        r
    }

    pub fn solve_inplace<Si: DataMut<Elem = A>>(&self, b: &mut ArrayBase<Si, Ix2>) -> Result<()> {
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
