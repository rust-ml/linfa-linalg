use ndarray::{Array1, ArrayBase, DataMut, Ix2, NdFloat};


pub trait CholeskyUpdate<F> {
    fn cholesky_update_inplace(&mut self, update_vector: &Array1<F>);
}

impl<V,F> CholeskyUpdate<F> for ArrayBase<V,Ix2>
where
    F: NdFloat,
    V: DataMut<Elem=F>,
{
    fn cholesky_update_inplace(&mut self, update_vector: &Array1<F>) {
        let n = self.shape()[0];
        if self.shape()[0] != update_vector.len() {
            panic!("update_vector should be same size as self");
        }
        let mut w=update_vector.to_owned();
        let mut b=F::from(1.0).unwrap();
        for j in 0..n{
            let ljj=self[(j,j)];
            let ljj2=ljj*ljj;
            let wj=w[j];
            let wj2=wj*wj;
            let nljj=(ljj2+wj2/b).sqrt();
            let gamma=ljj2*b+wj2;
            for k in j+1..n{
                let lkj=self[(k,j)];
                let wk=w[k]-wj*lkj/ljj;
                self[(k,j)]=nljj*(lkj/ljj+wj*wk/gamma);
                w[k]=wk;
            }
            b=b+wj2/ljj2;
            self[(j,j)]=nljj;
        }
    }
}



#[cfg(test)]
mod test{
    use approx::assert_abs_diff_eq;
    use super::*;
    use ndarray::{array, Array};
    use crate::cholesky::Cholesky;

    #[test]
    fn test_cholesky_update(){
        let mut arr=array![[1.0, 0.0, 2.0, 3.0, 4.0],
                                        [-2.0, 3.0, 10.0,5.0, 6.0],
                                        [-1.0,-2.0,-7.0, 8.0, 9.0],
                                        [11.0, 12.0, 3.0, 14.0, 5.0],
                                        [8.0, 2.0, 13.0, 4.0, 5.0]];
        arr=arr.t().dot(&arr);
        let mut l_tri = arr.cholesky().unwrap();

        let x = Array::from(vec![1.0, 2.0, 3.0,0.0, 1.0]);
        let vt=x.clone().into_shape((1,x.shape()[0])).unwrap();
        let v=x.clone().into_shape((x.shape()[0],1)).unwrap();

        l_tri.cholesky_update_inplace(&x);

        let restore=l_tri.dot(&l_tri.t());
        let expected=arr+v.dot(&vt);

        assert_abs_diff_eq!(restore, expected, epsilon=1e-7);
    }
}