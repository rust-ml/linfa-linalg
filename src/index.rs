//! This module provides indexing methods that do checked access in debug mode and unchecked
//! accesses in release mode. Technically these methods are all unsafe. However, they're all
//! internal APIs, so all callers are guaranteed to be well-tested to prevent out-of-bounds
//! accesses.

use ndarray::{ArrayBase, Data, DataMut, Dimension, NdIndex};

pub(crate) trait UncheckedIndex<I> {
    type Elem;
    unsafe fn at(&self, index: I) -> &Self::Elem;
}

pub(crate) trait UncheckedIndexMut<I> {
    type Elem;
    unsafe fn atm(&mut self, index: I) -> &mut Self::Elem;
}

impl<A, S: Data<Elem = A>, D: Dimension, I: NdIndex<D>> UncheckedIndex<I> for ArrayBase<S, D> {
    type Elem = A;

    unsafe fn at(&self, index: I) -> &Self::Elem {
        #[cfg(debug_assertions)]
        {
            self.get(index).unwrap()
        }
        #[cfg(not(debug_assertions))]
        self.uget(index)
    }
}

impl<A, S: DataMut<Elem = A>, D: Dimension, I: NdIndex<D>> UncheckedIndexMut<I>
    for ArrayBase<S, D>
{
    type Elem = A;

    unsafe fn atm(&mut self, index: I) -> &mut Self::Elem {
        #[cfg(debug_assertions)]
        {
            self.get_mut(index).unwrap()
        }
        #[cfg(not(debug_assertions))]
        self.uget_mut(index)
    }
}
