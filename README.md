# ndarray-linalg-rs

Provides pure-Rust implementations of linear algebra routines for `ndarray` without depending on external LAPACK/BLAS libraries.

## Eliminating BLAS dependencies

If this crate is being used as a BLAS-less replacement for `ndarray-linalg`, make sure to remove `ndarray-linalg` from the entire dependency tree of your crate. This is because `ndarray-linalg`, even as a transitive dependency, forces `ndarray` to be built with the `blas` feature, which forces all matrix multiplications to rely on a BLAS backend. This leads to linker errors if no BLAS backend is specified.

# License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
