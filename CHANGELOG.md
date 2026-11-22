# Changelog

## [Unreleased]

### Fixed
- SVD of rank-deficient matrices: `U * S * Vt` could differ from the input by up to a few percent
  when a singular value was zero up to rounding. The left singular vectors of the 2x2 subproblems
  are now computed from the largest singular value.
- SVD: making a negative singular value positive zeroed the matching column of `U` instead of
  negating it.

## [0.2.1] - 2025-05-14

### Changed
- Fix LOBPCG error message

## [0.2.0] - 2024-08-28

### Changed
- Update ndarray to 0.16 along with associated dependencies
