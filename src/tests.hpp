#pragma once

namespace cppgw {

// Built-in test suite (the miscellaneous checks formerly in main.cpp).
// Returns 0 on success; throws std::runtime_error on failure.
int run_tests();

// Story 03: analytic convergence tests for the Fourier transform.
// Returns 0 on success; throws std::runtime_error on failure.
int run_fourier_convergence_tests();

// Story 04: inverse (Matsubara -> tau) Fourier transform, fermionic tail
// subtraction, and tensor-valued behavior tests.
// Returns 0 on success; throws std::runtime_error on failure.
int run_inverse_fourier_tests();

// Story 06/06.1: index-symmetry (SymmetryGroup) API, label invariant, label
// addressing under duplicate (symmetric) labels, gemm on symmetric tensors,
// the TensorBackend seam (including dense axis permutation), general Tensor
// transpose/permutation behavior, and the GF2 physics construction.
// Returns 0 on success; throws std::runtime_error on failure.
int run_tensor_symmetry_tests();

// Story 06.2: arbitrary-rank Tensor linear solves, LAPACK dispatch, and the
// generic QR fallback.
int run_tensor_solve_tests();

// Story 05: InputCatalog framework -- keyword registry, input-file parser,
// defaults/provenance/sanitization, and semantic GF2 dataset loading (labels +
// advisory symmetries + float64 validation).
// Returns 0 on success; throws std::runtime_error on failure.
int run_input_data_loading_tests();

} // namespace cppgw
