#pragma once

namespace cppgw {

// Built-in test suite (the miscellaneous checks formerly in main.cpp).
// Returns 0 on success; throws std::runtime_error on failure.
int run_tests();

// Story 03: analytic convergence tests for the Fourier transform.
// Returns 0 on success; throws std::runtime_error on failure.
int run_fourier_convergence_tests();

} // namespace cppgw
