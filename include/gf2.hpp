#pragma once
// ============================================================================
//  GF2 semantic input (Story 05)
// ----------------------------------------------------------------------------
//  `Gf2Input` is the typed requirement object the InputCatalog returns for a
//  GF2 calculation. It bundles the loaded semantic Tensors together with the
//  resolved scalar parameters and their provenance. Calculations receive this
//  object (e.g. via `InputCatalog::require_gf2()`) and never touch input
//  keywords or HDF5 paths directly.
//
//  The tensors arrive via the catalog's definition-driven loading path
//  (`InputCatalog::require_dataset`, Story 05.1), which declares each
//  dataset's axes/symmetries in the dataset-definition table; the symmetry
//  metadata on their axes is ADVISORY (Story 06) and never alters storage.
// ============================================================================

#include "expansion.hpp"
#include "symmetry.hpp"
#include "tensor.hpp"

#include <complex>

namespace cppgw {

struct Gf2Input {
  // AO x AO core Hamiltonian H_0(u,v); the two `ao` axes are bound by one
  // shared Hermitian SymGroup (advisory metadata, real `double` storage).
  Tensor<double, Executor::Host> hcore;

  // AO x MO coefficients: Tensor dims { mo (fastest), ao } in the
  // fastest-to-slowest convention; plain axes, no SymGroup.
  Tensor<double, Executor::Host> mo_coeff;

  // RI density-fitting tensor B^Q_uv: Tensor dims { ao, ao, ri }; the two `ao`
  // axes are bound by one shared Symmetric SymGroup; `ri` is plain.
  Tensor<double, Executor::Host> eri3;

  // AO x AO overlap matrix S(u,v); shared Hermitian AO pair.
  Tensor<double, Executor::Host> overlap;

  // Spin-summed AO density matrix P(k,l); shared Hermitian AO pair.
  Tensor<double, Executor::Host> density_matrix;

  // Sanitized scalar parameters (defaulted or user-supplied; the catalog is
  // the single place where the default is applied).
  double eta = 1e-5;
  bool   eta_was_supplied = false;   // true  - user wrote ETA
                                      // false - registry default was used

  // Story 07.1 Matsubara initial-guess scalars.
  double beta = 0.0;
  size_t matsubara_half_n = 0;
  double mu = 0.0;
};

struct Gf2InitialGuess {
  Tensor<double, Executor::Host> sigma_hf;
  Tensor<double, Executor::Host> fock;
  GridExpansionMatsubara<std::complex<double>, Fermionic> green;
  double electron_count = 0.0;
};

Gf2InitialGuess make_gf2_initial_guess(const Gf2Input& input);

/// @brief Write the bounded Story-07.1 results to a new HDF5 output file.
///
/// Datasets (C-order):
///   g0       : (n_matsubara, nao, nao), complex128, G_0(iw_n) ascending in n
///   sigma_hf : (nao, nao), float64, the static restricted HF self-energy
/// File attributes (calculation input parameters only):
///   eta, eta_was_supplied, beta, matsubara_half_n, n_matsubara, mu
void write_gf2_output_hdf5(const Gf2Input& input, const Gf2InitialGuess& result,
                           const std::string& path);

} // namespace cppgw
