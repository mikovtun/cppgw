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
//  The tensors are stored exactly as written in the file: the symmetry
//  metadata on their axes is ADVISORY (Story 06) and never alters storage.
// ============================================================================

#include "symmetry.hpp"
#include "tensor.hpp"

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

  // Sanitized scalar parameters (defaulted or user-supplied; the catalog is
  // the single place where the default is applied).
  double eta = 1e-5;
  bool   eta_was_supplied = false;   // true  - user wrote ETA
                                      // false - registry default was used
};

} // namespace cppgw
