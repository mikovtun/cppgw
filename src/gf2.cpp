// ============================================================================
//  GF2 semantic data (Story 05 / 05.1 cleanup)
// ----------------------------------------------------------------------------
//  `InputCatalog::require_gf2()` is a THIN typed wrapper over the catalog's
//  definition-driven, string-keyed `require_dataset` loading path (the single
//  loading path, defined in src/input.cpp): it guards the selected
//  calculation, fetches each GF2 required dataset by its canonical keyword,
//  reads the sanitized scalars, and re-checks the scalar ranges. No per-
//  dataset layout, symmetry, extent, or HDF5 logic lives in this file.
// ============================================================================

#include "input.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cppgw {

Gf2Input InputCatalog::require_gf2() const {
  if (resolved_.calc != Calc::Gf2)
    throw std::logic_error("InputCatalog::require_gf2: the resolved calculation is "
        + std::string(calc_name(resolved_.calc)) + ", not GF2");

  Gf2Input gf2;
  // Datasets arrive from the catalog's single, definition-driven loading
  // path (rank/type/extent rules + declared labels/symmetries are checked
  // and constructed there, keyed by canonical keyword, cached).
  gf2.hcore          = require_dataset("HCORE");
  gf2.mo_coeff       = require_dataset("MO_COEFF");
  gf2.eri3           = require_dataset("ERI3");
  gf2.overlap        = require_dataset("OVERLAP");
  gf2.density_matrix = require_dataset("DENSITY_MATRIX");

  // ----- resolved scalars (defaulted or supplied by the user) -----
  gf2.eta = resolved_.get_double("ETA");
  gf2.eta_was_supplied = resolved_.supplied("ETA");
  gf2.beta = resolved_.get_double("BETA");
  gf2.matsubara_half_n = static_cast<size_t>(resolved_.get_int("MATSUBARA_HALF_N"));
  gf2.mu = resolved_.get_double("MU");

  // ---- Story 07.1 physical-range checks at the GF2 input-loading boundary.
  if (!std::isfinite(gf2.beta) || gf2.beta <= 0.0) {
    throw std::invalid_argument("keyword BETA: value must be finite and strictly positive (got "
        + std::to_string(gf2.beta) + ")");
  }
  if (gf2.matsubara_half_n == 0) {
    throw std::invalid_argument("keyword MATSUBARA_HALF_N: value must be a positive integer "
        "(zero and negative half-grid sizes are rejected)");
  }
  if (!std::isfinite(gf2.mu)) {
    throw std::invalid_argument("keyword MU: value must be finite (got " + std::to_string(gf2.mu) + ")");
  }

  return gf2;
}

Gf2InitialGuess make_gf2_initial_guess(const Gf2Input& input) {
  using Host = Executor::Host;
  using C = std::complex<double>;

  const size_t n_ao = input.hcore.dims().at(0).dim;
  if (input.hcore.rank() != 2 || input.hcore.dims().at(1).dim != n_ao)
    throw std::invalid_argument("make_gf2_initial_guess: hcore must be a square AO matrix");
  if (input.overlap.rank() != 2 || input.overlap.dims().at(0).dim != n_ao || input.overlap.dims().at(1).dim != n_ao)
    throw std::invalid_argument("make_gf2_initial_guess: overlap AO dimensions do not match hcore");
  if (input.density_matrix.rank() != 2 || input.density_matrix.dims().at(0).dim != n_ao || input.density_matrix.dims().at(1).dim != n_ao)
    throw std::invalid_argument("make_gf2_initial_guess: density_matrix AO dimensions do not match hcore");
  if (input.eri3.rank() != 3 || input.eri3.dims().at(0).dim != n_ao || input.eri3.dims().at(1).dim != n_ao)
    throw std::invalid_argument("make_gf2_initial_guess: eri3 AO dimensions do not match hcore");
  if (!(input.beta > 0.0) || !std::isfinite(input.beta))
    throw std::invalid_argument("make_gf2_initial_guess: beta must be finite and strictly positive");
  if (input.matsubara_half_n == 0)
    throw std::invalid_argument("make_gf2_initial_guess: matsubara_half_n must be positive");
  if (!std::isfinite(input.mu))
    throw std::invalid_argument("make_gf2_initial_guess: mu must be finite");

  const size_t n_ri = input.eri3.dims().at(2).dim;

  double electron_count = 0.0;
  for (size_t u = 0; u < n_ao; ++u)
    for (size_t v = 0; v < n_ao; ++v)
      electron_count += input.density_matrix(u, v) * input.overlap(v, u);

  std::vector<double> rho(n_ri, 0.0);
  for (size_t Q = 0; Q < n_ri; ++Q)
    for (size_t k = 0; k < n_ao; ++k)
      for (size_t l = 0; l < n_ao; ++l)
        rho[Q] += input.eri3(k, l, Q) * input.density_matrix(k, l);

  Tensor<double, Host> sigma({TensorDim{std::string("ao_row"), n_ao}, TensorDim{std::string("ao_col"), n_ao}});
  Tensor<double, Host> fock ({TensorDim{std::string("ao_row"), n_ao}, TensorDim{std::string("ao_col"), n_ao}});

  for (size_t u = 0; u < n_ao; ++u) {
    for (size_t v = 0; v < n_ao; ++v) {
      double J = 0.0;
      for (size_t Q = 0; Q < n_ri; ++Q)
        J += input.eri3(u, v, Q) * rho[Q];

      double K = 0.0;
      for (size_t Q = 0; Q < n_ri; ++Q)
        for (size_t k = 0; k < n_ao; ++k)
          for (size_t l = 0; l < n_ao; ++l)
            K += input.eri3(u, l, Q) * input.density_matrix(k, l) * input.eri3(k, v, Q);

      sigma(u, v) = J - 0.5 * K;
      fock(u, v) = input.hcore(u, v) + sigma(u, v);
    }
  }

  MatsubaraGrid<Fermionic> grid(input.matsubara_half_n, InverseTemperature{input.beta});
  GridExpansionMatsubara<C, Fermionic> green(
      TensorShape{{TensorDim{std::string("ao_row"), n_ao}, TensorDim{std::string("ao_col"), n_ao}}},
      grid);

  Tensor<C, Host> Awork({TensorDim{std::string("ao_equation"), n_ao}, TensorDim{std::string("ao_row"), n_ao}});
  Tensor<C, Host> Iwork({TensorDim{std::string("ao_equation"), n_ao}, TensorDim{std::string("ao_col"), n_ao}});
  Tensor<C, Host> Xwork;

  for (size_t n = 0; n < grid.size(); ++n) {
    const double omega = grid(n).value;
    for (size_t u = 0; u < n_ao; ++u) {
      for (size_t v = 0; v < n_ao; ++v) {
        Awork(u, v) = C(input.mu, omega) * input.overlap(u, v) - C(fock(u, v), 0.0);
        Iwork(u, v) = (u == v) ? C(1.0, 0.0) : C(0.0, 0.0);
      }
    }

    try {
      solve(Awork, Iwork, Xwork,
            {SolveAxis{TensorDimLabel(std::string("ao_equation")),
                       TensorDimLabel(std::string("ao_row"))}});
    } catch (const std::exception& e) {
      throw std::runtime_error("make_gf2_initial_guess: solve failed at Matsubara index "
          + std::to_string(n) + " (omega=" + std::to_string(omega) + "): " + e.what());
    }

    for (size_t u = 0; u < n_ao; ++u)
      for (size_t v = 0; v < n_ao; ++v)
        green.data()(n, u, v) = Xwork(u, v);
  }

  return Gf2InitialGuess{std::move(sigma), std::move(fock), std::move(green), electron_count};
}

void write_gf2_output_hdf5(const Gf2Input& input, const Gf2InitialGuess& result,
                           const std::string& path) {
  using C = std::complex<double>;

  const size_t n_ao = result.sigma_hf.dims()[0].dim;
  if (result.sigma_hf.rank() != 2 || result.sigma_hf.dims()[1].dim != n_ao)
    throw std::invalid_argument("write_gf2_output_hdf5: sigma_hf must be a square AO matrix");
  if (result.green.size() != result.green.data().dims()[0].dim)
    throw std::invalid_argument("write_gf2_output_hdf5: Green's-function storage must be Matsubara-fastest");
  for (size_t i = 1; i < 3; ++i)
    if (result.green.data().dims()[i].dim != n_ao)
      throw std::invalid_argument("write_gf2_output_hdf5: Green's-function AO extents do not match sigma_hf");

  const size_t n_matsu = result.green.size();

  HighFive::File f(path, HighFive::File::Truncate);

  // g0: shape (n_matsubara, nao, nao), C-order (Matsubara axis slowest in the
  // file). The internal Tensor is Matsubara-fastest, so transpose the copy.
  std::vector<C> g0buf(n_matsu * n_ao * n_ao);
  for (size_t v = 0; v < n_ao; ++v)
    for (size_t u = 0; u < n_ao; ++u)
      for (size_t n = 0; n < n_matsu; ++n)
        g0buf[(n * n_ao + u) * n_ao + v] = result.green.data()(n, u, v);
  f.createDataSet<C>("g0", HighFive::DataSpace(std::vector<size_t>{n_matsu, n_ao, n_ao}))
    .write_raw(g0buf.data());

  // sigma_hf: shape (nao, nao), C-order (u slowest). Internal tensor is u-fastest.
  std::vector<double> sbuf(n_ao * n_ao);
  for (size_t v = 0; v < n_ao; ++v)
    for (size_t u = 0; u < n_ao; ++u)
      sbuf[u * n_ao + v] = result.sigma_hf(u, v);
  f.createDataSet<double>("sigma_hf", HighFive::DataSpace(std::vector<size_t>{n_ao, n_ao}))
    .write_raw(sbuf.data());

  // Calculation input parameters (metadata only: no input datasets are copied).
  f.createAttribute("eta", input.eta);
  f.createAttribute("eta_was_supplied", static_cast<int>(input.eta_was_supplied));
  f.createAttribute("beta", input.beta);
  f.createAttribute("matsubara_half_n", static_cast<long long>(input.matsubara_half_n));
  f.createAttribute("n_matsubara", static_cast<long long>(n_matsu));
  f.createAttribute("mu", input.mu);
}

} // namespace cppgw
