// ============================================================================
//  GF2 semantic data (Story 05)
// ----------------------------------------------------------------------------
//  `InputCatalog::require_gf2()` is the typed GF2 requirement request. It
//  loads the HCORE / MO_COEFF / ERI3 datasets through the catalog's single
//  open HDF5 connection, validates element type/rank/extents, and constructs
//  the labelled, symmetry-bearing Tensors. No GF2 numerics live here: this is
//  ingestion + wiring only.
// ============================================================================

#include "input.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cppgw {

namespace {

// A fully checked, loadable float64 dataset.
struct ReadyDataset {
  HighFive::DataSet ds;
  std::string keyword;          // canonical keyword that named it
  std::string path;             // the HDF5 path value from the input
  std::vector<size_t> dims;     // HDF5 shape order: slowest index first
};

// Open the dataset named by a resolved keyword value and enforce the input
// contract: it must exist, be a dense dataset whose element type is
// float64, and have the expected rank. Errors name the keyword, the dataset
// path, and (for rank) the expected rank.
ReadyDataset open_dataset(HighFive::File& file, const std::string& keyword,
                          const std::string& path, size_t expected_rank) {
  // `getDataSet` throws (typically a HighFive error derived from
  // std::runtime_error) when the path does not name an existing object;
  // rethrow with keyword context so the message identifies the request.
  HighFive::DataSet ds;
  try {
    ds = file.getDataSet(path);
  } catch (const std::exception& e) {
    throw std::invalid_argument("keyword " + keyword + ": HDF5 dataset '" + path
        + "' does not exist in the data file (" + e.what() + ")");
  }

  const HighFive::DataType dt = ds.getDataType();
  if (!(dt == HighFive::AtomicType<double>())) {
    throw std::invalid_argument("keyword " + keyword + ": HDF5 dataset '" + path
        + "' must have float64 elements (found " + dt.string()
        + "); only double-precision datasets are accepted for this calculation");
  }

  const std::vector<size_t> dims = ds.getDimensions();
  if (dims.size() != expected_rank) {
    throw std::invalid_argument("keyword " + keyword + ": HDF5 dataset '" + path
        + "' must have rank " + std::to_string(expected_rank)
        + " (found rank " + std::to_string(dims.size()) + ")");
  }
  return ReadyDataset{std::move(ds), keyword, path, std::move(dims)};
}

Tensor<double, Executor::Host> load_square_ao_matrix(ReadyDataset& d, size_t n_ao,
                                                       const char* semantic) {
  const size_t rows = d.dims[0];
  const size_t cols = d.dims[1];
  if (rows != cols) {
    throw std::invalid_argument("keyword " + d.keyword + ": HDF5 dataset '" + d.path
        + "' must be a square AO x AO matrix for " + semantic + " (found "
        + std::to_string(rows) + " x " + std::to_string(cols) + ")");
  }
  if (rows != n_ao) {
    throw std::invalid_argument("keyword " + d.keyword + ": HDF5 dataset '" + d.path
        + "' AO extent does not match HCORE for " + semantic + " (found "
        + std::to_string(rows) + " x " + std::to_string(cols) + ", expected "
        + std::to_string(n_ao) + " x " + std::to_string(n_ao) + ")");
  }
  const SymGroup g = Hermitian();
  Tensor<double, Executor::Host> out(
      {TensorDim{std::string("ao"), n_ao, g}, TensorDim{std::string("ao"), n_ao, g}});
  d.ds.read_raw(out.data());
  return out;
}

} // namespace

Gf2Input InputCatalog::require_gf2() const {
  if (resolved_.calc != Calc::Gf2)
    throw std::logic_error("InputCatalog::require_gf2: the resolved calculation is "
        + std::string(calc_name(resolved_.calc)) + ", not GF2");

  Gf2Input gf2;
  size_t n_ao = 0;   // the single AO extent shared by every GF2 dataset

  // ----- HCORE: AO x AO core Hamiltonian -----
  {
    ReadyDataset d = open_dataset(*file_, "HCORE", resolved_.get_string("HCORE"), 2);
    const size_t rows = d.dims[0];
    const size_t cols = d.dims[1];
    if (rows != cols) {
      throw std::invalid_argument("keyword HCORE: HDF5 dataset '" + d.path
          + "' must be a square AO x AO matrix (found " + std::to_string(rows) + " x "
          + std::to_string(cols) + ")");
    }
    n_ao = rows;
    // Tensor dims are fastest-to-slowest; both axes are the shared `ao`
    // family bound by ONE fresh Hermitian() group (advisory metadata).
    const SymGroup g = Hermitian();
    gf2.hcore = Tensor<double, Executor::Host>(
        {TensorDim{std::string("ao"), n_ao, g}, TensorDim{std::string("ao"), n_ao, g}});
    d.ds.read_raw(gf2.hcore.data());
  }

  // ----- MO_COEFF: AO x MO coefficients (plain axes) -----
  {
    ReadyDataset d = open_dataset(*file_, "MO_COEFF", resolved_.get_string("MO_COEFF"), 2);
    const size_t rows = d.dims[0];   // AO
    const size_t cols = d.dims[1];   // MO
    if (rows != n_ao) {
      throw std::invalid_argument("keyword MO_COEFF: HDF5 dataset '" + d.path
          + "' AO extent does not match HCORE (found " + std::to_string(rows)
          + " x " + std::to_string(cols) + ", expected " + std::to_string(n_ao)
          + " x MO)");
    }
    // Physical matrix is AO x MO. Tensor dims are fastest-to-slowest:
    // { mo (fastest), ao } -- a direct read, no transpose.
    gf2.mo_coeff = Tensor<double, Executor::Host>(
        {TensorDim{std::string("mo"), cols}, TensorDim{std::string("ao"), n_ao}});
    d.ds.read_raw(gf2.mo_coeff.data());
  }

  // ----- ERI3: RI density-fitting tensor B^Q_uv, shape (ri, ao, ao) -----
  {
    ReadyDataset d = open_dataset(*file_, "ERI3", resolved_.get_string("ERI3"), 3);
    const size_t ri = d.dims[0];
    const size_t ao0 = d.dims[1];
    const size_t ao1 = d.dims[2];
    if (ao0 != n_ao || ao1 != n_ao) {
      throw std::invalid_argument("keyword ERI3: HDF5 dataset '" + d.path
          + "' AO extents do not match HCORE (found (RI=" + std::to_string(ri) + ", AO="
          + std::to_string(ao0) + ", AO=" + std::to_string(ao1) + "), expected AO="
          + std::to_string(n_ao) + " on both AO axes)");
    }
    // Tensor dims { ao, ao, ri }; the two `ao` axes are ONE fresh
    // Symmetric() family; `ri` is plain.
    const SymGroup g = Symmetric();
    gf2.eri3 = Tensor<double, Executor::Host>(
        {TensorDim{std::string("ao"), n_ao, g}, TensorDim{std::string("ao"), n_ao, g},
         TensorDim{std::string("ri"), ri}});
    d.ds.read_raw(gf2.eri3.data());
  }

  // ----- OVERLAP / DENSITY_MATRIX: AO x AO Hermitian inputs -----
  {
    ReadyDataset d = open_dataset(*file_, "OVERLAP", resolved_.get_string("OVERLAP"), 2);
    gf2.overlap = load_square_ao_matrix(d, n_ao, "overlap");
  }
  {
    ReadyDataset d = open_dataset(*file_, "DENSITY_MATRIX", resolved_.get_string("DENSITY_MATRIX"), 2);
    gf2.density_matrix = load_square_ao_matrix(d, n_ao, "density matrix");
  }

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
