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

  // ----- resolved scalars (defaulted or supplied by the user) -----
  gf2.eta = resolved_.get_double("ETA");
  gf2.eta_was_supplied = resolved_.supplied("ETA");

  return gf2;
}

} // namespace cppgw
