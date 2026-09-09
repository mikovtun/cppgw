#pragma once
#include "grid.hpp"
#include "types.hpp"
#include "grid/tau.hpp"
#include "grid/matsubara.hpp"
#include "tensor.hpp"
#include "multiprecision.hpp"
#include "chebyshev/basis.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cppgw {
// Make the numeric concepts visible in the cppgw namespace (as in chebyshev/basis.hpp)
using numerics::FloatingPoint;
using numerics::RealFloatingPoint;
using numerics::ComplexFloatingPoint;

// ============================================================================
//  Unified interface for Tensor representations of a function on a
//  *function space*, expressed either as point values on a grid or as
//  coefficients in a basis.
//
//  Every representation below shares the same core:
//    * a function-space tag   : using space = ...
//    * a named tensor axis    : using DimLabel  /  dim_label
//    * owning data            : data() -> Tensor<data_type, Executor::Host>
//    * a length               : size() -> number of grid points / coefficients
//    * a uniform constructor pair (mirroring each other):
//        1. (TensorShape spatial, <representation>)  -> new zero tensor + repr axis
//        2. (Tensor spatial,       <representation>) -> adopt, adding/broadcasting
//                                                       the repr axis if absent
//
//  The four concrete classes:
//    GridExpansionTau            grid over the ImaginaryTime space
//    GridExpansionMatsubara      grid over the (Matsubara) ImaginaryFrequency space
//    ChebyshevExpansionTau       Chebyshev basis over the ImaginaryTime space
//    ChebyshevExpansionMatsubara Chebyshev basis over the ImaginaryFrequency space
//
//  Transformations (see transform.hpp) operate on anything satisfying the
//  TensorExpansion concept below; they deliberately leave the representation
//  choice to the caller and the construction to these classes.
// ============================================================================

// A data-owning tensor representation of a function on its function space.
template <class R, typename scalar_type, typename exec>
concept TensorExpansion =
  HasFunctionSpace<R> &&
  requires(const R& x) {
    typename R::space;
    typename R::DimLabel;
    { x.dim_label };
    { x.size() };
    { x.data() };
    { x(0) };   // the i-th point of the representation (grid point / basis node)
    requires std::convertible_to<decltype(x.size()), size_t>;
    requires std::same_as<std::remove_cvref_t<decltype(x.data())>, Tensor<scalar_type, exec>>;
  };


// ----- shared construction helpers (all representations) --------------------

// Count the dims of t that are NOT repr_label (i.e. the "spatial" rank).
template <FloatingPoint dt>
size_t spatial_rank(const Tensor<dt, Executor::Host>& t, const TensorDimLabel& repr_label) {
  size_t r = 0;
  for (const auto& d : t.dims())
    if (d.label != repr_label) ++r;
  return r;
}

// Ensure t carries the representation axis of the given size, as the FASTEST
// (innermost) axis:
//   * absent  -> add it, broadcasting the existing data onto every point/coeff
//   * present -> require its size to agree with repr_size
// `what` is only used in the error messages.
template <FloatingPoint dt>
void ensure_repr_dim(Tensor<dt, Executor::Host>& t, const TensorDimLabel& repr_label,
                     size_t repr_size, const char* what) {
  if (spatial_rank<dt>(t, repr_label) == 0)
    throw std::invalid_argument(std::string(what) + ": input must have at least one spatial dim");
  if (t.has_label(repr_label)) {
    const size_t idx = t.label_index(repr_label);
    if (t.dims()[idx].dim != repr_size)
      throw std::invalid_argument(std::string(what)
          + ": representation dim size (" + std::to_string(t.dims()[idx].dim)
          + ") does not match the representation size (" + std::to_string(repr_size) + ")");
  } else {
    t.add_dim(TensorDim{repr_label, repr_size}, /*slow=*/false, /*fill=*/true);
  }
}


// ============================================================================
//  GRID EXPANSIONS
// ============================================================================

// A data-owning Tensor-valued function sampled on a concrete ImaginaryTime grid G.
//   data_( spatial_indices..., <grid> )   with <grid> the FASTEST axis (size == grid.size()),
//   labelled G's GridDimLabel<ImaginaryTimeSpace> so it can be contracted as a gemm index.
template <FloatingPoint data_type, ImaginaryTimeGrid G>
class GridExpansionTau {
public:
  using space      = ImaginaryTimeSpace;
  using point_type = typename G::point_type;          // == ImaginaryTime
  using DimLabel   = GridDimLabel<space>;
  using grid_type  = G;
  using scalar_type = data_type;

  // The tensor axis this representation owns; it must be exactly the grid's label.
  inline static constexpr DimLabel dim_label{};
  static_assert( std::same_as<std::remove_cvref_t<decltype(G::dim_label)>, DimLabel>,
      "GridExpansionTau<G>: grid's dim_label must be GridDimLabel<ImaginaryTimeSpace>" );

private:
  Tensor<data_type, Executor::Host> data_;
  G                                 grid_;

public:
  GridExpansionTau() = delete;

  // ----- Accessors -----
  Tensor<data_type, Executor::Host>&       data()       { return data_; }
  const Tensor<data_type, Executor::Host>& data() const { return data_; }
  size_t                                   size()  const { return grid_.size(); }
  const G&                                 grid()  const { return grid_; }
  const std::vector<point_type>&           points() const { return grid_.points(); }
  point_type                               operator()(size_t i) const { return grid_(i); }

  // ----- Constructors -----
  // 1. spatial shape + grid: new all-zero spatial x grid tensor.
  explicit GridExpansionTau(TensorShape spatial, G grid) : grid_(std::move(grid)) {
    if (spatial.size() == 0)
      throw std::invalid_argument("GridExpansionTau: spatial shape must have at least one dim");
    spatial.insert_fast(TensorDim{dim_label, grid_.size()});   // grid axis = fastest
    data_ = Tensor<data_type, Executor::Host>(spatial);
  }

  // 2. spatial tensor + grid: adopt it, adding the grid axis (fastest) + broadcasting if absent.
  explicit GridExpansionTau(Tensor<data_type, Executor::Host> d, G grid) : grid_(std::move(grid)) {
    ensure_repr_dim<data_type>(d, dim_label, grid_.size(), "GridExpansionTau");
    data_ = std::move(d);
  }
};


// A data-owning Tensor-valued function sampled on the Matsubara grid S (ImaginaryFrequency).
//   data_( spatial_indices..., <Matsubara> )   with <Matsubara> the FASTEST axis.
// S is the statistics tag (Fermionic/Bosonic); the concrete grid is MatsubaraGrid<S>.
template <FloatingPoint data_type, StatisticsTag S>
class GridExpansionMatsubara {
public:
  using space      = ImaginaryFrequencySpace;
  using point_type = typename MatsubaraGrid<S>::point_type;   // == ImaginaryFrequency
  using DimLabel   = GridDimLabel<space>;
  using grid_type  = MatsubaraGrid<S>;
  using scalar_type = data_type;

  inline static constexpr DimLabel dim_label{};
  static_assert( std::same_as<std::remove_cvref_t<decltype(grid_type::dim_label)>, DimLabel>,
      "GridExpansionMatsubara<S>: MatsubaraGrid's dim_label must be GridDimLabel<ImaginaryFrequencySpace>" );

private:
  Tensor<data_type, Executor::Host> data_;
  grid_type                         grid_;

public:
  GridExpansionMatsubara() = delete;

  // ----- Accessors -----
  Tensor<data_type, Executor::Host>&       data()       { return data_; }
  const Tensor<data_type, Executor::Host>& data() const { return data_; }
  size_t                                   size()  const { return grid_.size(); }
  const grid_type&                         grid()  const { return grid_; }
  const std::vector<point_type>&           points() const { return grid_.points(); }
  std::vector<point_type>                  points_no_zero() const { return grid_.points_no_zero(); }
  // the i-th Matsubara point (of the full set, zero mode included for Bosonic)
  point_type                               operator()(size_t i) const { return grid_(i); }

  // ----- Constructors -----
  explicit GridExpansionMatsubara(TensorShape spatial, grid_type grid) : grid_(std::move(grid)) {
    if (spatial.size() == 0)
      throw std::invalid_argument("GridExpansionMatsubara: spatial shape must have at least one dim");
    spatial.insert_fast(TensorDim{dim_label, grid_.size()});   // Matsubara axis = fastest
    data_ = Tensor<data_type, Executor::Host>(spatial);
  }

  explicit GridExpansionMatsubara(Tensor<data_type, Executor::Host> d, grid_type grid) : grid_(std::move(grid)) {
    ensure_repr_dim<data_type>(d, dim_label, grid_.size(), "GridExpansionMatsubara");
    data_ = std::move(d);
  }
};


// ============================================================================
//  CHEBYSHEV BASIS EXPANSIONS
//
//   data_( spatial_indices..., <coeff> )  with <coeff> the FASTEST axis of size (order + 1),
//   labelled ChebyshevExpansionDimLabel<space>. The coefficient values live in data_;
//   the basis (nodes / evaluation) is provided by ChebyshevBasisImpl.
// ============================================================================

// A data-owning Tensor Chebyshev expansion over the ImaginaryTime space.
template <FloatingPoint data_type, RealFloatingPoint cheb_impl_type = double>
class ChebyshevExpansionTau {
public:
  using space      = ImaginaryTimeSpace;
  using DimLabel   = ChebyshevExpansionDimLabel<space>;
  using basis_type = ChebyshevBasisImpl<cheb_impl_type>;

  inline static constexpr DimLabel dim_label{};

private:
  basis_type                    impl_;
  Tensor<data_type, Executor::Host> data_;

public:
  ChebyshevExpansionTau() = delete;

  // ----- Accessors -----
  Tensor<data_type, Executor::Host>&       data()  { return data_; }
  const Tensor<data_type, Executor::Host>& data() const { return data_; }
  size_t                                   size()  const { return impl_.size(); }
  size_t                                   order() const { return impl_.order(); }
  const basis_type&                        impl()  const { return impl_; }
  std::vector<cheb_impl_type>              nodes() const { return impl_.nodes(); }
  // the i-th Chebyshev node (the i-th "point" of this expansion)
  cheb_impl_type                           operator()(size_t i) const {
    const std::vector<cheb_impl_type> ns = impl_.nodes();
    if (i >= ns.size())
      throw std::out_of_range("ChebyshevExpansionTau: node index out of range");
    return ns[i];
  }

  // ----- Constructors -----
  // 1. spatial shape + order: new all-zero spatial x coefficient tensor.
  explicit ChebyshevExpansionTau(TensorShape spatial, size_t order) : impl_(order) {
    if (spatial.size() == 0)
      throw std::invalid_argument("ChebyshevExpansionTau: spatial shape must have at least one dim");
    spatial.insert_fast(TensorDim{dim_label, impl_.size()});   // coeff axis = fastest
    data_ = Tensor<data_type, Executor::Host>(spatial);
  }

  // 2. spatial tensor + order: adopt it, adding the coefficient axis (fastest) if absent.
  explicit ChebyshevExpansionTau(Tensor<data_type, Executor::Host> d, size_t order) : impl_(order) {
    ensure_repr_dim<data_type>(d, dim_label, impl_.size(), "ChebyshevExpansionTau");
    data_ = std::move(d);
  }
};


// A data-owning Tensor Chebyshev expansion over the (Matsubara) ImaginaryFrequency space.
// S is the statistics tag; the basis itself does not depend on S (S only tags the space).
template <FloatingPoint data_type, StatisticsTag S, RealFloatingPoint cheb_impl_type = double>
class ChebyshevExpansionMatsubara {
public:
  using space      = ImaginaryFrequencySpace;
  using DimLabel   = ChebyshevExpansionDimLabel<space>;
  using basis_type = ChebyshevBasisImpl<cheb_impl_type>;

  inline static constexpr DimLabel dim_label{};

private:
  basis_type                     impl_;
  Tensor<data_type, Executor::Host> data_;

public:
  ChebyshevExpansionMatsubara() = delete;

  // ----- Accessors -----
  Tensor<data_type, Executor::Host>&       data()  { return data_; }
  const Tensor<data_type, Executor::Host>& data() const { return data_; }
  size_t                                   size()  const { return impl_.size(); }
  size_t                                   order() const { return impl_.order(); }
  const basis_type&                        impl()  const { return impl_; }
  std::vector<cheb_impl_type>              nodes() const { return impl_.nodes(); }
  // the i-th Chebyshev node (the i-th "point" of this expansion)
  cheb_impl_type                           operator()(size_t i) const {
    const std::vector<cheb_impl_type> ns = impl_.nodes();
    if (i >= ns.size())
      throw std::out_of_range("ChebyshevExpansionMatsubara: node index out of range");
    return ns[i];
  }

  // ----- Constructors -----
  explicit ChebyshevExpansionMatsubara(TensorShape spatial, size_t order) : impl_(order) {
    if (spatial.size() == 0)
      throw std::invalid_argument("ChebyshevExpansionMatsubara: spatial shape must have at least one dim");
    spatial.insert_fast(TensorDim{dim_label, impl_.size()});   // coeff axis = fastest
    data_ = Tensor<data_type, Executor::Host>(spatial);
  }

  explicit ChebyshevExpansionMatsubara(Tensor<data_type, Executor::Host> d, size_t order) : impl_(order) {
    ensure_repr_dim<data_type>(d, dim_label, impl_.size(), "ChebyshevExpansionMatsubara");
    data_ = std::move(d);
  }
};


// ----- interface conformance ------------------------------------------------
static_assert( TensorExpansion<GridExpansionTau<double, UniformImaginaryTimeGrid>, double, Executor::Host>);
static_assert( TensorExpansion<GridExpansionMatsubara<double, Fermionic>, double, Executor::Host>);
static_assert( TensorExpansion<ChebyshevExpansionTau<double>, double, Executor::Host>);
static_assert( TensorExpansion<ChebyshevExpansionMatsubara<double, Fermionic>, double, Executor::Host>);

}
