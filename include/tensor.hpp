#pragma once
#include "common.hpp"
#include "symmetry.hpp"
#include "tensor_buffer.hpp"
#include "tensor_shape.hpp"
#include "backend.hpp"
#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <optional>
#include <span>
#include <ranges>
#include <iostream>
#include <typeinfo>
#include <sstream>

namespace cppgw {
// This file specifies the Tensor class, which provides a generic interface for storing arbitrary-rank
// tensors on either the host or device.
// Tensor shape is specified by a list of TensorDim: a tuple of a string label and size_t dimension
// i.e. a 8x5 matrix is specified by {TensorDim("mu", 8), TensorDim("nu", 5)}
// The list is ordered FASTEST to SLOWEST: the FIRST TensorDim is the fastest
// (innermost, stride 1 / contiguous) and the LAST is the slowest (outermost).
// The slowest (last) index should be the one used for contractions: the free
// gemm(...) contracts X's LAST (slowest) axis against Y's FIRST (fastest) axis.
// This class owns a pointer to a TensorBuffer which implements the storage
// Operations are delegated to the TensorBackend class (the Story 06 backend seam)
// which implements the numerical kernels
//
// Story 06 (index symmetry):
// * Axes may be *bound* to a SymmetryGroup (a handle-based identity over a pair
//   of interchangeable, identical axes: symmetric / Hermitian / antisymmetric).
// * Duplicate labels are legal, but ONLY as one symmetry family (one SymGroup).
// * Symmetry is ADVISORY: it names the family, lets no-op transposes be served
//   by an elementwise op (Tensor::transpose), and is validated at construction.
//   It is never passed to the backend and never compresses storage.
//
// Design policy:
// Strides are recomputed every time a modification to dims is made




template <typename scalar_type, typename exec>
class Tensor {
  using Buffer  = TensorBuffer<scalar_type, exec>;
private:
  // Layout information
  std::vector<TensorDim>  dims_;      // Order: fastest to slowest
  std::vector<size_t>     strides_;
  size_t                  total_elements_ = 0;
  // Storage
  std::shared_ptr<Buffer> buffer_;

  void compute_strides() {
    size_t stride = 1;
    strides_.resize(dims_.size());
    for (size_t i=0; i<dims_.size(); ++i) {
      strides_[i] = stride;
      stride *= dims_[i].dim;   // Accumulate stride for next dim
    }
    total_elements_ = stride;
  }
  

  std::optional<size_t> find_label_index(const TensorDimLabel& label) const {
    for(size_t i=0; i<dims_.size(); ++i)
      if (dims_[i].label == label)
        return i;
    return std::nullopt;
  }

  static void validate_dims_and_symmetry(const std::vector<TensorDim>& dims) {
    // Story 06 label/symmetry invariant (replaces validate_unique_labels).
    //
    // Rules:
    //   (1) All members of one SymGroup are identical axes: same string label
    //       AND same size. (A group is a set of interchangeable identical axes.)
    //   (2) A string label may be repeated only as ONE symmetry family: if a
    //       label occurs on more than one axis, every occurrence must be bound
    //       to the same (non-null) SymGroup. A plain (null-group) label is
    //       therefore necessarily unique, so it stays unambiguously addressable.
    //
    // Violations throw std::invalid_argument naming the label and position(s).
    const size_t n = dims.size();
    for (size_t i = 0; i < n; ++i) {
      // Rule 1: same SymGroup => identical axis (same label and size).
      if (dims[i].symmetry) {
        for (size_t j = 0; j < n; ++j) {
          if (j == i || dims[j].symmetry != dims[i].symmetry)
            continue;
          if (dims[j].label != dims[i].label || dims[j].dim != dims[i].dim)
            throw std::invalid_argument("Tensor: axes " + std::to_string(i) + " and "
                + std::to_string(j) + " bound to symmetry group " + sym_kind_name(dims[i].symmetry->kind())
                + " must have the same label and size (got '" + label_to_string(dims[j].label)
                + "=" + std::to_string(dims[j].dim) + " vs '" + label_to_string(dims[i].label) + "=" + std::to_string(dims[i].dim) + "')");
        }
      }
      // Rule 2: a repeated label must be one single symmetry family.
      for (size_t j = i + 1; j < n; ++j) {
        if (dims[j].label != dims[i].label)
          continue;
        if (!dims[i].symmetry || !dims[j].symmetry || dims[i].symmetry != dims[j].symmetry)
          throw std::invalid_argument("Tensor: duplicate dimension label '"
              + label_to_string(dims[i].label) + "' (axes " + std::to_string(i) + " and "
              + std::to_string(j) + ") must be bound to the same SymmetryGroup to be interchangeable");
      }
    }
  }

  void validate_dims_and_symmetry() const { validate_dims_and_symmetry(dims_); }

  // Shared by both operator() overloads: validates the index count and the
  // bounds of each index, and returns the linear storage offset. Argument #k
  // indexes dimension k in the tensor's own (fast->slow) order.
  template <typename... Idx>
  requires ((std::convertible_to<Idx, size_t> && ...))
  size_t element_offset(Idx... idx) const {
    static_assert(sizeof...(Idx) > 0, "Tensor::operator() needs at least one index");
    const size_t ii[sizeof...(Idx)] = { static_cast<size_t>(idx)... };
    if (sizeof...(Idx) != rank())
      throw std::invalid_argument("Tensor::operator(): expected " + std::to_string(rank())
          + " index/indices (one per dimension, fast->slow order), but got "
          + std::to_string(sizeof...(Idx)));
    size_t off = 0;
    for (size_t a = 0; a < sizeof...(Idx); ++a) {
      if (ii[a] >= dims_[a].dim)
        throw std::out_of_range("Tensor::operator(): index " + std::to_string(ii[a])
            + " is out of range for dimension '" + label_to_string(dims_[a].label)
            + "' (size " + std::to_string(dims_[a].dim) + ")");
      off += ii[a] * strides_[a];
    }
    return off;
  }

  // Shared by both linear() overloads: checks the tensor is allocated and that the
  // linear (flat, rank-agnostic) offset is within bounds. Used for single-offset
  // element access (complements the per-dimension operator()).
  void check_linear(size_t linear_offset) const {
    if (!buffer_)
      throw std::logic_error("Tensor::linear(): tensor is not yet allocated");
    if (linear_offset >= total_elements_)
      throw std::out_of_range("Tensor::linear(): linear index " + std::to_string(linear_offset)
          + " is out of range (tensor size " + std::to_string(total_elements_) + ")");
  }

  // Printing
  void print_data(std::ostream& os, size_t dim, size_t offset) const {
    const size_t n = dims_[dim].dim;

    os << "[";

    if (dim == 0) {
      // Innermost displayed dimension.
      for (size_t i = 0; i < n; ++i) {
        if (i > 0)
          os << ", ";

        os << data()[offset + i * strides_[dim]];
      }
    } else {
      for (size_t i = 0; i < n; ++i) {
        if (i > 0)
          os << ",\n";

        print_data(os, dim - 1,
                   offset + i * strides_[dim]);
      }
    }

    os << "]";
  }

public:
  // ----- Construction -----
  // A default-construction Tensor has no dims and no storage, the "not yet allocated" state
  Tensor() = default;

  // Accept any container of TensorDim (vector, array, etc)
  template <typename R>
  explicit Tensor(R&& inputDims) 
    : dims_(std::begin(inputDims), std::end(inputDims)) { validate_dims_and_symmetry(); compute_strides(); allocate(); }
  // Accept brace-enclosed initializer lists too
  Tensor(std::initializer_list<TensorDim> inputDims) : dims_(inputDims) { validate_dims_and_symmetry(); compute_strides(); allocate(); }

  // ----- Copy / move semantics -----
  // A copied Tensor has INDEPENDENT (deep) storage: copying never aliases the
  // underlying (resizable) buffer, so mutating a copy (e.g. add_dim -> resize)
  // cannot corrupt another Tensor that still references the original data.
  // Moves steal storage (zero copy). This is what makes the by-value "adopt"
  // constructors in expansion.hpp safe.
  Tensor(const Tensor& o)
    : dims_(o.dims_), strides_(o.strides_), total_elements_(o.total_elements_) {
    if (o.buffer_) {
      buffer_ = std::make_shared<Buffer>(total_elements_);
      if (total_elements_ > 0)
        std::copy(o.data(), o.data() + total_elements_, buffer_->data());
    }
  }
  Tensor(Tensor&& o) noexcept = default;
  Tensor& operator=(const Tensor& o) {
    if (this == &o) return *this;
    dims_ = o.dims_;
    strides_ = o.strides_;
    total_elements_ = o.total_elements_;
    if (o.buffer_) {
      buffer_ = std::make_shared<Buffer>(total_elements_);
      if (total_elements_ > 0)
        std::copy(o.data(), o.data() + total_elements_, buffer_->data());
    } else {
      buffer_ = nullptr;
    }
    return *this;
  }
  Tensor& operator=(Tensor&& o) noexcept = default;

  // ----- Layout -----
  size_t total_elements() const { return total_elements_; }
  size_t rank() const { return dims_.size(); }
  const std::vector<TensorDim>& dims() const { return dims_; }
  const std::vector<size_t>& strides() const { return strides_; }

  bool has_label(const TensorDimLabel& label) const {
    return find_label_index(label).has_value();
  }

  // Position of the FIRST dim bearing `label`. Plain labels occur exactly once
  // (Story 06 invariant), so this is the only member; for a symmetric family it
  // is a valid representative ("indexing rab vs. rba is the same").
  size_t label_index(const TensorDimLabel& label) const {
    auto idx = find_label_index(label);
    if (!idx) {
      throw std::invalid_argument("Tensor: label '" + label_to_string(label) + "' not found.");
    }
    return *idx;
  }

  // ALL positions bearing `label` (fast->slow). For a plain label this is
  // exactly one element; for a symmetric family it lists every member —
  // the tool gemm / prepare_output use to reason about duplicate labels.
  std::vector<size_t> label_indices(const TensorDimLabel& label) const {
    std::vector<size_t> out;
    for (size_t i = 0; i < dims_.size(); ++i)
      if (dims_[i].label == label)
        out.push_back(i);
    if (out.empty())
      throw std::invalid_argument("Tensor: label '" + label_to_string(label) + "' not found.");
    return out;
  }

  // ----- Storage -----
  bool allocated() const { return static_cast<bool>(buffer_); }
  void allocate() { 
    // Do not create a buffer object if no dims are in the Tensor
    if (total_elements_ > 0)
      buffer_ = std::make_shared<Buffer>(total_elements_);
  }

  
  scalar_type*          data()        { return buffer_ ? buffer_->data() : nullptr; }
  const scalar_type*    data() const  { return buffer_ ? buffer_->data() : nullptr; }

  // Handle to the underlying storage buffer (used by the TensorBackend kernels)
  Buffer&       buffer()       { if (!buffer_) throw std::logic_error("Tensor::buffer(): tensor is not yet allocated"); return *buffer_; }
  const Buffer& buffer() const { if (!buffer_) throw std::logic_error("Tensor::buffer(): tensor is not yet allocated"); return *buffer_; }

  /// @brief Direct element access, resolving through the tensor's own storage order:
  ///        the `k`th argument indexes dimension `k` in the tensor's (fast->slow)
  ///        dimension order, so the first argument is the fastest axis.
  ///        E.g. for a rank 4 tensor `X`, `X(1,5,3,20)` is a valid call (if all
  ///        indices are in range).
  /// @throws std::out_of_range     if an index is out of range for its dimension
  /// @throws std::invalid_argument if the number of indices differs from the rank
  /// @note  Defined only for `Executor::Host` - the overload is not a candidate
  ///         for device tensors, since device storage cannot be touched from the
  ///         host.
  template <typename... Idx>
  requires (std::same_as<exec, Executor::Host> && (std::convertible_to<Idx, size_t> && ...))
  scalar_type& operator()(Idx... idx) {
    return buffer().data()[element_offset(idx...)];
  }
  template <typename... Idx>
  requires (std::same_as<exec, Executor::Host> && (std::convertible_to<Idx, size_t> && ...))
  const scalar_type& operator()(Idx... idx) const {
    return buffer().data()[element_offset(idx...)];
  }

  /// @brief Element access by a single *linear storage* index: `linear(n)` returns
  ///        the element at flat offset `n` in storage order (`linear(0)` is the first
  ///        element, `linear(n+1)` the next, ...). Bounds-checked.
  ///        The complement of the per-dimension `operator()`: it takes ONE flat,
  ///        rank-agnostic offset rather than one index per dimension, so it composes
  ///        with loops that treat a block of trailing dimensions as one contiguous
  ///        chunk (e.g. the tau--Matsubara contraction over spatial blocks).
  /// @throws std::out_of_range if `n >= total_elements()`
  /// @note  Defined only for `Executor::Host` - device storage cannot be touched
  ///         from the host, and this is then not a viable candidate either.
  scalar_type& linear(size_t linear_offset) requires std::same_as<exec, Executor::Host> {
    check_linear(linear_offset);
    return buffer().data()[linear_offset];
  }
  const scalar_type& linear(size_t linear_offset) const requires std::same_as<exec, Executor::Host> {
    check_linear(linear_offset);
    return buffer().data()[linear_offset];
  }

  // Shape an output tensor to `dims`: allocate + zero-initialize it if it is not yet
  // allocated, or validate that it already matches if it is. Used by operations
  // (mixed-type gemm, ...) that write their result into this tensor. `dims` is
  // taken by value (moved in on allocation) and is only read otherwise.
  void prepare_output(std::vector<TensorDim> dims, const char* what = "Tensor") {
    validate_dims_and_symmetry(dims);
    if (!allocated()) {
      dims_           = std::move(dims);
      compute_strides();
      allocate();
    } else {
      if (rank() != dims.size())
        throw std::invalid_argument(std::string(what) + ": output has wrong rank");
      for (size_t i = 0; i < dims.size(); ++i)
        if (dims_[i] != dims[i])
          throw std::invalid_argument(std::string(what) + ": output dim '"
              + label_to_string(dims_[i].label) + "' incompatible with expected '"
              + label_to_string(dims[i].label) + "'");
    }
  }

  // ============================================================================
  //  Tensor::transpose(i, j)  —  the NO-OP-ECQUIVALENT of a symmetry transpose
  // ----------------------------------------------------------------------------
  //  Story 06 goal (B): when axes i and j are bound to the SAME SymGroup,
  //  "transposing" them never reorders storage — the pair has identical sizes,
  //  so the flat layout is unchanged — and the correct result is served by a
  //  cheap elementwise op (or an identity copy) chosen by the group's kind:
  //    * Symmetric, or Hermitian on a REAL scalar (conjugation is a no-op
  //      when there is no imaginary part to flip)        -> identity copy
  //    * Hermitian on a COMPLEX scalar                   -> elementwise conjugate
  //    * Antisymmetric (complex or real)                 -> elementwise negation
  //  Deliberately NOT a physical index permutation: no restride, no data
  //  movement, works whether or not i,j are adjacent. Tensor has no view type,
  //  so the result is a by-value deep copy — that is fine and intended; the
  //  payoff is that a real transpose is never implemented here.
  //
  //  Otherwise (i or j out of range, or the pair is NOT bound to the same
  //  SymGroup — including both plain) this throws std::logic_error:
  //  the ACTIVE transpose of genuinely distinct axes is a future story.
  //  (Advisory: no data-validity check is done — the tensor is not examined
  //  to confirm it actually satisfies its declared symmetry.)
  // @throws std::logic_error     pair not bound to a shared non-null SymGroup,
  //                              or i/j out of range (not-yet-implemented)
  // @note  Device-safe: the elementwise work goes through the TensorBackend
  //         contract (conjugate / scale), which is executor-templated.
  Tensor transpose(size_t i, size_t j) const {
    if (i >= dims_.size() || j >= dims_.size())
      throw std::logic_error("Tensor::transpose: index out of range (rank "
          + std::to_string(dims_.size()) + "); transpose of non-identical axes not yet "
          + "implemented (Story 06); use a SymmetryGroup-bound pair for the no-op case");
    const SymGroup& g = dims_[i].symmetry;
    // Resolve the pair through the Story 06 symmetry machinery: both positions
    // must be bound to the SAME (non-null) SymGroup (handle identity).
    if (null_symgroup(g) || dims_[j].symmetry != g)
      throw std::logic_error("Tensor::transpose: transpose of non-identical axes not yet "
          "implemented (Story 06); use a SymmetryGroup-bound pair for the no-op case");
    using Backend = TensorBackend<scalar_type, exec>;
    const SymKind k = g->kind();
    // Symmetric -> identity; Hermitian on a real scalar -> identity (the complex
    // conjugate is a no-op when the scalar has no imaginary part), decided by the
    // codebase's floating-point scalar taxonomy (multiprecision.hpp):
    // an elementwise-identical copy, no restride/permute computed.
    if (k == SymKind::Symmetric || (k == SymKind::Hermitian && !numerics::ComplexFloatingPoint<scalar_type>))
      return *this;
    // Hermitian on a complex scalar -> conjugated copy; Antisymmetric -> sign flip.
    Tensor out(*this);
    if (k == SymKind::Hermitian)
      Backend::conjugate(out.buffer());
    else
      Backend::scale(out.buffer(), scalar_type(-1));
    return out;
  }

  // ----- Printers -----
  // `with_data` = false prints only the metadata block (rank/shape/symmetry/size),
  // which is what is useful when a tensor is freshly constructed (all-zero), e.g.
  // the GF2 physics construction in the tests.
  void print(std::ostream& os = std::cout, bool with_data = true) const {
    // Metadata
    os << "Tensor<"
      << typeid(scalar_type).name()
      << ">\n";

    os << "  rank: " << rank() << "\n";
    os << "  shape: (";

    for (size_t i = 0; i < dims_.size(); ++i) {
      if (i > 0)
        os << ", ";

      os << label_to_string(dims_[i].label) << "=" << dims_[i].dim;
      if (dims_[i].symmetry)
        os << "[" << sym_kind_name(dims_[i].symmetry->kind()) << "]";
    }

    os << ")\n";

    os << "  size: " << total_elements_ << "\n";
    os << "  allocated: " << std::boolalpha << allocated() << "\n";

    if (!with_data)
      return;

    // Data
    os << "  data:\n";

    if (!allocated()) {
      os << "<unallocated>\n";
      return;
    }

    if (rank() == 0 || total_elements_ == 0) {
      os << "[]\n";
      return;
    }

    //os << "    ";
    print_data(os, rank() - 1, 0);
    os << "\n";
  }
  friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    tensor.print(os);
    return os;
  }

  // ----- Modifiers -----
  // * Adds a new dimension to the dim list
  // * If slow = true, the new dim is the slowest
  // * If fill = true, copies existing data onto every slice of the new axis (broadcast)
  //   otherwise the data lives only in slice k = 0 and the rest of the buffer is zero
  void add_dim(const TensorDim& newdim, bool slow = true, bool fill = false) {
    if (newdim.dim == 0)
      throw std::invalid_argument("Tensor::add_dim: dimension size must be > 0");
    if (has_label(newdim.label)) {
      // Story 06: the label may only be repeated as ONE symmetry family.
      // The existing members are already one family (the tensor invariant);
      // the new axis may join it only via the SAME SymGroup handle.
      const TensorDim& first = dims_[label_index(newdim.label)];
      if (!first.symmetry || !newdim.symmetry || first.symmetry != newdim.symmetry)
        throw std::invalid_argument("Tensor::add_dim: duplicate dimension label '"
            + label_to_string(newdim.label) + "' (existing axes are "
            + (first.symmetry ? sym_kind_name(first.symmetry->kind()) : std::string("plain"))
            + "; bind the new axis to the same SymmetryGroup to add another member)");
    }

    if (slow)
      dims_.push_back(newdim);
    else
      dims_.insert(dims_.begin(), newdim);   // fastest axis = front of dims_

    compute_strides();

    if (buffer_) {
      // Existing data: grow + broadcast in a single allocation
      buffer_->resize_new_dim(newdim.dim, slow, fill);
    } else if (total_elements_ > 0) {
      // No prior data (e.g. rank-0 tensor): fresh zero storage
      allocate();
    }
  }

  // ----- Operations -----
  // Linear algebra is provided by the free function gemm(...), defined below with the
  // class. It is the mixed-type entry point (X, Y, Z may have different scalar types).

};

// ============================================================================
//  Mixed-type gemm (Tensor frontend)
// ----------------------------------------------------------------------------
// Contract the last dimension of X (named labelX) against the first dimension of Y
// (named labelY), writing the result into Z.
//   * X's labelX must be its last (slowest) dimension,
//   * Y's labelY must be its  first (fastest) dimension,
//   * X, Y, Z may have different scalar types; the arithmetic is carried out in
//     Z's scalar type (the output type) -- see TensorBackend::gemm.
// Z's resulting shape is (X minus labelX) followed by (Y minus labelY); Z is
// allocated (and zero-initialized) if it is not already.
//
// Story 06: labelX/labelY may be symmetric families (duplicate labels bound by
// the same SymGroup); matching is done per-POSITION (X's last dim, Y's first
// dim), so surviving twin axes keep their positions in the output. Symmetry
// itself is advisory only: it is validated, displayed, and usable for no-op
// transposes, but never passed to the backend and not auto-propagated to Z.
template <typename TX, typename TY, typename TZ, typename Exec>
void gemm(const Tensor<TX, Exec>& X,
          const Tensor<TY, Exec>& Y,
                Tensor<TZ, Exec>& Z,
          const TensorDimLabel& labelX,
          const TensorDimLabel& labelY) {
  // 1. Match per position: X's LAST dim must be labelled labelX, Y's FIRST dim
  //    labelY. (For plain labels this is identical to the old label_index rule;
  //    for symmetric families label_index would return the first member, so
  //    position-based matching is the unambiguous reading.)
  if (X.rank() == 0 || X.dims()[X.rank() - 1].label != labelX)
    throw std::invalid_argument("gemm: '"
        + label_to_string(labelX) + "' must be the last dimension of X");
  if (Y.rank() == 0 || Y.dims()[0].label != labelY)
    throw std::invalid_argument("gemm: '"
        + label_to_string(labelY) + "' must be the first dimension of Y");

  // 2. Contracted dimension sizes must agree (the two specific positions)
  const size_t K = X.dims()[X.rank() - 1].dim;
  if (Y.dims()[0].dim != K) {
    std::ostringstream oss;
    oss << "gemm: contracted dimension size mismatch (" << labelX << "=" << K
        << " vs " << labelY << "=" << Y.dims()[0].dim << ")";
    throw std::invalid_argument(oss.str());
  }

  // 3. Output shape: X's dims minus the last, Y's dims minus the first.
  //    Per-position bookkeeping: if a twin of a symmetric family survives, it
  //    keeps its (label, size, SymGroup) entries exactly as on the operand.
  std::vector<TensorDim> result_dims;
  result_dims.reserve((X.rank() - 1) + (Y.rank() - 1));
  size_t M = 1;
  for (size_t i = 0; i + 1 < X.rank(); ++i) {
    result_dims.push_back(X.dims()[i]);
    M *= X.dims()[i].dim;
  }
  size_t N = 1;
  for (size_t i = 1; i < Y.rank(); ++i) {
    result_dims.push_back(Y.dims()[i]);
    N *= Y.dims()[i].dim;
  }

  // 4. Shape/allocate the output, then delegate the arithmetic to the (mixed-type) backend
  Z.prepare_output(result_dims, "gemm");
  TensorBackend<TZ, Exec>::template gemm<TX, TY>(
      X.buffer(), M, K,
      Y.buffer(), N,
      Z.buffer());
}




}
