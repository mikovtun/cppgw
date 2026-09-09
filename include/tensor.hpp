#pragma once
#include "common.hpp"
#include "tensor_buffer.hpp"
#include "tensor_shape.hpp"
#include "linalgbackend.hpp"
#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <optional>
#include <span>
#include <unordered_set>
#include <ranges>
#include <iostream>
#include <typeinfo>
#include <sstream>

namespace cppgw {
// This file specifies the Tensor class, which provides a generic interface for storing arbitrary-rank
// tensors on either the host or device.
// Tensor shape is specified by a list of TensorDim: a tuple of a string label and size_t dimension
// i.e. a 8x5 matrix is specified by {TensorDim("mu", 8), TensorDim("nu", 5)}
// The later the index in the list, the faster it is (last index is fast and should be used for contractions)
// This class owns a pointer to a TensorBuffer which implements the storage
// Operations are delegated to the LinAlgBackend class which implements operations
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

  static void validate_unique_labels(const std::span<TensorDim>& dims) {
    std::unordered_set<TensorDimLabel, TensorDimLabelHash> seen;
    for (const auto& d : dims) {
      if (!seen.insert(d.label).second)
        throw std::invalid_argument("Tensor: duplicate dimension label '"
            + label_to_string(d.label) + "'");
    }
  }

  void validate_unique_labels() const { validate_unique_labels(dims_); }

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
    : dims_(std::begin(inputDims), std::end(inputDims)) { compute_strides(); allocate(); }
  // Accept brace-enclosed initializer lists too
  Tensor(std::initializer_list<TensorDim> inputDims) : dims_(inputDims) { compute_strides(); allocate(); }

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

  size_t label_index(const TensorDimLabel& label) const {
    auto idx = find_label_index(label);
    if (!idx) {
      throw std::invalid_argument("Tensor: label '" + label_to_string(label) + "' not found.");
    }
    return *idx;
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

  // Handle to the underlying storage buffer (used by the LinAlgBackend kernels)
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
    validate_unique_labels(std::span<TensorDim>{dims});
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

  // ----- Printers -----
  void print(std::ostream& os = std::cout) const {
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
    }

    os << ")\n";

    os << "  size: " << total_elements_ << "\n";
    os << "  allocated: " << std::boolalpha << allocated() << "\n";

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
    if (has_label(newdim.label))
      throw std::invalid_argument("Tensor::add_dim: duplicate dimension label '"
          + label_to_string(newdim.label) + "'");

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
//     Z's scalar type (the output type) -- see LinAlgBackend::gemm.
// Z's resulting shape is (X minus labelX) followed by (Y minus labelY); Z is
// allocated (and zero-initialized) if it is not already.
template <typename TX, typename TY, typename TZ, typename Exec>
void gemm(const Tensor<TX, Exec>& X,
          const Tensor<TY, Exec>& Y,
                Tensor<TZ, Exec>& Z,
          const TensorDimLabel& labelX,
          const TensorDimLabel& labelY) {
  // 1. Labels must exist
  const size_t idxX = X.label_index(labelX);
  const size_t idxY = Y.label_index(labelY);

  // 2. Must be in gemm order
  if (idxX != X.rank() - 1)
    throw std::invalid_argument("gemm: '"
        + label_to_string(labelX) + "' must be the last dimension of X");
  if (idxY != 0)
    throw std::invalid_argument("gemm: '"
        + label_to_string(labelY) + "' must be the first dimension of Y");

  // 3. Contracted dimension sizes must agree
  const size_t K = X.dims()[idxX].dim;
  if (Y.dims()[idxY].dim != K) {
    std::ostringstream oss;
    oss << "gemm: contracted dimension size mismatch (" << labelX << "=" << K
        << " vs " << labelY << "=" << Y.dims()[idxY].dim << ")";
    throw std::invalid_argument(oss.str());
  }

  // 4. Output shape: X's dims minus labelX, Y's dims minus labelY.
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

  // 5. Shape/allocate the output, then delegate the arithmetic to the (mixed-type) backend
  Z.prepare_output(result_dims, "gemm");
  LinAlgBackend<TZ, Exec>::template gemm<TX, TY>(
      X.buffer(), M, K,
      Y.buffer(), N,
      Z.buffer());
}




}
