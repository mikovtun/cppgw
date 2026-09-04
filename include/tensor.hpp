#pragma once
#include "enums.hpp"
#include "tensor_buffer.hpp"
#include "linalgbackend.hpp"
#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <optional>
#include <span>
#include <unordered_set>
#include <iostream>
#include <typeinfo>

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


// One named axis of a Tensor
struct TensorDim {
  std::string label;
  size_t      dim = 0;
  TensorDim(std::string l_, size_t d_) : label(std::move(l_)), dim(d_) {};
  bool operator==(const TensorDim& other) const {
    return label == other.label && dim == other.dim;
  }
};




template <typename scalar_type, typename exec>
class Tensor {
  using Buffer  = TensorBuffer<scalar_type, exec>;
  using Backend = LinAlgBackend<scalar_type, exec>;
private:
  // Layout information
  std::vector<TensorDim>  dims_;
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
  

  std::optional<size_t> find_label_index(const std::string& label) const {
    for(size_t i=0; i<dims_.size(); ++i)
      if(dims_[i].label == label)
        return i;
    return std::nullopt;
  }
  
  static void validate_unique_labels(const std::span<TensorDim>& dims) {
    std::unordered_set<std::string> seen;
    for (const auto& d : dims) {
      if (!seen.insert(d.label).second)
        throw std::invalid_argument("Tensor: duplicate dimension label '" + d.label + "'");
    }
  }

  void validate_unique_labels() const { validate_unique_labels(dims_); }

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
  Tensor(std::initializer_list<TensorDim> inputDims) : dims_(inputDims) { compute_strides(); }

  // ----- Layout -----
  size_t total_elements() const { return total_elements_; }
  size_t rank() const { return dims_.size(); }
  const std::vector<TensorDim>& dims() const { return dims_; }
  const std::vector<size_t>& strides() const { return strides_; }

  bool has_label(const std::string& label) const {
    return find_label_index(label).has_value();
  }

  size_t label_index(const std::string& label) const {
    auto idx = find_label_index(label);
    if (!idx) {
      throw std::invalid_argument("Tensor: label '" + label + "' not found.");
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

      os << dims_[i].label << "=" << dims_[i].dim;
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

  // ----- Operations -----
  
  // Contract the last dimension of X (named 'labelX') against the first dimension of Y (named 'labelY') and write the result into Z
  // X, Y, and Z must be on the same Executor (compile-time checked)
  static void gemm(const Tensor& X, const Tensor& Y, Tensor& Z,
                    const std::string& labelX, const std::string& labelY) {
    // 1. Labels must exist
    const size_t idxX = X.label_index(labelX);
    const size_t idxY = Y.label_index(labelY);

    // 2. Must be in gemm order
    if (idxX != X.rank() - 1)
      throw std::invalid_argument(
          "Tensor::gemm: '" + labelX + "' must be the last dimension of X");
    if (idxY != 0)
      throw std::invalid_argument(
          "Tensor::gemm: '" + labelY + "' must be the first dimension of Y");

    // 3. Contracted dimension sizes must agree
    const size_t K = X.dims_[idxX].dim;
    if (Y.dims_[idxY].dim != K) {
      std::ostringstream oss;
      oss << "Tensor::gemm: contracted dimension size mismatch (" << labelX << "=" << K
        << " vs " << labelY << "=" << Y.dims_[idxY].dim << ")";
      throw std::invalid_argument(oss.str());
    }

    // 4. Output shape: X's dims minus labelX, Y's dims minus labelY.
    // Collect product of X's surviving dims = M
    // Collect product of Y's surviving dims = N
    std::vector<TensorDim> result_dims;
    result_dims.reserve(X.rank() - 1 + Y.rank() - 1);
    size_t M = 1;
    for (size_t i=0; i<X.rank()-1; ++i) {
      result_dims.push_back(X.dims_[i]);
      M *= X.dims_[i].dim;
    }
    size_t N = 1;
    for (size_t i=1; i<Y.rank(); ++i) {
      result_dims.push_back(Y.dims_[i]);
      N *= Y.dims_[i].dim;
    }
    // Guard against X and Y shaing a surviving label
    validate_unique_labels(result_dims);

    // 4. Allocate or validate Z
    // Throw if Z is already allocated and has unexpected dims
    if (!Z.allocated()) {
      Z.dims_ = std::move(result_dims);
      Z.compute_strides();
      Z.allocate();
    } else {
      if(Z.dims_.size() != result_dims.size()) 
        throw std::invalid_argument("Tensor::gemm: output has wrong rank");
      for (size_t i=0; i < result_dims.size(); ++i) {
        if (Z.dims_[i] != result_dims[i])
          throw std::invalid_argument("Tensor::gemm: output dim '" + 
              Z.dims_[i].label + "' is incompatible with expected '" + 
              result_dims[i].label + "'");
      }
    }

    // Hand off to backend
    Backend::gemm(*X.buffer_, M, K, *Y.buffer_, N, *Z.buffer_);
  }
  

};




}
