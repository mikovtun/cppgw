#pragma once
#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <span>

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

enum class Executor { Host, Device };

// forward declare Buffer
//template <Executor E>
//  class TensorBuffer;

struct TensorDim {
  std::string label = "";
  size_t      dim = 0;
  TensorDim(std::string l_, size_t d_) : label(l_), dim(d_) {};
};


template <typename scalar_type, Executor exec>
class Tensor {
private:
  // Layout information
  std::vector<TensorDim>  dims_;
  std::vector<size_t>     strides_;
  size_t                  total_elements_ = 0;
  
  // Storage
  //std::shared_ptr<TensorBuffer<exec>> buffer_;
public:

  // Calculates strides based off of dims
  void compute_strides() {
    size_t stride = 1;
    strides_.resize(dims_.size());
    for (size_t i=0; i<dims_.size(); ++i) {
      strides_[i] = stride;
      stride *= dims_[i].dim;   // Accumulate stride for next dim
    }
    total_elements_ = stride;
  }

  size_t total_elements() const {
    return total_elements_;
  }

  // Accept any container of TensorDim
  template <typename R>
  explicit Tensor(R&& inputDims) 
    : dims_(std::begin(inputDims), std::end(inputDims)) { compute_strides(); }

  // Accept brace-enclosed initializer lists
  Tensor(std::initializer_list<TensorDim> inputDims) : dims_(inputDims) { compute_strides(); }

};

}
