#pragma once
#include "common.hpp"
#include "tensor_buffer.hpp"
#include "tensor_shape.hpp"
#include <vector>
#include <complex>

namespace cppgw {

// LinAlgBackend forward declaration
//
// LinAlgBackend<T, E> implements raw numerical kernels directly on TensorBuffer<T, E>.
// It knows nothing about labels or validation: all of that happens in Tensor before delegating here,
// so every method here may assume its inputs are fully sanitized
template <typename T, typename Executor>
  class LinAlgBackend;

template <typename T>
class LinAlgBackend<T, Executor::Host>  {
  using E = Executor::Host;
  public:

  static void gemm(const TensorBuffer<T, E>& A, size_t M, size_t K,
      const TensorBuffer<T, E>& B, size_t N,
      TensorBuffer<T, E>& C) {
    const T* a = A.data();
    const T* b = B.data();
    T*       c = C.data();
    for (size_t j = 0; j < N; ++j) {
      for (size_t i = 0; i < M; ++i) {
        T sum{};
        for (size_t k = 0; k < K; ++k) {
          sum += a[i + k * M] * b[k + j * K];
        }
        c[i + j * M] = sum;
      }
    }
  }
};

}
