#pragma once
#include "common.hpp"
#include "tensor_buffer.hpp"
#include <vector>
#include <complex>
#include <cstddef>

namespace cppgw {

// LinAlgBackend<TOut, Executor> implements raw numerical kernels directly on
// TensorBuffer objects.
//
//   * The first template parameter TOut is the OUTPUT (result) scalar type.
//   * Each kernel takes its input buffers as (template) type parameters TA, TB,
//     which may differ from TOut and from each other. The arithmetic is carried
//     out in TOut (the correct output type), so mixed-type operations are
//     supported, e.g.  RealFloatingPoint x ComplexFloatingPoint -> ComplexFloatingPoint.
//
// It knows nothing about labels or validation: all of that happens in the Tensor
// frontend before delegating here, so every method here may assume its inputs
// are fully sanitized.
template <typename TOut, typename Executor>
class LinAlgBackend;

// ----- Host implementation -----
template <typename TOut>
class LinAlgBackend<TOut, Executor::Host> {
  using E = Executor::Host;
public:
  // Matrix product   C[i, j] = sum_k A[i, k] * B[k, j]
  //   A : M x K, row-major (i fastest)  ->  a[i + k*M]
  //   B : K x N, row-major (k fastest)  ->  b[k + j*K]
  //   C : M x N, row-major (i fastest)  ->  c[i + j*M]
  // A (type TA), B (type TB), C (type TOut) may have different scalar types; each
  // operand is promoted to TOut before multiplication, so the whole accumulation
  // happens in the output type.
  template <typename TA, typename TB>
  static void gemm(const TensorBuffer<TA, E>& A, size_t M, size_t K,
                   const TensorBuffer<TB, E>& B, size_t N,
                   TensorBuffer<TOut, E>& C) {
    const TA*  a = A.data();
    const TB*  b = B.data();
    TOut*      c = C.data();
    for (size_t j = 0; j < N; ++j)
      for (size_t i = 0; i < M; ++i) {
        TOut sum{};
        for (size_t k = 0; k < K; ++k)
          sum += TOut(a[i + k * M]) * TOut(b[k + j * K]);
        c[i + j * M] = sum;
      }
  }
};

}
