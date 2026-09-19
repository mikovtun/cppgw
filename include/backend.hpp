#pragma once
#include "common.hpp"
#include "tensor_buffer.hpp"
#include "multiprecision.hpp"
#include <concepts>
#include <complex>
#include <cstddef>
#include <algorithm>
#include <span>
#include <vector>
#include <stdexcept>

namespace cppgw {

// ============================================================================
//  TensorBackend seam (Story 06, decision Q4)
// ----------------------------------------------------------------------------
// TensorBackend<TOut, Executor> is the required-operation *contract* a tensor
// numerical backend must provide:
//
//   * linear algebra : gemm
//   * unary ops      : fill, zero, scale, conjugate
//   * data movement  : permute (dense axis permutation)
//
// Templated over the OUTPUT scalar type TOut and the Executor, exactly as the
// former LinAlgBackend was. Kernels operate on raw TensorBuffers and know
// NOTHING about labels or symmetry (that stays in the Tensor frontend).
//
// The Tensor frontend calls *only* ops in this contract, so swapping or
// adding a backend later (BLAS / Eigen / cuTENSOR) touches no validation,
// label, or symmetry code.
//
// This story ships the host implementation only, hand-written (moved out of
// the old LinAlgBackend). BLAS/Eigen/cuTENSOR are future implementations of
// the same contract. A symmetry-aware dispatch (e.g. cuTENSOR) is explicitly
// deferred: symmetry is advisory frontend metadata, never passed here.
// ============================================================================

// Forward declare the generic (unimplemented) backend; specializations provide
// the contract.
template <typename TOut, typename Executor>
class TensorBackend;

// The required-op contract. Satisfied iff TensorBackend<T, Executor> provides
// fill/zero/scale/conjugate, dense axis permutation, and the (mixed-type) gemm
// on TensorBuffers.
template <typename T, typename Executor>
concept TensorBackendOps =
  requires(TensorBuffer<T, Executor> A, TensorBuffer<T, Executor> B, TensorBuffer<T, Executor> C) {
    { TensorBackend<T, Executor>::fill(C, T{}) };
    { TensorBackend<T, Executor>::zero(C) };
    { TensorBackend<T, Executor>::scale(C, T{1}) };
    { TensorBackend<T, Executor>::conjugate(C) };
    { TensorBackend<T, Executor>::permute(
        A, std::span<const size_t>{}, std::span<const size_t>{},
        std::span<const size_t>{}, std::span<const size_t>{}, C) };
    { TensorBackend<T, Executor>::template gemm<T, T>(A, (size_t)1, (size_t)1, B, (size_t)1, C) };
  };

// ----- Host implementation (hand-written this story; BLAS/Eigen land here later) -----
template <typename TOut>
class TensorBackend<TOut, Executor::Host> {
  using E = Executor::Host;
public:
  // ----- Linear algebra -----

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

  // ----- Dense data movement -----

  // Copy a dense tensor through an already-validated axis permutation.
  //
  // The frontend owns the labels and validates the permutation. This backend
  // sees only storage metadata:
  //   permutation[new_axis] = old_axis
  // and all strides are in the Tensor's fastest-to-slowest convention.
  // Source and destination are distinct buffers; in-place permutation is not
  // part of this contract.
  static void permute(const TensorBuffer<TOut, E>& source,
                      std::span<const size_t> source_strides,
                      std::span<const size_t> output_dims,
                      std::span<const size_t> output_strides,
                      std::span<const size_t> permutation,
                      TensorBuffer<TOut, E>& destination) {
    const size_t rank = permutation.size();
    if (source_strides.size() != rank || output_dims.size() != rank
        || output_strides.size() != rank)
      throw std::invalid_argument("TensorBackend::permute: inconsistent rank metadata");

    // A zero-sized dimension has no elements and therefore no valid data
    // pointer to dereference. The Tensor frontend still preserves its shape.
    if (destination.size() == 0)
      return;

    const TOut* src = source.data();
    TOut* dst = destination.data();
    std::vector<size_t> coordinates(rank, 0);

    // Iterate in destination storage order. Decode the current output
    // coordinate and map it back to the source coordinate using the inverse
    // direction-free convention permutation[new] = old.
    for (size_t out_offset = 0; out_offset < destination.size(); ++out_offset) {
      size_t source_offset = 0;
      for (size_t new_axis = 0; new_axis < rank; ++new_axis)
        source_offset += coordinates[new_axis] * source_strides[permutation[new_axis]];
      dst[out_offset] = src[source_offset];

      // Increment an odometer in fastest-to-slowest order. The final
      // increment is harmless because the loop terminates immediately.
      for (size_t axis = 0; axis < rank; ++axis) {
        ++coordinates[axis];
        if (coordinates[axis] < output_dims[axis])
          break;
        coordinates[axis] = 0;
      }
    }
  }

  // ----- Unary / elementwise ops -----

  // Set every element to `value`.
  static void fill(TensorBuffer<TOut, E>& T, TOut value) {
    std::fill(T.data(), T.data() + T.size(), value);
  }

  // Set every element to zero.
  static void zero(TensorBuffer<TOut, E>& T) {
    fill(T, TOut{});
  }

  // Scale every element by `alpha`.
  static void scale(TensorBuffer<TOut, E>& T, TOut alpha) {
    TOut* d = T.data();
    for (size_t i = 0; i < T.size(); ++i)
      d[i] *= alpha;
  }

  // Complex-conjugate every element. A no-op for real scalar types
  // (the codebase's floating-point scalar taxonomy decides, not a one-off check).
  static void conjugate(TensorBuffer<TOut, E>& T) {
    if constexpr (numerics::ComplexFloatingPoint<TOut>) {
      TOut* d = T.data();
      for (size_t i = 0; i < T.size(); ++i)
        d[i] = std::conj(d[i]);
    }
  }
};

} // namespace cppgw
