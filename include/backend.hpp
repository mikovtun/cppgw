#pragma once
#include "common.hpp"
#include "tensor_buffer.hpp"
#include "multiprecision.hpp"
#include "types.hpp"
#include <concepts>
#include <complex>
#include <cstddef>
#include <algorithm>
#include <numeric>
#include <span>
#include <vector>
#include <stdexcept>
#include <limits>
#include <sstream>
#include <type_traits>

namespace cppgw {

namespace detail {

// Minimal scalar-operation abstraction used by backend code. These are
// intentionally unqualified after importing std overloads so ADL can find Boost
// multiprecision real/complex functions too.
template <typename T>
auto numeric_abs(const T& x) {
  using std::abs;
  return abs(x);
}

template <typename T>
T numeric_conj(const T& x) {
  if constexpr (numerics::ComplexFloatingPoint<T>) {
    using std::conj;
    return conj(x);
  } else {
    return x;
  }
}

template <typename R>
R numeric_sqrt(const R& x) {
  using std::sqrt;
  return sqrt(x);
}

// LAPACK details live in src/lapack_solve.cpp. Keep the public header free of
// Fortran ABI declarations; the template backend calls only these C++ wrappers.
void lapack_gesv_in_place(TensorBuffer<double, Executor::Host>& coefficient,
                          size_t n,
                          TensorBuffer<double, Executor::Host>& rhs,
                          size_t nrhs);
void lapack_gesv_in_place(TensorBuffer<std::complex<double>, Executor::Host>& coefficient,
                          size_t n,
                          TensorBuffer<std::complex<double>, Executor::Host>& rhs,
                          size_t nrhs);

template <numerics::FloatingPoint T>
void qr_solve_in_place(TensorBuffer<T, Executor::Host>& coefficient, size_t n,
                       TensorBuffer<T, Executor::Host>& rhs, size_t nrhs) {
  using R = decltype(numeric_abs(T{}));
  T* a = coefficient.data();
  T* b = rhs.data();

  R max_col_norm = R(0);
  for (size_t j = 0; j < n; ++j) {
    R norm2 = R(0);
    for (size_t i = 0; i < n; ++i) {
      const R q = numeric_abs(a[i + n * j]);
      norm2 += q * q;
    }
    const R norm = numeric_sqrt(norm2);
    if (norm > max_col_norm) max_col_norm = norm;
  }
  const R scale = max_col_norm > R(1) ? max_col_norm : R(1);
  const R tol = R(64) * std::numeric_limits<R>::epsilon()
              * R(n == 0 ? 1 : n) * scale;

  std::vector<size_t> permutation(n);
  std::iota(permutation.begin(), permutation.end(), size_t(0));

  // Column-pivoted Householder QR: A P = Q R.  Reflectors are applied
  // immediately to all remaining columns and RHSs, so only R is retained.
  for (size_t k = 0; k < n; ++k) {
    size_t pivot = k;
    R pivot_norm2 = R(-1);
    for (size_t j = k; j < n; ++j) {
      R norm2 = R(0);
      for (size_t i = k; i < n; ++i) {
        const R q = numeric_abs(a[i + n * j]);
        norm2 += q * q;
      }
      if (norm2 > pivot_norm2) {
        pivot_norm2 = norm2;
        pivot = j;
      }
    }
    const R xnorm = numeric_sqrt(pivot_norm2);
    if (xnorm <= tol) {
      std::ostringstream oss;
      oss << "TensorBackend::solve_in_place QR fallback: numerically singular "
          << "matrix at rank " << k << " (threshold " << tol << ")";
      warn_if(1, oss.str());
      throw std::runtime_error(oss.str());
    }
    if (pivot != k) {
      for (size_t i = 0; i < n; ++i)
        std::swap(a[i + n * k], a[i + n * pivot]);
      std::swap(permutation[k], permutation[pivot]);
    }

    const T x0 = a[k + n * k];
    const R abs_x0 = numeric_abs(x0);
    const T phase = abs_x0 == R(0) ? T(1) : x0 / abs_x0;
    const T alpha = -phase * xnorm;
    std::vector<T> v(n - k);
    for (size_t i = k; i < n; ++i)
      v[i - k] = a[i + n * k];
    v[0] -= alpha;
    R vnorm2 = R(0);
    for (const T& value : v) {
      const R q = numeric_abs(value);
      vnorm2 += q * q;
    }
    if (vnorm2 <= R(0)) {
      const std::string msg = "TensorBackend::solve_in_place QR fallback: degenerate Householder reflector";
      warn_if(1, msg);
      throw std::runtime_error(msg);
    }
    const T beta = T(2) / vnorm2;

    auto apply_reflector = [&](T* column) {
      T dot{};
      for (size_t i = 0; i < v.size(); ++i)
        dot += numeric_conj(v[i]) * column[k + i];
      const T factor = beta * dot;
      for (size_t i = 0; i < v.size(); ++i)
        column[k + i] -= v[i] * factor;
    };
    for (size_t j = k; j < n; ++j)
      apply_reflector(a + n * j);
    for (size_t j = 0; j < nrhs; ++j)
      apply_reflector(b + n * j);

    // Make the retained R explicitly triangular; this also prevents harmless
    // roundoff remnants below the diagonal from confusing later inspection.
    a[k + n * k] = alpha;
    for (size_t i = k + 1; i < n; ++i)
      a[i + n * k] = T{};
  }

  // Solve R y = Q^H B, then undo A P = Q R: x = P y.
  for (size_t j = 0; j < nrhs; ++j) {
    for (size_t kk = n; kk-- > 0;) {
      T sum = b[kk + n * j];
      for (size_t l = kk + 1; l < n; ++l)
        sum -= a[kk + n * l] * b[l + n * j];
      const R diagonal = numeric_abs(a[kk + n * kk]);
      if (diagonal <= tol) {
        std::ostringstream oss;
        oss << "TensorBackend::solve_in_place QR fallback: numerically singular "
            << "triangular factor at pivot " << kk;
        warn_if(1, oss.str());
        throw std::runtime_error(oss.str());
      }
      b[kk + n * j] = sum / a[kk + n * kk];
    }
  }
  std::vector<T> solution(n * nrhs);
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t k = 0; k < n; ++k)
      solution[permutation[k] + n * j] = b[k + n * j];
  std::copy(solution.begin(), solution.end(), b);
}

} // namespace detail

// ============================================================================
//  TensorBackend seam (Story 06, decision Q4)
// ----------------------------------------------------------------------------
// TensorBackend<TOut, Executor> is the required-operation *contract* a tensor
// numerical backend must provide:
//
//   * linear algebra : gemm, solve_in_place (dense A X = B)
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
// fill/zero/scale/conjugate, dense axis permutation, the in-place dense solve,
// and the (mixed-type) gemm on TensorBuffers.
template <typename T, typename Executor>
concept TensorBackendOps =
  numerics::FloatingPoint<T> &&
  requires(TensorBuffer<T, Executor> A, TensorBuffer<T, Executor> B, TensorBuffer<T, Executor> C) {
    { TensorBackend<T, Executor>::fill(C, T{}) };
    { TensorBackend<T, Executor>::zero(C) };
    { TensorBackend<T, Executor>::scale(C, T{1}) };
    { TensorBackend<T, Executor>::conjugate(C) };
    { TensorBackend<T, Executor>::permute(
        A, std::span<const size_t>{}, std::span<const size_t>{},
        std::span<const size_t>{}, std::span<const size_t>{}, C) };
    { TensorBackend<T, Executor>::template gemm<T, T>(A, (size_t)1, (size_t)1, B, (size_t)1, C) };
    { TensorBackend<T, Executor>::solve_in_place(A, (size_t)1, B, (size_t)1) };
  };

// ----- Host implementation (hand-written this story; BLAS/Eigen land here later) -----
template <numerics::FloatingPoint TOut>
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

  // ----- Dense linear solve -----

  // Overwrite B with X satisfying A X = B.  Both dense buffers use the
  // Tensor/LAPACK-compatible column-major layout (row fastest).  This raw
  // backend entry point deliberately knows nothing about Tensor dimensions or
  // labels; the Tensor frontend validates and batches arbitrary-rank operands.
  static void solve_in_place(TensorBuffer<TOut, E>& coefficient, size_t n,
                             TensorBuffer<TOut, E>& rhs, size_t nrhs) {
    if (n == 0 || nrhs == 0)
      throw std::invalid_argument("TensorBackend::solve_in_place: n and nrhs must be positive");
    if (n > std::numeric_limits<size_t>::max() / n
        || coefficient.size() != n * n)
      throw std::invalid_argument("TensorBackend::solve_in_place: coefficient buffer has wrong size");
    if (n > std::numeric_limits<size_t>::max() / nrhs
        || rhs.size() != n * nrhs)
      throw std::invalid_argument("TensorBackend::solve_in_place: RHS buffer has wrong size");

    if constexpr (std::same_as<TOut, double> || std::same_as<TOut, std::complex<double>>) {
      detail::lapack_gesv_in_place(coefficient, n, rhs, nrhs);
    } else {
      detail::qr_solve_in_place(coefficient, n, rhs, nrhs);
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
        d[i] = detail::numeric_conj(d[i]);
    }
  }
};

} // namespace cppgw
