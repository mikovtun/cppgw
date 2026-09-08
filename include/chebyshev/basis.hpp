#pragma once
#include <numbers>
#include <vector>
#include <span>
#include <concepts>
#include <iterator>
#include <cmath>
#include <type_traits>

#include "common.hpp"
#include "grid.hpp"
#include "../grid/matsubara.hpp"
#include "../tensor.hpp"
#include "multiprecision.hpp"

namespace cppgw {
// Make the numeric concepts visible in the cppgw namespace
using numerics::RealFloatingPoint;
using numerics::FloatingPoint;


// C is a range whose element is a real floating-point type
// (native std::floating_point or a boost multiprecision real).
template <class C>
concept RealFloatRange = RealFloatingPoint<Elem<C>>;

// The numerical implementation of the Chebyshev basis:
// Provides routines for evaluating Chebyshev polynomials and for
// solving for chebyshev coefficients in arbitrary precision.
//
// API contract:
//   * inputs  -> any contiguous range of real floats (std::vector, std::span, ...)
//   * returns -> scalar T, or std::vector<T>
template <RealFloatingPoint T>
class ChebyshevBasisImpl {
private:
  size_t order_;
public:
  ChebyshevBasisImpl() = delete;
  explicit ChebyshevBasisImpl(size_t order) : order_(order) {}

  size_t order() const noexcept { return order_; }
  size_t size() const noexcept { return order_ + 1; }

  // Chebyshev nodes:
  // x_j = cos(pi*j/N), j = 0,...,N
  std::vector<T> nodes() const {
    std::vector<T> x(size());
    if (order_ == 0) {
      x[0] = T(1);
      return x;
    }
    for(size_t j=0; j<size(); ++j) {
      const double arg = std::numbers::pi * static_cast<double>(j) / static_cast<double>(order_);
      x[j] = T(std::cos(arg));
    }
    return x;
  }

  // Evaluate the Chebyshev polynomial T_n(x) using three-term recurrence
  static T polynomial(size_t n, T x) {
    if (n==0)
      return T(1);
    if (n==1)
      return x;
    T prev2 = T(1);
    T prev1 = x;
    T tcurr {};
    for (size_t k=2; k<=n; ++k) {
      tcurr = T(2) * x * prev1 - prev2;
      prev2 = prev1;
      prev1 = tcurr;
    }
    return prev1;
  }

  // Evaluate a coefficient vector at a single sample point x (Clenshaw's
  // recurrence). coeffs[k] is the coefficient of T_k; the degree evaluated is
  // min(order_, coeffs.size()-1), so a short coefficient vector is safely
  // clamped instead of reading out of bounds.
  template <class Coeffs, class U>
  requires RealFloatRange<Coeffs> && RealFloatingPoint<U>
  T evaluate(const Coeffs& coeffs, U x) const {
    const size_t total = static_cast<size_t>(coeffs.size());
    if (total < 2) return T(0);
    const size_t n = (order_ < total - 1) ? order_ : (total - 1);  // top degree used
    T bK1{0};   // b_{k+1}
    T bK2{0};   // b_{k+2}
    for(size_t i=0; i<n; ++i) {
      const size_t k = n - i;          // k runs n, n-1, ..., 1
      const T bK = T(coeffs[k]) + T(2) * T(x) * bK1 - bK2;
      bK2 = bK1;
      bK1 = bK;
    }
    return T(coeffs[0]) + T(x) * bK1 - bK2;
  }

  // Evaluate a coefficient vector at many sample points x.
  template <class Coeffs, class Xs>
  requires RealFloatRange<Coeffs> && RealFloatRange<Xs>
  std::vector<T> evaluate(const Coeffs& coeffs, const Xs& x) const {
    std::vector<T> out(x.size());
    for(size_t i=0; i<x.size(); ++i)
      out[i] = evaluate(coeffs, x[i]);
    return out;
  }

};


// Expansion types: (such as ChebyshevExpansion)
// 1. Own a coefficient expansion on a FunctionSpace
// Transformation classes handle transforming between ChebyshevExpansionTau and ChebyshevExpansionMatsubara via Fourier transform
// 2. Have a function to produce/update coefficients from Grid data on that function space
// 3. Have a function to produce/update Grid data from coefficients
//      * The above are Transformations, and Transformations should be data-owning classes
//
// Constructors: I want to be able to build these from...
// 1. a properly formatted tensor (i.e. with special tensor indices)
// 2. a spatial tensor shape: append special tensor index and allocate
// 3. a spatial tensor: append special tensor index and optionally copy
//
// Extras:
// It would be convenient to have an operator(i) that accesses the spatial tensor at the special index i


// The τ-space Chebyshev tensor expansion
template <FloatingPoint data_type, RealFloatingPoint cheb_impl_type = double>
class ChebyshevExpansionTau {
public:
  using DimLabel = ChebyshevExpansionDimLabel<ImaginaryTimeSpace>;
  inline static constexpr DimLabel dim_label{};
  using space = ImaginaryTimeSpace;
private:
  ChebyshevBasisImpl<cheb_impl_type> impl_;
  Tensor<data_type, Executor::Host> data_;
public:
  // Delete default constructor
  ChebyshevExpansionTau() = delete;
  // Constructor: Take Tensor shape and add one dim for coefficients
  explicit ChebyshevExpansionTau(TensorShape spatial_tensor_shape, size_t order) 
    : impl_(order) {
    // Throw if there are no dims in the spatial_tensor_shape: need at least a scalar
    if (spatial_tensor_shape.size() == 0)
      throw std::invalid_argument("ChebyshevExpansionTau: Need a tensor of nonzero rank");
    size_t coeffsize = impl_.size();

    // Add coefficient dimension as fastest
    spatial_tensor_shape.insert_fast(TensorDim{dim_label, coeffsize});
    data_ = Tensor<data_type, Executor::Host>(spatial_tensor_shape);
  }

  // Constructor: Take Tensor and add one dim for coefficients
  explicit ChebyshevExpansionTau(Tensor<data_type, Executor::Host>& spatial_tensor, size_t order) 
    : impl_(order) {
    // Throw if there are no dims in the spatial_tensor_shape: need at least a scalar
    if (spatial_tensor.rank() == 0)
      throw std::invalid_argument("ChebyshevExpansionTau: Need a tensor of nonzero rank");
    size_t coeffsize = impl_.size();

    // Add coefficient dimension as fastest
    data_ = std::move(spatial_tensor);
    data_.add_dim({dim_label, coeffsize}, /* slow */ false, /* copy */ true);
  }

  Tensor<data_type, Executor::Host>& data() { return data_; }
};



// The Matsubara-space Chebyshev tensor expansion
template <StatisticsTag S, FloatingPoint data_type, RealFloatingPoint cheb_impl_type = double>
class ChebyshevExpansionMatsubara {
public:
  using DimLabel = ChebyshevExpansionDimLabel<ImaginaryFrequencySpace>;
  inline static constexpr DimLabel dim_label{};
  using space = ImaginaryFrequencySpace;
private:
  ChebyshevBasisImpl<cheb_impl_type> impl_;
  Tensor<data_type, Executor::Host> data_;
public:
  // Constructor: Take Tensor shape and add one dim for coefficients
  explicit ChebyshevExpansionMatsubara(TensorShape spatial_tensor_shape, size_t order) 
    : impl_(order) {
    // Throw if there are no dims in the spatial_tensor_shape: need at least a scalar
    if (spatial_tensor_shape.size() == 0)
      throw std::invalid_argument("ChebyshevExpansionMatsubara: Need a tensor of nonzero rank");
    size_t coeffsize = impl_.size();

    // Add coefficient dimension as fastest
    spatial_tensor_shape.insert_fast(TensorDim{"cheb_coeff", coeffsize});
    data_ = Tensor<data_type, Executor::Host>(spatial_tensor_shape);
  }

  // Constructor: Take Tensor and add one dim for coefficients
  explicit ChebyshevExpansionMatsubara(Tensor<data_type, Executor::Host>& spatial_tensor, size_t order) 
    : impl_(order) {
    // Throw if there are no dims in the spatial_tensor_shape: need at least a scalar
    if (spatial_tensor.rank() == 0)
      throw std::invalid_argument("ChebyshevExpansionMatsubara: Need a tensor of nonzero rank");
    size_t coeffsize = impl_.size();

    // Add coefficient dimension as fastest
    data_ = std::move(spatial_tensor);
    data_.add_dim({"cheb_coeff", coeffsize}, /* slow */ false, /* copy */ true);
  }

  Tensor<data_type, Executor::Host>& data() { return data_; }

};

}
