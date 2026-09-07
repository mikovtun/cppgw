#pragma once
#include <numbers>

#include "grid.hpp"
#include "enums.hpp"
#include "tensor.hpp"

#include "multiprecision.hpp"

namespace cppgw {
// The numerical implementation of the Chebyshev basis:
// Provides routines for evaluating and solving for chebyshev coefficients in arbitrary precision
template <RealFloatingPoint T>
class ChebyshevBasisImpl {
private:
  size_t order_;
public:
  ChebyshevBasisImpl() = delete;
  explicit ChebyshevBasisImpl(size_t order) : order_(order) {};
  
  size_t order() const noexcept { return order_; }
  size_t size() const noexcept { return order_ + 1; }
  
  // Chebyshev nodes:
  // x_j = cos(pi*j/N), j = 0,...,N
  std::vector<T> nodes() const {
    std::vector<T> x(size());
    const T p = p<T>();

    if (order_ == 0)
      return std::vector({T(1)});
    else
      for(size_t j=0; j<size(); ++j)
        x[j] = cos(p * static_cast<T>(j) / static_cast<T>(order_));
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
      tCurr = T(2) * x * tPrev1 - tPrev2;
      tPrev2 = tPrev1;
      tPrev1 = tCurr;
    }
    return tPrev1;
  }
  
  // Evaluate a coefficient vector at a sample point
  template <RealFloatingPoint U>
  T evaluate(std::span<const U> coeffs, U x) const {
    const size_t n = order_;
    T bK1{0};   // b_{k+1}
    T bK2{0};   // b_{k+2}
    for(size_t i=0; i<n; ++i) {
      const size_t k = n - i;
      const T bK = T(coeffs[k]) + T(2) * T(x) * bK1 - bK2;
      bK2 = bK1;
      bK1 = bK;
    }
    return coefficients[0] + x * bK1 - bK2;
  }

  // Evaluate a coefficient vector at many sample points
  template <RealFloatingPoint U>
  std::vector<T> evaluate(std::span<const U> coeffs, std::span<const U> x) const {
    std::vector<T> out(x.size());
    for(size_t i=0; i<x.size(); ++i)
      out[i] = evaluate<U>(coeffs, x[i]);
    return out;
  }



};




}
