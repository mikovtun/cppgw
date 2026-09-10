#pragma once
#include "grid.hpp"
#include "types.hpp"
#include "tensor.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace cppgw {
/*
 * This file specifies grids on the ImaginaryTime axis, which are all bounded by [0,β].
 * These include:
 * 1. Uniform grid
 * 2. Gauss-legendre grid
 * 3. Chebyshev node grid
 *
 * These grids all must expose interfaces to the ImaginaryTime type.
 * Since these grids carry temporal information, they get a DimLabel for tensor storage
 */
  
// ---------------------------------------------------------------------------
// UniformImaginaryTimeGrid
//   n equally spaced half-centered points  τ_i = (i + 1/2)·β/n  ∈ (0, β)
//   Trapezoidal weights  w_i = β/n
// ---------------------------------------------------------------------------
class UniformImaginaryTimeGrid {
public:
  using point_type = ImaginaryTime;
  using space = ImaginaryTimeSpace;
  using DimLabel = GridDimLabel<space>;
  inline static constexpr DimLabel dim_label{};
private:
  std::vector<point_type>  points_;
  std::vector<double>         weights_;
  InverseTemperature          beta_;
public:
  UniformImaginaryTimeGrid(InverseTemperature b, size_t n): beta_(b) {
    if (n == 0)
      throw std::invalid_argument("UniformImaginaryTimeGrid: need at least 1 grid point");
    const double h = b.value / static_cast<double>(n);
    points_.resize(n);
    weights_.assign(n, h);
    for (size_t i = 0; i < n; ++i)
      points_[i] = point_type((static_cast<double>(i) + 0.5) * h);
  }
  UniformImaginaryTimeGrid() = delete;

  size_t size() const { return points_.size(); }
  InverseTemperature beta() const { return beta_; }

  // Indexed access to a grid point
  point_type operator()(size_t i) const {
    if (i >= points_.size())
      throw std::out_of_range("UniformImaginaryTimeGrid: index out of range");
    return points_[i];
  }

  const std::vector<point_type>& points()  const { return points_; }
  const std::vector<double>&        weights() const { return weights_; }
  const double        weights(size_t i) const { return weights_[i]; }
};

// ---------------------------------------------------------------------------
// GaussLegendreImaginaryTimeGrid
//   n Gauss–Legendre nodes mapped from (-1,1) onto (0, β):
//     τ_i = (β/2)(1 + x_i),   w_i = (β/2) · W_i
//   Exact integration of polynomials up to degree 2n-1 on [0, β]
// ---------------------------------------------------------------------------
class GaussLegendreImaginaryTimeGrid {
public:
  using point_type = ImaginaryTime;
  using space = ImaginaryTimeSpace;
  using DimLabel = GridDimLabel<space>;
  inline static constexpr DimLabel dim_label{};
private:
  std::vector<point_type>  points_;
  std::vector<double>         weights_;
  InverseTemperature          beta_;

  // Gauss–Legendre nodes x and weights W on (-1,1), ascending order.
  //
  // Root-finding with a bracketing safeguard: the root x_i = cos(i·π/(n+1)) of
  // P_n is strictly bracketed between the classic NR starting points, in the
  // sense that, for i < m,
  //     cos(π(2i+1)/(2n+2))  <  x_i  <  cos(π(2i−1)/(2n+2))
  // and the smallest positive root x_m is bracketed by (0, cos(π(2m−1)/(2n+2))).
  // (For odd n the middle root x_m = 0 exactly.)
  //
  // The naive A&S/Numerical-Recipes Newton (no bracket) is NOT robust here:
  // for even n, P_n'(0) = 0, so near the middle roots the Newton step
  // |P_n/P_n'| is huge, the iterate crosses the origin, and some i's converge
  // to the WRONG root (observed: n=16, i=8 converging to −x_6 instead of x_8),
  // silently corrupting both nodes and weights. We therefore keep Newton as
  // long as the candidate stays inside the bracket and bisect otherwise;
  // the root is always found and no root crossing is possible. (The
  // starting points are never roots, and all roots of P_n are simple, so the
  // sign-based bracket update is well defined.)
  static void legendre(size_t n, std::vector<double>& x, std::vector<double>& w) {
    x.resize(n);
    w.resize(n);
    const long m = static_cast<long>((n + 1) / 2);
    const double EPS = std::numeric_limits<double>::epsilon();
    const double Pi  = std::numbers::pi;

    // P_n(z) and P'_n(z) by the standard three-term recurrence (A&S 25.2.17).
    // (Same P_n as before; kept as a helper so both are available.)
    auto pn = [n](double z, double& f, double& df) {
      double p1 = 1.0, p2 = 0.0;
      for (long j = 1; j <= n; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
      }
      df = static_cast<double>(n) * (z * p1 - p2) / (z * z - 1.0);
      f  = p1;
    };

    for (long i = 1; i <= m; ++i) {
      double z;
      if ((n % 2 == 1) && (i == m)) {
        z = 0.0;                                  // exact middle root (odd n)
      } else {
        const double lo = ((i == m) ? 0.0
                      : std::cos(Pi * (2.0 * i + 1.0) / (2.0 * n + 2.0)));
        const double hi = std::cos(Pi * (2.0 * i - 1.0) / (2.0 * n + 2.0));

        // Invariant: the unique root x_i lies in (a, b);  fa = P_n(a) ≠ 0.
        double a  = lo, b = hi;
        double ffa, dfa;
        pn(a, ffa, dfa);
        double fa = ffa;

        double zt = 0.5 * (a + b);
        for (int it = 0; it < 80 && (b - a) > 16.0 * EPS; ++it) {
          double f, df;
          pn(zt, f, df);
          const double zn = (df == 0.0) ? 0.5 * (a + b) : zt - f / df;
          // Shrink the bracket to the side that still contains the root.
          if ((fa > 0.0 && f > 0.0) || (fa < 0.0 && f < 0.0)) { a = zt; fa = f; }
          else                                                  { b = zt; }
          // Newton step if it respects the bracket, otherwise bisect.
          zt = (zn > a && zn < b) ? zn : 0.5 * (a + b);
        }
        z = 0.5 * (a + b);
        { // one final Newton polish (quadratic) plus refreshed P'_n(z)
          double f, df;
          pn(z, f, df);
          if (df != 0.0) z -= f / df;
          pn(z, f, df);
        }
      }

      // Gauss weight:  W_i = 2 / ( (1 - x_i^2) * P'_n(x_i)^2 ).
      double f_dummy, dfz;
      pn(z, f_dummy, dfz);
      const double wi = 2.0 / ((1.0 - z * z) * dfz * dfz);

      x[static_cast<size_t>(i - 1)] = -z;
      x[n - i]                      =  z;
      w[static_cast<size_t>(i - 1)] = wi;
      w[n - i]                      = wi;
    }
  }
public:
  GaussLegendreImaginaryTimeGrid(InverseTemperature b, size_t n): beta_(b) {
    if (n == 0)
      throw std::invalid_argument("GaussLegendreImaginaryTimeGrid: need at least 1 grid point");
    std::vector<double> x, w;
    legendre(n, x, w);
    points_.resize(n);
    weights_.resize(n);
    for (size_t i = 0; i < n; ++i) {
      points_[i]    = point_type(b.value * 0.5 * (1.0 + x[i]));
      weights_[i] = b.value * 0.5 * w[i];
    }
  }
  GaussLegendreImaginaryTimeGrid() = delete;

  size_t size() const { return points_.size(); }
  InverseTemperature beta() const { return beta_; }

  point_type operator()(size_t i) const {
    if (i >= points_.size())
      throw std::out_of_range("GaussLegendreImaginaryTimeGrid: index out of range");
    return points_[i];
  }

  const std::vector<point_type>& points()  const { return points_; }
  const std::vector<double>&        weights() const { return weights_; }
  const double        weights(size_t i) const { return weights_[i]; }
};

// ---------------------------------------------------------------------------
// ChebyshevNodeImaginaryTimeGrid
//   First-kind Chebyshev nodes, the same convention as ChebyshevBasisImpl:
//     x_j = cos(π j / N),  j = 0..N   (N = polynomial order, N+1 nodes)
//   mapped onto [0, β] by  τ_j = (β/2)(1 + x_j)
//   No weights: integration over this grid is done via the Chebyshev DCT,
//   hence it satisfies the Grid (but not Quadrature) concept.
// ---------------------------------------------------------------------------
class ChebyshevNodeImaginaryTimeGrid {
public:
  using point_type = ImaginaryTime;
  using space = ImaginaryTimeSpace;
  using DimLabel = GridDimLabel<space>;
  inline static constexpr DimLabel dim_label{};
private:
  std::vector<point_type> points_;
  InverseTemperature         beta_;
  size_t                     order_;
public:
  // order = highest Chebyshev polynomial order (N); yields N + 1 nodes
  ChebyshevNodeImaginaryTimeGrid(InverseTemperature b, size_t order): beta_(b), order_(order) {
    const size_t N = order_;
    points_.resize(N + 1);
    for (size_t j = 0; j <= N; ++j) {
      const double x = (N == 0) ? 1.0
          : std::cos(std::numbers::pi * static_cast<double>(j) / static_cast<double>(N));
      points_[j] = point_type(b.value * 0.5 * (1.0 + x));
    }
  }
  ChebyshevNodeImaginaryTimeGrid() = delete;

  size_t size() const { return points_.size(); }
  size_t order() const { return order_; }
  InverseTemperature beta() const { return beta_; }

  point_type operator()(size_t i) const {
    if (i >= points_.size())
      throw std::out_of_range("ChebyshevNodeImaginaryTimeGrid: index out of range");
    return points_[i];
  }

  const std::vector<point_type>& points() const { return points_; }
};

// ---------------------------------------------------------------------------
// Concept conformance
// ---------------------------------------------------------------------------
static_assert( HasFunctionSpace<UniformImaginaryTimeGrid>);
static_assert( HasFunctionSpace<GaussLegendreImaginaryTimeGrid>);
static_assert( HasFunctionSpace<ChebyshevNodeImaginaryTimeGrid>);
static_assert( Quadrature<UniformImaginaryTimeGrid>);
static_assert( Quadrature<GaussLegendreImaginaryTimeGrid>);
static_assert( Grid<ChebyshevNodeImaginaryTimeGrid>);

static_assert( ImaginaryTimeGrid<UniformImaginaryTimeGrid>);
static_assert( ImaginaryTimeGrid<ChebyshevNodeImaginaryTimeGrid>);
static_assert( ImaginaryTimeGrid<GaussLegendreImaginaryTimeGrid>);

}
