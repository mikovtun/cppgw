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

  // Gauss–Legendre nodes x and weights W on (-1,1), ascending order
  // (Newton–Raphson, A&S 25.2.17 / Numerical Recipes `gaussl`)
  static void legendre(size_t n, std::vector<double>& x, std::vector<double>& w) {
    x.resize(n);
    w.resize(n);
    const long m = static_cast<long>((n + 1) / 2);
    const double EPS = std::numeric_limits<double>::epsilon();
    for (long i = 1; i <= m; ++i) {
      double z  = std::cos(std::numbers::pi * (2.0 * i - 1.0) / (2.0 * n + 2.0));
      double z1 = z;
      double pp = 1.0;
      do {
        double p1 = 1.0, p2 = 0.0;
        for (long j = 1; j <= static_cast<long>(n); ++j) {
          double p3 = p2;
          p2 = p1;
          p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
        }
        pp = static_cast<double>(n) * (z * p1 - p2) / (z * z - 1.0);
        z1 = z;
        z -= p1 / pp;
      } while (std::fabs(z - z1) > 16.0 * EPS);
      x[static_cast<size_t>(i - 1)] = -z;
      x[n - i]                      =  z;
      const double wi = 2.0 / ((1.0 - z * z) * pp * pp);
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
