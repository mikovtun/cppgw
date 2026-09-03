#pragma once
#include "types.hpp"

#include <cmath>
#include <numbers>
#include <vector>

namespace cppgw {


// Grid points have a function space and a value that can be read as a scalar
template <typename T>
concept GridPoint = 
    HasFunctionSpace<T> &&
  requires(const T& x) {
  { x.value } -> std::convertible_to<double>;
  };

// Grids have points
template <typename G>
concept Grid = 
  requires(const G& g) {
    g.points();
  };

// Quadratures have points and weights
template <typename G>
concept Quadrature = 
  Grid<G> && 
  requires(const G& g) {
    g.weights();
  };



// Matsubara imaginary frequency grid. Always symmetrical around 0 (the zero mode).
// The zero mode is treated specially.
// 
// Construction depends on:
//    N: Number of points on a half-grid
//    Beta: InverseTemperature
//    StatisticsTag: Fermionic or Bosonic
//
// points(): returns all non-zero frequencies
// full_points(): returns all frequencies including zero (identical to points() for Fermionic)
template <StatisticsTag S>
class MatsubaraGrid {
  using space = ImaginaryFrequencySpace;
  using statistics_type = S;
  using point_type = ImaginaryFrequency;
private:
  std::vector<point_type> points_;
  InverseTemperature beta_;
  size_t halfN_;
  size_t N_;
public:

  // Data members
  
  const auto& points() const { return points_; }
  const auto& N() const { return N_; }

  MatsubaraGrid() = delete;
  
  // Constructor calculates grid points
  // Zero mode is omitted if bosonic: handled separately
  MatsubaraGrid(size_t halfN, InverseTemperature beta)
    : beta_(beta), halfN_(halfN)
  {
    if(halfN_ == 0) throw std::invalid_argument("Must have at least 1 grid point for MatsubaraGrid");
    N_ = 2*halfN_;

    points_.resize(N_);
    double fac = std::numbers::pi / static_cast<double>(beta.value);
    for(size_t i=0; i < halfN_; i++) {
      if constexpr (std::same_as<S, Fermionic>) {
        points_[halfN_+i]   =  fac * (2.0 * static_cast<double>(i) + 1.0);
        points_[halfN_-i-1] = -fac * (2.0 * static_cast<double>(i) + 1.0);
      }
      else if constexpr (std::same_as<S, Bosonic>) {
        points_[halfN_+i]   =  fac * (2.0 * static_cast<double>(i+1));
        points_[halfN_-i-1] = -fac * (2.0 * static_cast<double>(i+1));
      }
    }
    if constexpr (std::same_as<S, Bosonic>)
      N_++;
  }

  std::vector<point_type> full_points() const 
  {
    if constexpr (std::same_as<S, Fermionic>) return points_;
    else if constexpr (std::same_as<S, Bosonic>) {
      std::vector<point_type> points = points_;
      points.push_back(ImaginaryFrequency{0.0});
      return points;
    }
  }

};

}
