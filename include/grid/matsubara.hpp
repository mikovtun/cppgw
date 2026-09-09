#pragma once
#include "types.hpp"
#include "grid.hpp"

#include <cmath>
#include <numbers>
#include <vector>

namespace cppgw {


// Matsubara imaginary frequency grid. Always symmetrical around 0 (the zero mode).
//
// Construction depends on:
//    halfN: Number of points on a half-grid
//    Beta: InverseTemperature
//    StatisticsTag: Fermionic (half-integer freqs) or Bosonic (integer freqs)
//
// Interface (uniform with the other grids):
//   points()          : the FULL set of Matsubara points (Bosonic includes the zero mode in the middle)
//   points_no_zero()  : the non-zero Matsubara points (Bosonic: drops the zero mode; Fermionic: == points())
//   size()            : number of points  (== points().size())
//   operator()(i)     : the i-th point of the full set (of type point_type)
template <StatisticsTag S>
class MatsubaraGrid {
public:
  using statistics_type = S;
  using point_type = ImaginaryFrequency;
  using space = ImaginaryFrequencySpace;
  using DimLabel = GridDimLabel<space>;
  inline static constexpr DimLabel dim_label{};
private:
  std::vector<point_type> points_;   // full set, ascending, zero mode (Bosonic) in the middle
  InverseTemperature      beta_;
  size_t                  halfN_;
public:
  MatsubaraGrid() = delete;

  // Build the full, ascending set of Matsubara frequencies.
  MatsubaraGrid(size_t halfN, InverseTemperature beta)
    : beta_(beta), halfN_(halfN)
  {
    if (halfN_ == 0) throw std::invalid_argument("Must have at least 1 grid point for MatsubaraGrid");
    const double fac = std::numbers::pi / static_cast<double>(beta.value);
    if constexpr (std::same_as<S, Fermionic>) {
      // Fermionic: half-integer frequencies, none are zero.  2*halfN_ points.
      points_.resize(2 * halfN_);
      for (size_t i = 0; i < halfN_; ++i) {
        const double w = fac * (2.0 * static_cast<double>(i) + 1.0);
        points_[halfN_ + i]     = point_type(w);
        points_[halfN_ - i - 1] = point_type(-w);
      }
    } else {  // Bosonic
      // Bosonic: integer frequencies including the zero mode, which sits at the middle index.
      points_.resize(2 * halfN_ + 1);
      points_[halfN_] = point_type(0.0);
      for (size_t i = 0; i < halfN_; ++i) {
        const double w = fac * (2.0 * static_cast<double>(i + 1));
        points_[halfN_ + 1 + i]   = point_type(w);
        points_[halfN_ - 1 - i]   = point_type(-w);
      }
    }
  }

  const std::vector<point_type>& points() const { return points_; }
  size_t                           size()  const { return points_.size(); }

  point_type operator()(size_t i) const {
    if (i >= points_.size())
      throw std::out_of_range("MatsubaraGrid: grid point index out of range");
    return points_[i];
  }

  // The non-zero Matsubara frequencies (for Bosonic this drops the single zero mode).
  std::vector<point_type> points_no_zero() const {
    std::vector<point_type> out;
    out.reserve(points_.size());
    for (const auto& p : points_)
      if (p.value != 0.0) out.push_back(p);
    return out;
  }
};

static_assert( Grid<MatsubaraGrid<Fermionic>>);

}
