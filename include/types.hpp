#pragma once

#include <complex>
#include <concepts>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <string>

namespace cppgw {


// -------------------------------------------------------------------------
//  Statistics tags
// -------------------------------------------------------------------------
struct Fermionic {};
struct Bosonic {};

template <class Statistics>
concept StatisticsTag = 
  std::same_as<Statistics, Fermionic> ||
  std::same_as<Statistics, Bosonic>;


// -------------------------------------------------------------------------
// Function space tags + points in them
// -------------------------------------------------------------------------

// Which mathematical space a representation lives in
struct ImaginaryTimeSpace  {};
struct ImaginaryFrequencySpace      {};
struct RealFrequencySpace  {};
struct RealTimeSpace       {};

template <class F>
concept FunctionSpaceTag = 
  std::same_as<F, ImaginaryTimeSpace> ||
  std::same_as<F, ImaginaryFrequencySpace> ||
  std::same_as<F, RealFrequencySpace> ||
  std::same_as<F, RealTimeSpace>;

template <class T>
concept HasFunctionSpace =
  requires {
    typename T::space;
  } && FunctionSpaceTag<typename T::space>;


// Tau 
struct ImaginaryTime {
  using space = ImaginaryTimeSpace;
  double value {};
  constexpr explicit ImaginaryTime(double x) : value(x) {}
  constexpr bool operator==(const ImaginaryTime& o) const noexcept { return value == o.value; }
};

// iω
struct ImaginaryFrequency {
  using space = ImaginaryFrequencySpace;
  double value {};

  constexpr ImaginaryFrequency(double x) : value(x) {}
  constexpr ImaginaryFrequency() noexcept = default;
  constexpr bool operator==(const ImaginaryFrequency& o) const noexcept { return value == o.value; }
};

// t
struct RealTime {
  using space = RealTimeSpace;
  double value {};

  constexpr explicit RealTime(double x) : value(x) {}
  constexpr bool operator==(const ImaginaryFrequency& o) const noexcept { return value == o.value; }
};

// ω
struct RealFrequency {
  using space = RealFrequencySpace;
  double value {};

  constexpr explicit RealFrequency(double x) : value(x) {}
  constexpr bool operator==(const ImaginaryFrequency& o) const noexcept { return value == o.value; }
};

// Beta
struct InverseTemperature {
  double value {};
  constexpr InverseTemperature(double x) : value(x) {}
  constexpr InverseTemperature() noexcept = default;
  constexpr bool operator==(const InverseTemperature& o) const noexcept { return value == o.value; }
};

// -------------------------------------------------------------------------
// Grids:
// Owns a vector of points that live in a function space
// -------------------------------------------------------------------------





// ===========================================================================
//  Matsubara index algebra
//
//  The self-energy / response kernels are convolutions *over Matsubara index*:
//      Σ_m = ½ Σ_n G(n) G(n−m) P(n−m)      ...
//  Two facts force the shape of this API (see AGENTS.md #14):
//
//    F1. the index is a SIGNED integer — n−m is negative for m>n, so the index
//        must live in ℤ, not size_t;
//    F2. a difference of two same-statistics points is OFFSET-FREE:
//            ω_n − ω_{n−m} = n·Δω − (n−m)·Δω = m·Δω     (Δω = 2π/β),
//        so the *polarizability* kernel P is a function of the offset-free shift.
//
//  We therefore model two distinct quantities that share one physical value type
//  (`ImaginaryFrequency`) but differ in index semantics (offset vs none) — the
//  arithmetic below enforces the distinction (a point can never be added to a
//  point; only a point and a shift compose).
// ===========================================================================

// A Matsubara frequency *point*:  ω_n = offset(S) + n·Δω,  with signed n ∈ ℤ.
template<StatisticsTag S>
struct MatsubaraFrequency {
  using space = ImaginaryFrequencySpace;
  using index_type = std::int64_t;

  index_type         n;                          // signed Matsubara index (n ∈ ℤ)
  InverseTemperature beta;

  constexpr MatsubaraFrequency(index_type n_, InverseTemperature beta_)
    : n(n_), beta(beta_) {}
  MatsubaraFrequency() = delete;

  // Δω = 2π/β — the spacing (raw double), statistics-independent.
  [[nodiscard]] constexpr double delta_value()  const noexcept { return 2.0 * std::numbers::pi / beta.value; }
  // offset(S) = π/β (Fermi) or 0 (Bose), in raw doubles — the statistics-dependent part.
  [[nodiscard]] constexpr double offset_value() const noexcept {
    if constexpr (std::same_as<S, Fermionic>) return std::numbers::pi / beta.value;
    else                                      return 0.0;
  }
  // The point as a TYPED imaginary frequency:  ω_n = offset + n·Δω.
  // (Read the raw double as `.omega().value`.)
  [[nodiscard]] constexpr ImaginaryFrequency omega() const noexcept {
    return ImaginaryFrequency{offset_value() + static_cast<double>(n) * delta_value()};
  }

  constexpr index_type index() const noexcept { return n; }
  constexpr bool operator==(const MatsubaraFrequency<S>& o) const noexcept { return n == o.n && beta == o.beta; }
};

// A Matsubara *shift* (frequency difference):  δ_m = m·Δω,  with signed m ∈ ℤ.
//  Offset-FREE, and deliberately statistics-INDEPENDENT, so one MatsubaraShift
//  composes with points of *either* Fermionic or Bosonic statistics (the
//  polarizability kernel P is a function of the shift, not of the statistics).
struct MatsubaraShift {
  using index_type = std::int64_t;

  index_type         m;                          // signed, offset-free index (m ∈ ℤ)
  InverseTemperature beta;

  constexpr MatsubaraShift(index_type m_, InverseTemperature beta_)
    : m(m_), beta(beta_) {}
  MatsubaraShift() = delete;

  // Δω = 2π/β — the spacing (raw double), statistics-independent.
  [[nodiscard]] constexpr double delta_value() const noexcept { return 2.0 * std::numbers::pi / beta.value; }
  // The shift as a TYPED imaginary frequency:  δ_m = m·Δω   (no offset).
  // (Read the raw double as `.omega().value`.)
  [[nodiscard]] constexpr ImaginaryFrequency omega() const noexcept {
    return ImaginaryFrequency{static_cast<double>(m) * delta_value()};
  }

  constexpr index_type index() const noexcept { return m; }
  constexpr bool operator==(const MatsubaraShift& o) const noexcept { return m == o.m && beta == o.beta; }
};

// ---- index algebra --------------------------------------------------------
//   point ± shift  ->  point   ;   shift ± shift  ->  shift
//   point − point  ->  shift   (F2: the offset cancels in a difference)
template<StatisticsTag S>
constexpr MatsubaraFrequency<S> operator+(MatsubaraFrequency<S> p, const MatsubaraShift& s) {
  p.n += s.m; return p;
}
template<StatisticsTag S>
constexpr MatsubaraFrequency<S> operator-(MatsubaraFrequency<S> p, const MatsubaraShift& s) {
  p.n -= s.m; return p;
}
template<StatisticsTag S>
constexpr MatsubaraShift operator-(const MatsubaraFrequency<S>& a, const MatsubaraFrequency<S>& b) {
  return MatsubaraShift(a.n - b.n, a.beta);
}
constexpr MatsubaraShift operator+(const MatsubaraShift& a, const MatsubaraShift& b) {
  return MatsubaraShift(a.m + b.m, a.beta);
}
constexpr MatsubaraShift operator-(const MatsubaraShift& a, const MatsubaraShift& b) {
  return MatsubaraShift(a.m - b.m, a.beta);
}


// -------------------------------------------------------------------------
//  Verbosity: a single namespace-wide integer setting (see /btw)
//
//    0  (default)  -- silent (no side-channel output)
//    1+            -- emit informational warnings on std::cerr
//
//  Transforms / algorithms call `warn_if(level, fmt, args...)` which is a
//  **no-op** when `verbosity < level`. This is the hook for the "M > N case
//  is a projection, not an inverse" warning and similar.
// -------------------------------------------------------------------------
inline int  g_verbosity_level = 0;
inline void set_verbosity(int lvl) { g_verbosity_level = lvl; }
inline int  verbosity()         { return g_verbosity_level; }

// Emit a warning on stderr only if verbosity is >= required_level.
inline void warn_if(int required_level, const std::string& msg) {
  if (g_verbosity_level >= required_level) {
    std::cerr << "[cppgw] " << msg << std::endl;
  }
}

}
