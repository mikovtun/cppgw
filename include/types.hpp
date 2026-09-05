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

template <class F>
concept ImagFunctionSpaceTag = 
  std::same_as<F, ImaginaryTimeSpace> ||
  std::same_as<F, ImaginaryFrequencySpace>;

template <class F>
concept RealFunctionSpaceTag = 
  std::same_as<F, RealTimeSpace> ||
  std::same_as<F, RealFrequencySpace>;

template <class T>
concept HasFunctionSpace =
  requires {
    typename T::space;
  } && FunctionSpaceTag<typename T::space>;

// These all satisfy the GridPoint concept, defined in grid.hpp
// (except InverseTemperature)
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
