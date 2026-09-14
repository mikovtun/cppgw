#pragma once
#include "types.hpp"
#include "symmetry.hpp"
#include <concepts>
#include <ostream>
#include <string>
#include <ranges>
#include <type_traits>
#include <variant>
#include <vector>
#include <utility>

namespace cppgw {
// ----- Special label tags for coefficient expansions and grid reps-----
template <ImagFunctionSpaceTag S> 
struct ChebyshevExpansionDimLabel   { using space = S;
  bool operator==(const ChebyshevExpansionDimLabel&) const = default; };
template <FunctionSpaceTag S> 
struct GridDimLabel   { using space = S;
  bool operator==(const GridDimLabel&) const = default; };

using TensorDimLabel = std::variant<
  ChebyshevExpansionDimLabel<ImaginaryTimeSpace>,
  ChebyshevExpansionDimLabel<ImaginaryFrequencySpace>,
  GridDimLabel<ImaginaryTimeSpace>,
  GridDimLabel<ImaginaryFrequencySpace>,
  GridDimLabel<RealTimeSpace>,
  GridDimLabel<RealFrequencySpace>,
  std::string
  >;


// ----- Label helpers ----- (variant-vs-string compares, printing, hashing)
template <class T> struct IsChebLabel : std::false_type {};
template <class S> struct IsChebLabel<ChebyshevExpansionDimLabel<S>> : std::true_type {};
template <class T> struct IsGridLabel : std::false_type {};
template <class S> struct IsGridLabel<GridDimLabel<S>> : std::true_type {};

inline const char* space_name(ImaginaryTimeSpace)     { return "imag-tau"; }
inline const char* space_name(ImaginaryFrequencySpace){ return "imag-freq"; }
inline const char* space_name(RealTimeSpace)          { return "real-t"; }
inline const char* space_name(RealFrequencySpace)     { return "real-freq"; }

// Readable rendering of any label; unique per tag type
inline std::string label_to_string(const TensorDimLabel& l) {
  return std::visit([](const auto& x) -> std::string {
    using X = std::decay_t<decltype(x)>;
    if constexpr (std::same_as<X, std::string>)
      return x;
    else if constexpr (IsChebLabel<X>::value)
      return std::string("cheb:") + space_name(typename X::space{});
    else
      return std::string("grid:") + space_name(typename X::space{});
  }, l);
}

// variant-vs-string equality (variant == string only works if every
// alternative is comparable to string, so compare through a variant)
inline bool label_equals(const TensorDimLabel& l, const std::string& s) {
  return l == TensorDimLabel(s);
}

// Stable hash so TensorDimLabel works in std::unordered_set/map
struct TensorDimLabelHash {
  size_t operator()(const TensorDimLabel& l) const {
    return std::hash<std::string>{}(label_to_string(l));
  }
};

inline std::ostream& operator<<(std::ostream& os, const TensorDimLabel& l) {
  return os << label_to_string(l);
}


// A single named axis of a Tensor
struct TensorDim {
  TensorDimLabel label;
  size_t      dim = 0;
  SymGroup    symmetry = nullptr;   // Story 06: null => plain axis (the old, common case);
                                     //            non-null => member of this symmetry family.
  TensorDim(TensorDimLabel l_, size_t d_, SymGroup s_ = nullptr)
      : label(std::move(l_)), dim(d_), symmetry(std::move(s_)) {}
  bool operator==(const TensorDim& other) const {
    // Now also compares the SymGroup handle ("same group" == same object).
    return label == other.label && dim == other.dim && symmetry == other.symmetry;
  }
};

// A range of TensorDims makes a TensorShape
// * Instantiable from any container (or span) of TensorDims
template <typename T>
concept TensorShapeLike = 
  std::ranges::range<T> &&
  std::same_as<std::ranges::range_value_t<T>, TensorDim>;

// TODO: make this satisfy the above!
struct TensorShape {
  std::vector<TensorDim> dims;
  template<TensorShapeLike R>
    explicit TensorShape(R&& d) : dims(std::ranges::begin(d), std::ranges::end(d)) {}
  TensorShape(std::initializer_list<TensorDim> d) : dims(d) {}

  void insert_fast(const TensorDim d) {
    dims.insert(dims.begin(), d);
  }
  void insert_slow(const TensorDim d) {
    dims.push_back(d);
  }
  size_t size() const { return dims.size(); }

  auto begin() { return dims.begin(); }
  auto end() { return dims.end(); }
};


}
