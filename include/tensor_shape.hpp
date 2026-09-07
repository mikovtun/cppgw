#pragma once
#include <concepts>
#include <string>
#include <ranges>

namespace cppgw {

// A single named axis of a Tensor
struct TensorDim {
  std::string label;
  size_t      dim = 0;
  TensorDim(std::string l_, size_t d_) : label(std::move(l_)), dim(d_) {};
  bool operator==(const TensorDim& other) const {
    return label == other.label && dim == other.dim;
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
