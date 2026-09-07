#pragma once
#include "types.hpp"
#include "tensor_shape.hpp"

#include <cmath>
#include <numbers>
#include <vector>
#include <stdexcept>

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

}
