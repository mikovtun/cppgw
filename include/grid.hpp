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

// Grids expose:
//   points()          : the set of grid points
//   size()            : the number of points
//   dim_label         : the tensor label to use for the grid axis
//   operator()(i)     : the i-th grid point (of type point_type)
template <typename G>
concept Grid = 
  requires(const G& g) {
    g.points();
    g.size();
    g.dim_label;
    g(0);
    requires std::same_as<decltype(g(0)), typename G::point_type>;
  };

// Quadratures additionally have weights:
//   weights()          : the sequence of (real) quadrature weights
//   weights(i)         : the weight at index i (same real type as weights()[i])
template <typename G>
concept Quadrature = 
  Grid<G> && 
  requires(const G& g) {
    g.weights();
    { g.weights(0) } -> std::convertible_to<double>;
    requires std::same_as<std::remove_cvref_t<decltype(g.weights(0))>,
                          std::remove_cvref_t<decltype(g.weights()[0])>>;
  };


template <typename G>
concept ImaginaryTimeGrid = 
  Grid<G> &&
  std::same_as<typename G::point_type, ImaginaryTime>;


}
