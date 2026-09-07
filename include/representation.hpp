#pragma once
#include "types.hpp"
#include "grid.hpp"
#include "common.hpp"
#include "tensor.hpp"

namespace cppgw {

// TensorValuedRep:
// * a Tensor for spatial data storage (matrix elements of G, elements of Chi, etc)
// * a Representation that they live in
// One or more of the Tensor dims will belong to the Representation: these can be values
// evaluated on a grid, or coefficients for a basis set expansion.
template <class R, typename scalar_type, typename exec>
concept TensorValuedRep = 
  HasFunctionSpace<R> &&
  requires(const R& x) {
    requires std::same_as<std::remove_cvref_t<decltype(x.data)>, Tensor<scalar_type, exec>>;
  };



}
