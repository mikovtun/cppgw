#pragma once

#include "chebyshev/basis.hpp"


// ------------------------------------------------
// This file specifies the TensorValuedRepresentation for the Chebyshev basis
// ------------------------------------------------

namespace cppgw {

  // Specification for a basis expansion in Chebyshev polynomials
  // Chebyshev bases are good for finite domains: [0, β]
  template <ImaginaryFunctionSpace F, typename scalar_type, Executor exec>
    class ChebyshevRepresentation {
      using space = F;
      public:
      size_t size = 5;   // Highest order of polynomials to expand in
      Tensor<scalar_type, exec> data;

      // Constructor should take:
      // 1. A span of TensorDims that specify the spatial components of the data
      // 2. The order(size) of the expansion, which will be added as an additional dimension for to the tensor

    };


}
