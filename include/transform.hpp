#pragma once
#include "types.hpp"
#include "grid.hpp"
#include "enums.hpp"
#include "tensor.hpp"

namespace cppgw {

// A BasisSet is either a traditional basis set (a. la. a Chebyshev expansion, or IR)
// or a Quadrature over a Grid
// This is a non-owning class and should be a concept
// Flesh this out: I want to separate the BasisSet concept out from a particular axis, and
// it should provide basis-set-specific helper functions. Transforms will use these helper functions.
// Maybe impossible to separate from FunctionSpace?


// BasisSet classes should specify a basis set expansion without owning the coefficient data
// Requires:
// A grid->coefficients function that fits or solves for expansion coefficients
// A coefficients->grid function that evaluates a set of coefficients on a grid
// Questions to be answered:
// Should these functions act on a Tensor?
// Should these functions work on both tau and imaginary time?
//template <class B>
//concept Basis

// A Representation is how temporal information about a function is stored
// The tuple (FunctionSpace, BasisSet) specifies the Representation. 
// Everything about where a function lives and how its represented


// TensorValuedRep:
// * a Tensor for spatial data storage (matrix elements of G, elements of Chi, etc)
// * a Representation that they live in
// One or more of the Tensor dims will belong to the Representation: these can be values
// evaluated on a grid, or coefficients for a basis set expansion.
template <class R, typename scalar_type, typename exec>
concept TensorValuedRep = 
  HasFunctionSpace<R> &&
  requires(const R& x) {
  { x.data } -> std::same_as<Tensor<scalar_type, exec>;
};





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


// A Transform is a class that mutates a TensorValuedRep from one Representation to another

// A Fourier transform
//template <class FT,
//          TensorValuedRep<class Rfrom,  typename scalar_type, typename exec>,
//          TensorValuedRep<class Rto,    typename scalar_type, typename exec>>
//concept FourierTransform =
//requires(Rfrom

class ChebyshevFourierTransform {
  // Inputs: A τ grid and a Matsubara iω_n grid. Templated over StatisticsTag.
  // Stores: F_{mn}  <-- Transformation matrix for the mth Chebyshev polynomial
  //                          from τ to Matsubara iω grid.
  //         The inverse transform, which is obtained by evaluating the above at the 
  //         Chebyshev nodes and using a discrete cosine transform
  //
  // Upon construction: Builds the F_{mn} matrix, which must be done in high precision
  //                    according to the paper
  // operator(TensorValuedRep<Time/Frequency FunctionSpace>): Applies the transform from time->frequency or frequency->time, dispatching according to the received FunctionSpace of the TVRep.
};


}
