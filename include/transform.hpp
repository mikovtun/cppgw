#pragma once
#include "types.hpp"
#include "grid.hpp"
#include "common.hpp"
#include "tensor.hpp"

namespace cppgw {

// A Transform is a class that mutates a TensorValuedRep from one Representation to another
// Examples include:
// * Fourier transforms between ImaginaryTime (tau) and ImaginaryFrequency (Matsubara)
// * Analytic continuations from ImaginaryFrequency to RealFrequency (Nevanlinna)
// * Basis evaluations: going from a basis expansion (coefficients) representation to a grid representation
// * Basis constructions: going from a grid representation to a basis expansion (coefficients) representation

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
