#pragma once


namespace cppgw {

// A BasisSet is either a traditional basis set (a. la. a Chebyshev expansion, or IR)
// or a Quadrature over a Grid
// This is a non-owning class and should be a concept
// Flesh this out: I want to separate the BasisSet concept out from a particular axis, and
// it should provide basis-set-specific helper functions. Transforms will use these helper functions.
// Maybe impossible to separate from FunctionSpace?


// A Representation is how temporal information about a function is stored
// The tuple (FunctionSpace, BasisSet) specifies the Representation. 
// Everything about where a function lives and how its represented


// TensorValuedRep:
// * a Tensor for spatial data storage (matrix elements of G, elements of Chi, etc)
// * a Representation that they live in
// One or more of the Tensor dims will belong to the Representation: these can be values
// evaluated on a grid, or coefficients for a basis set expansion.

// A Transform is a class that mutates a TensorValuedRep from one Representation to another

}
