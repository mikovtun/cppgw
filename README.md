# cppgw

Welcome to `cppgw`, a little project of mine that makes absolutely no guarantees of being useful or correct (although thats the goal!). 
I'm using this project to become more fluent in C++20 and AI-assisted development. 
It's early in development and incomplete, so it's a bit messy and everything is subject to change.
Here's a very AI-flavored README that does a decent job of explaining the motivation of the project and its goals:


`cppgw` is a C++20 electronic-structure code, in the spirit of the GW/GF2
family of many-body approximations, built around a simple idea: **the
equations are the program.**

The code is designed so that the mathematical and physical relationships
between quantities (Green's functions, self-energies, Hamiltonians, basis
tensors) are expressed in the *types*, not left as bookkeeping a user has to
keep straight and correct. When a calculation is written against these types,
many of the ways a numerical code can silently go wrong are impossible to
write — not by linting or convention, but because the relationship is encoded
in the interface the compiler enforces.

The goal is a Green's-function code that is:

* **legible as physics** — the code reads like the formula it implements,
* **hard to get wrong** — invalid combinations of quantities and operations
  are rejected at compile time rather than discovered at runtime,
* **portable across hardware and precision** — the same source describes Host
  or Device execution and real, complex, or arbitrary-precision arithmetic,

and yet small enough to be understood end to end.

---

## Core tenets

### The math is in the types

Physical objects are first-class, distinct types: an AO core Hamiltonian,
a density-fitting interaction tensor, a Matsubara-grid Green's function, a
self-energy in imaginary time are not interchangeable "arrays." The type
system records *what a thing is* — its indices, its function space (real
space, Matsubara frequency, imaginary time), and which statistics it obeys
(fermionic vs. bosonic) — so that a transform, contraction, or solve only
type-checks when it makes sense for the quantities involved.

Where a relationship is genuinely a law (symmetry of an interaction tensor,
the statistics of a Green's function, the pairing of indices in a
contraction), that law is attached to the object and honoured by the
operations, so the user neither has to remember it nor can casually violate
it.

### Tensors are abstracted over linear algebra

Numerical work on tensors — matrix products, linear-system solves, axis
permutations — is presented through a thin, portable interface. The code is
written in terms of *what* a computation does on a labelled tensor, not
*how* a particular library lays out memory or dispatches kernels. This keeps
the mathematical intent visible and separable from:

* **the linear-algebra backend** — dense kernels can be implemented by hand,
  or later delegated to BLAS, a tensor library, or a GPU, without rewriting
  the physics, and
* **hardware residency** — tensors carry an execution tag (Host vs. Device)
  as part of their type, so the same code expresses both on today's machines
  (Host) and on accelerators (Device) in future, without changing how the
  calculations are written.

The point is not to build a tensor library for its own sake: it is to make
the *relationship* between physical quantities and the operations on them
the unit of design, and to keep the machinery underneath interchangeable.

### Precision and hardware are parameters, not forks

Because scalar types are drawn from a small taxonomy of (real and complex)
floating-point types — native `double`/`long double`, `std::complex<double>`,
and arbitrary-precision types from Boost — the same templates and tests work
across precisions. This matters for a code that trades convergence, round-off,
and conditioning for accuracy: it is a design value, not a convenience.

---

## What it computes

Today the code implements the first, non-self-consistent step of a GF2
calculation: from an AO core Hamiltonian `H_0`, overlap `S`, density matrix
`P`, and a density-fitting tensor `B`, it forms the static Hartree–Fock
self-energy and the initial Matsubara Green's function

```
G_0(iω_n) = [(iω_n + μ)S - (H_0 + Σ_HF[P])]⁻¹
```

solving the linear system at each frequency rather than inverting. The
surrounding machinery — imaginary-time and Matsubara grids, forward and
tail-corrected Fourier transforms, labelled symmetric tensors, and the
linear-algebra abstractions above — is in place to support the fuller,
self-consistent GF2 and GW calculations that follow.

---

## Working with it

The code is developed **story by story**: each numbered file in
[`stories/`](stories/) is a design brief with a "definition of done," and the
implementation in `include/` and `src/` tracks it. This makes the design
intent and its history easy to trace.

### Build

Requires a C++20 compiler, CMake ≥ 3.20, a BLAS/LAPACK, and Boost. HighFive
(HDF5) is fetched automatically.

```
cmake -S . -B build && cmake --build build
```

### Run

```
./cppgw -h                            help
./cppgw -i INPUT -d DATA              run the calculation named in INPUT
./cppgw -t                            run the built-in test suite
```

`INPUT` is a small text file of `KEYWORD = VALUE` lines selecting the
calculation and the datasets to read from the HDF5 `DATA` file; results are
written to an HDF5 output file. See `tools/rhf_df.in` for a sample input.
