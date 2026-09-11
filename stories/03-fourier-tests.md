# Story — Fourier transform convergence tests

Implement analytic convergence tests for the Tau-space → Matsubara-space Fourier transform.

## Motivation

The most useful validation of the Fourier transform is comparison against functions with known analytic transforms.

The tests should demonstrate that:

1. The numerical Fourier transform agrees with the analytic result.
2. The error decreases as the Tau and Matsubara grids are refined.
3. A sufficiently dense grid produces a result very close to the analytic solution.

These tests should exercise the existing `FourierTransform` implementation rather than testing its internal implementation details.

Before implementing the tests, inspect the existing FourierTransform, GridExpansionTau, and MatsubaraGrid implementations and explicitly determine the mathematical convention they implement. Compare that convention with the three analytic expressions above. If there is a discrepancy, report it before changing code.


## Analytic test functions

Implement the following analytic functions and their corresponding Fourier transforms.

Use the same Fourier-transform convention currently implemented by `FourierTransform`. The expressions below define the expected results for that convention.

### 1. Constant function

Tau-space:

```text
f(τ) = 1
```

Analytic Matsubara-space transform:

```text
F(iω_n) = -2 / (iω_n)
```

### 2. Exponential function

For a real parameter `a`:

```text
f(τ) = exp(-a τ)
```

Analytic transform:

```text
F(iω_n) = [exp(β(-a + iω_n)) - 1] / (iω_n - a)
```

Use a small, positive value of `a`.

### 3. One-particle Green's function

For a small, positive real orbital energy `ε`, define the fermionic occupation number:

```text
nF(ε) = 1 / (exp(β ε) + 1)
```

and Tau-space Green's function:

```text
g(τ) = -(1 - nF(ε)) exp(-ε τ)
```

with analytic Matsubara-space Green's function:

```text
G(iω_n) = 1 / (iω_n - ε)
```

Use a small positive value of `ε`.

## Test implementation

Create a new source file dedicated to Fourier-transform tests.

The analytic functions should be implemented as test fixtures/helpers in this source file rather than added to the production API.

For each test function:

1. Construct scalar-valued `GridExpansionTau` objects on several Tau grids.
2. Construct corresponding Matsubara grids with several grid sizes.
3. Evaluate the analytic function on the Tau grid.
4. Apply the existing `FourierTransform` class.
5. Compare the numerical Matsubara coefficients/values against the corresponding analytic result.

The tests should use multiple values of `β` so that the transform is tested over different inverse-temperature domains.

## Grid refinement

For each function, test several increasing grid sizes.

The exact grid sizes and number of refinement levels can be chosen based on the existing grid implementations, but there should be enough levels to clearly demonstrate convergence.

The tests should include:

* several relatively coarse grids;
* intermediate grid sizes;
* a very dense Tau grid;
* a very dense Matsubara grid.

The dense-grid case should produce numerical results that are close to the analytic result to a suitably tight tolerance.

## Convergence criterion

Do not require the numerical result to exactly equal the analytic result at finite grid size.

Instead, verify that refinement improves the result.

For each test case, calculate an appropriate error between the numerical transform and the analytic transform, for example:

```text
error = |F_numerical(iω_n) - F_exact(iω_n)|
```

or an appropriate norm over the tested Matsubara points.

Verify that the error decreases as the grid is refined, subject to the expected numerical behavior of the discretization.

The test should not rely on exact equality of floating-point values.

For the densest grid, additionally require the error to be below a reasonable absolute/relative tolerance.

Choose tolerances based on the actual numerical convergence of the existing implementation rather than making them unnecessarily strict.

## Matsubara frequencies

The analytic expressions must be evaluated at the actual Matsubara frequencies represented by the target Matsubara grid.

Respect the statistics and frequency convention of the existing `MatsubaraGrid` implementation. In particular, ensure that the Green's-function test uses fermionic Matsubara frequencies.

Do not hard-code an assumed frequency indexing convention if the existing grid API already provides the frequencies.

## Test-suite integration

Add the new tests to the test suite introduced in Story 02.

The Fourier-transform tests should be executed when the existing test-suite entry point is invoked.

The tests should fail if:

* the numerical transform is substantially inconsistent with the analytic result;
* refinement does not produce convergence;
* the dense-grid result is not sufficiently close to the analytic solution.

## Scope

This story is strictly a validation story.

Do not modify the Fourier-transform algorithm merely to make these tests pass.

If the tests reveal an apparent discrepancy between the analytic expressions and the existing Fourier-transform convention, stop and investigate the convention rather than silently changing either the implementation or the analytic reference.

Do not add the analytic test functions to the production library unless required by the existing testing architecture.

## Grid coverage (extended requirement)

The Fourier-transform tests must cover **all usable source (tau) grids**, i.e. all
grids that satisfy the `Quadrature` concept required by `FourierTransform`:

* `UniformImaginaryTimeGrid` (midpoint rule) — O(n^-2) quadrature error;
  the convergence test asserts a **strictly decreasing** error at every
  refinement level (n = 16, 32, 64, 128, 256) plus a dense case below
  1e-4 absolute max error.
* `GaussLegendreImaginaryTimeGrid` — spectrally accurate; the error is at the
  ~1e-12..1e-15 machine floor already at n = 16, so the expected
  "refinement" behavior is asserted as **near-machine precision at every
  level** (bound 1e-9) plus the dense case below 1e-4.

`ChebyshevNodeImaginaryTimeGrid` exposes no quadrature weights (it is a
`Grid`, not a `Quadrature`; its integration path is the Chebyshev/DCT one),
so it is **not usable with `FourierTransform` and is out of scope** for this
story.

## Defect found and fixed (for the record)

Adding the Gauss–Legendre coverage exposed a real defect **in the grid,
not the Fourier transform**: the grid's naive A&S/Numerical-Recipes Newton
root-finder (for even-degree `P_n`, where `P_n'(0) = 0`) let the iterate
cross the origin near the middle roots and converge to the **wrong root**
(e.g. n = 16, i = 8 converged to −x_6 instead of x_8), silently producing
duplicate nodes (~0.542 / 1.458) and mismatched weights.

Fix: `include/grid/tau.hpp` — `GaussLegendreImaginaryTimeGrid::legendre`
now uses a **bracketed Newton** (bisection fallback on the bracket
`cos(π(2i+1)/(2n+2)) < x_i < cos(π(2i−1)/(2n+2))`), so no root crossing is
possible and the exact middle root (odd n) / smallest positive root (even n)
is handled. Verified against an independent reference (numpy `leggauss`):
sum of weights = β, and identical quadrature results to ~1e-15 for all test
functions at every n.

`FourierTransform` itself was **not modified** — it correctly applies any
supplied quadrature rule (validated by the Uniform-grid cases against the
analytic closed forms and cross-checked independently). The grid fix is a
genuine bug fix discovered by these tests, in line with the story's
instruction to report and investigate rather than silently change either
side.

## Definition of done

* [x] A dedicated source file contains the Fourier-transform tests and analytic reference functions.
* [x] The constant-function transform is tested.
* [x] The exponential-function transform is tested.
* [x] The one-particle Green's-function transform is tested.
* [x] Tests cover multiple `β` values.
* [x] Tests cover multiple Tau-grid sizes.
* [x] Tests cover multiple Matsubara-grid sizes.
* [x] Scalar-valued `GridExpansionTau` objects are used.
* [x] The existing `FourierTransform` class is used to perform the numerical transforms.
* [x] Numerical results are compared against analytic results.
* [x] Convergence with increasing grid size is tested.
* [x] A dense-grid case is required to be sufficiently close to the analytic solution.
* [x] The tests are integrated into the Story 02 test suite.
* [x] The complete test suite passes.
* [x] **All usable Quadrature tau grids are covered** (Uniform midpoint + Gauss–Legendre);
      ChebyshevNode excluded (not a `Quadrature`, not usable with `FourierTransform`).
* [x] The Fourier-transform implementation is unmodified; the Gauss–Legendre
      grid root-convergence defect discovered by these tests was fixed in
      `include/grid/tau.hpp` (bracketed Newton) and documented above.

## Status

**Done.** All of the above acceptance criteria are met; `./cppgw -t` (and the
`-test` alias) run the full suite and pass: 9 Uniform-grid cases (strict
`O(n^-2)` convergence to dense-error ~2e-6..2e-5 < 1e-4) and 9
Gauss–Legendre cases (near machine precision, ≤ ~2.3e-11, at every node count
from n = 16 to the dense n = 4096 case). Invalid invocations and the remaining
Story 02 invocation matrix are unaffected and still behave as specified.


