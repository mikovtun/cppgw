# Story 04 — Inverse Fourier transform (Matsubara → τ): statistics, tail, and polynomial fit

## Background

This story implements the Matsubara-to-τ direction of the Fourier transform.

So far, the Tensor-valued-representation and FourierTransform machinery has been
statistically agnostic: any tensor could be expanded on the τ-interval [0,β] and
transformed to Matsubara frequencies. The existing forward transform
(`include/grid/fourier.hpp`, convention `F(iω_m) = ∫₀^β e^{iω_m τ} f(τ) dτ`)
works unchanged for both statistics, because the Matsubara *grid* already carries
the statistics: `MatsubaraGrid<Fermionic>` gives half-integer
`(±1, ±3, …)π/β` frequencies and `MatsubaraGrid<Bosonic>` gives integer
`0, ±2, ±4, …` π/β frequencies.

The inverse direction, Matsubara back to τ, is where statistics and boundary
behavior become important.

Fermionic and bosonic tensors obey different conditions on the endpoints of the
[0,β] interval:

```text
Fermionic:  f(τ + β) = -f(τ)
Bosonic:    f(τ + β) =  f(τ)
```

Bosonic tensors can be transformed back using the obvious inverse:

```text
G(τ) = 1/β sum_n e^{-i ω_n τ} G(i ω_n)
```

Fermionic tensors are different. Most Green's functions of physical interest have
a discontinuity at one or both endpoints (e.g. the one-particle GF has
`G(0+) ≠ G(β-)`), and that endpoint structure produces a slowly decaying,
high-frequency expansion. A direct truncated inverse Matsubara transform therefore
converges poorly and produces large errors near the imaginary-time boundaries
(Gibbs-like behavior, error ~1/ω_max instead of spectral decay). For example,
the one-level GF `G(iω_n) = 1/(iω_n - ε)` has the expansion

```text
G(i ω_n) = 1/(i ω_n) + ε/(i ω_n)^2 + ε^2/(i ω_n)^3 + ...
```

The 1/ω term alone yields a O(1) truncation error after summing over ±N
frequencies — precisely the high-frequency component that must be removed before
the finite sum.

### The standard fix: high-frequency "tail" subtraction

Subtract a *known analytic* high-frequency expansion before transforming, and add
it back in τ-space where it has a closed form:

```text
G(τ) = 1/β sum_n e^{-i ω_n τ} (G(i ω_n) - G_tail(i ω_n)) + G_tail(τ)
```

where `G_tail` is the high-frequency expansion:

```text
G_tail(i ω_n) = c_1/(i ω_n) + c_2/(i ω_n)^2 + c_3/(i ω_n)^3 + ...
```

The coefficients are determined by the endpoint derivatives of G. For the
forward convention implemented in this code (`+iωτ` in the kernel), the
relationship is

```text
c_{k+1} = (-1)^{k+1} ( G^(k)(0+) + G^(k)(β-) )
```

where `G^(k)` is the k-th derivative, and the one-sided limits are the values
from *inside* [0,β] at 0 and β respectively (no assumption about any relation
between the two limits — the function may genuinely jump at the boundary).

The corresponding powers of `1/(iω_n)` have a known analytic inverse: their τ-space
counterparts `T_k(τ) = (1/β) Σ_n e^{-iω_n τ} (iω_n)^{-k}` are polynomials in β
and τ (Euler–Bernoulli polynomials of order k−1). The first six are:

```text
T_1(τ) = -1/2
T_2(τ) = -1/4*β + 1/2*τ
T_3(τ) = 1/2(1/2*βτ - 1/2*τ^2)
T_4(τ) = 1/6(1/8*β^3 - 3/4*βτ^2 + 1/2*τ^3)
T_5(τ) = 1/24(-1/2*β^3τ + βτ^3 - 1/2*τ^4)
T_6(τ) = 1/120(-1/4*β^5 + 5/4*β^3τ^2 - 5/4*βτ^4 + 1/2*τ^5)
```

so the corresponding function in τ-space is:

```text
G_tail(τ) = sum_k c_k * T_k(τ)
```

These six `T_k` formulas are correct under this code's Fourier convention and can
be used verbatim (a test will verify them against the explicit Matsubara sums
numerically).

A tail of order `P` therefore stores the coefficients

```text
c_1, ..., c_P
```

and requires the endpoint derivatives of orders `0, ..., P-1`.

(Reference note: the original draft of this story said a 6th-order expansion
requires the 7th derivative. That is an off-by-one. A tail of order `P`
requires the derivatives of orders `0, …, P-1`. In particular, a 6th-order
expansion — `1/(iω) + … + 1/(iω)^6` — needs derivatives of orders 0 through 5,
*not* the 7th.)

### Two independent polynomial fits (left and right boundary)

To extract `c_1, …, c_P` from a tensor given on an arbitrary τ-grid, we need the
one-sided derivatives `G^(k)(0+)` and `G^(k)(β-)`.

**This story will use two *independent* local polynomial fits — one near
τ=0⁺, one near τ=β⁻ — and *not* a single polynomial fitted across the wrapped
boundary.** A single fit straddling the join would implicitly assume the
function is continuous and smooth at the boundary, which is exactly the
assumption that fails for the Green's functions we care about (a fermionic GF is
antiperiodic, not continuous, at its endpoints).

Concretely, for the left boundary we take the `M = M_left` nearest points with
smallest positive τ, and for the right boundary the `M = M_right` nearest points
with largest τ. On each domain we form the Lagrange interpolation polynomial
`L_σ(τ)` of degree `M-1` through those points and read off the derivatives
`L_σ^{(k)}` at the endpoint — this is exact for polynomials of degree `< M`,
which makes the estimator directly testable on exact polynomial data.

Design choices carried into this story:

1. **One-sided stencils, not wrapped.** Each boundary is fitted on its own
   neighborhood. The two domains are disjoint sets of grid points. `M_left` and
   `M_right` are independent configuration parameters.

2. **M (stencil size) and P (tail order) are independent.** The polynomial fit
   must be of degree at least `P-1` on each side to carry orders up to `P`; a good
   default is `M ≥ P`, but the parameters are separate. Start with `M ≈ 8` for
   both, as the story originally specified, and allow the caller to choose
   different values for the two boundaries.

3. **Degree equals the number of stencil points minus one**, i.e. the classical
   Lagrange interpolation polynomial through `M` points. This is exactly a
   polynomial of degree `M-1`, and its `k`-th derivative at a point (the endpoint,
   here) can be read off analytically from the Lagrange basis.

4. **The machinery is grid-agnostic but must work correctly for the supported
   grids of this code** (uniform, Gauss–Legendre, Chebyshev-node τ-grids),
   including nonuniform ones. On a non-uniform grid the Lagrange basis
   polynomials are no longer equally spaced but the endpoint-derivative formula
   is the same: it is the sum, over the `i`-th basis polynomial `ℓ_i(τ)` of the
   stencil, of the value `f(τ_i)·(k-th derivative of ℓ_i at the endpoint)`. The
   derivatives of the basis polynomials are themselves computed analytically
   (they are polynomials, so their derivatives are polynomials).

5. **Numerical precision.** The coefficients `c_k` are small combinations of the
   endpoint derivatives, but for large `P` or very close grid points the
   Lagrange basis derivatives grow factorially (`~M! / distances^k`). In the
   intended regime (`P` small, `M` modest, reasonable β/τ scale) `double` is
   sufficient. When very high `P` is needed the implementation can route through
   the `multiprecision` machinery already in the code base, or a higher-precision
   `Tensor` scalar — the API should not preclude this.

### Fermionic vs Bosonic and the Matsubara zero mode

The bosonic Matsubara grid includes the zero mode `ω_0 = 0`. The tail
`c_1/(iω) + c_2/(iω)^2 + …` is *singular* at ω=0: its leading term diverges
like `c_1/(iω)`. This is not a numerically meaningful statement about the
function — the high-frequency tail is an asymptotic expansion *at large ω*;
evaluating it at `ω=0` is mathematically a zero-frequency statement of a
high-frequency object, and it has no physical content for a bosonic GF.

Practically: for the *bosonic* Matsubara zero mode, define

```text
G_tail(i ω_0 = 0)  := 0
```

i.e. the tail is subtracted only at nonzero modes. Since the zero mode is a
single frequency, the loss of one point in an otherwise large Matsubara sum is
negligible and the inverse bosonic transform remains well-conditioned.

The code already distinguishes Fermionic and Bosonic Matsubara grids
(`MatsubaraGrid<Fermionic>`, `MatsubaraGrid<Bosonic>`), and
`GridExpansionMatsubara<data_type, S>` is already parameterised on the statistics
tag. The new work in this story is:

1. Making the *τ* representation equally explicit about statistics.
2. The inverse (Matsubara→τ) transform itself.
3. The tail object and its coefficient extraction.

### What the forward transform *is* (existing, to be left alone)

`include/grid/fourier.hpp` implements `FourierTransform<FromExp, ToExp, cplx>`
with `FromExp = GridExpansionTau<…, G>` (a *τ*-grid expansion) and
`ToExp = GridExpansionMatsubara<…, S>` (a Matsubara-grid expansion) for a
`Fermionic` or `Bosonic` S. It requires the source τ grid to be a `Quadrature`
(`GaussLegendreImaginaryTimeGrid`, `UniformImaginaryTimeGrid`); the
`ChebyshevNodeImaginaryTimeGrid` (no weights) is not usable with it. Its
kernel is `K(m,t) = w_t·exp(i ω_m τ_t)` and two execution modes (stored vs
on-the-fly) are supported.

The *inverse* transform is not the adjoint of that particular kernel; it is a
new object. But it should follow exactly the same structural pattern:

- The same two execution modes: `save = true` builds and stores the kernel
  once (and reuses it) via a `gemm`-based path, `save = false` computes the
  kernel element-by-element directly into the output tensor.
- The same error-checking style for axis order and size. The *sign
  convention* of the forward kernel is `+iω_m τ_t` (weighted by the τ-grid
  quadrature weights `w_t`), so the inverse kernel is `e^{-iω_m τ_t}` with a
  `1/β` prefactor. The two differ in both the sign convention and the
  normalization (the forward has quadrature weights `w_t`, the inverse has a
  uniform `1/β` prefactor); they are not a simple matrix transposition, but
  the *shape* of the kernel (one factor per Matsubara index, one per τ index,
  product of a phase and a weight) is the same, so the same `gemm`-based
  machinery applies.

### Summary of new machinery (this story adds)

1. A `StatisticsTag` parameter on `GridExpansionTau` (mirroring the existing
   `StatisticsTag` parameter on `GridExpansionMatsubara`), so that the two
   sides of a Fourier pair are statistics-matched and the transform can
   enforce this.
2. `InverseFourierTransform<FromExp, ToExp, DataType>` — the plain inverse
   (Matsubara → τ) finite Matsubara sum. Mirrors the forward
   `FourierTransform<FromExp, ToExp, DataType>` signature exactly:
   `FromExp` is a `GridExpansionMatsubara<…, S>`, `ToExp` is a
   `GridExpansionTau<…, G, S>`, `DataType` is the complex scalar. Same `save`
   flag, same `gemm`-based path, same axis-order/size checks. Works for both
   statistics (bosonic zero mode included as a normal summand).
3. `FermionicTail<data_type>` — a new coefficient-carrying object, following
   the same *mantra* as `ChebyshevExpansion*`: it owns a tensor
   `data_( spatial_indices..., <tail coeff> )` where the fastest axis (size
   `P`) holds `c_1, …, c_P`. For this Story it is fermionic by construction; a
   bosonic tail would be a future story. Its two evaluation entry points are:
   `at_matsubara(iω)  ->  Σ_k c_k / (iω)^k`
   `at_tau(τ)         ->  Σ_k c_k · T_k(τ)`
   (plus an `at_tau` batch that returns a full `GridExpansionTau` for a given
   τ grid, for direct composition in `TailCorrectedInverseFourierTransform`).
4. `TauBoundaryStencils` — the two independent one-sided polynomial fits
   (`M_left` stencils near 0+, `M_right` near β−) with their analytic
   derivative extractor. Templated over the real/complex scalar; the internal
   Lagrange-basis arithmetic is templated over `RealFloatingPoint` so it can
   be instantiated with either `double` or a multiprecision type (Objective
   feedback 4).
5. `TailCorrectedInverseFourierTransform<FromExp, ToExp, DataType>` —
   the single public call for the fermionic pipeline. Takes a
   `FermionicTail`, runs `r = G − tail`, applies `InverseFourierTransform`, and
   adds `tail(τ)`. Configuration: `M_left`, `M_right`, `save` flag. All the
   individual steps (stencil fit, coefficient extraction, tail evaluation, plain
   inverse) remain directly callable from their own classes for testing.
6. A new file in `src/inverse_fourier_tests.cpp` with
   `run_inverse_fourier_tests()`, declared in `src/tests.hpp` and called from
   `run_tests()` in `src/tests.cpp` (the same wiring used for Story 03).

## Objectives

1. Parameterise `GridExpansionTau` on a `StatisticsTag` (the same `S` parameter
   that `GridExpansionMatsubara` already carries), so that τ-side expansions
   explicitly encode fermionic or bosonic statistics. The Matsubara side already
   has this via `MatsubaraGrid<S>`; no change there is required. The new inverse
   transform must *check* that the two sides agree — because this Story's
   inverse transform is a template over `FromExp` and `ToExp` (both of which
   carry `S`), a statistics mismatch is a *template-instantiation error* in
   C++, checked at compile time by the primary template's `static_assert`;
   no runtime dispatch is required.

2. Implement `InverseFourierTransform<FromExp, ToExp, DataType>`, mirroring
   the forward `FourierTransform` signature: `FromExp` is a
   `GridExpansionMatsubara<…, S>`, `ToExp` is a `GridExpansionTau<…, G, S>`,
   `DataType` is the complex scalar. `save` flag for stored vs on-the-fly
   kernel, `gemm`-based path when `save=true`, inline path when `save=false`,
   axis/size checks. The statistics parameter `S` of `FromExp` and `ToExp`
   must agree (enforced by `static_assert` in the primary template, per
   Objective 2).

3. For `Bosonic` statistics, the inverse is the plain truncated Matsubara sum:

   ```text
   G(τ) = 1/β sum_n e^{-i ω_n τ} G(i ω_n)
   ```

   (The zero mode, if present, contributes its full weight; no tail.)

4. For `Fermionic` statistics, support a tail-subtracted inverse:

   ```text
   G(τ) = 1/β sum_n e^{-i ω_n τ} [G(i ω_n) - G_tail(i ω_n)] + G_tail(τ)
   ```

   with `G_tail` evaluated (a) on the Matsubara grid (subtracted) and
   (b) on the τ grid (added).

5. Implement the two independent one-sided polynomial fits
   (left boundary, right boundary) over an arbitrary τ-grid:

   ```text
   fit_left:  use the M_left τ-grid points with smallest τ;    produce f^(k)(0+)
   fit_right: use the M_right τ-grid points with largest τ;    produce f^(k)(β-)
   ```

   for `k = 0, ..., M-1` (i.e. any requested derivative order up to degree-1).

6. Implement the tail-coefficient extraction:

   ```text
   c_{k+1} = (-1)^{k+1} ( f^(k)(0+) + f^(k)(β-) ),   for k = 0, ..., P-1
   ```

   producing `c_1, ..., c_P` for a user-configurable `P ≤ min(M_left, M_right)`.

7. Implement `FermionicTail<data_type>` — a coefficient-carrying object on
   `data_( spatial..., <coeff> )` (fastest axis size `P` holds `c_1, …, c_P`),
   following the `ChebyshevExpansion*` mantra of owning a coefficient tensor
   and exposing `size()`, `data()`, and `operator()(i)`. It provides:
   - `at_matsubara(double iw)` — `Σ_k c_k/(iω)^k` at a single frequency
     (`G_tail(iω)`),
   - `at_tau(double tau)` — `Σ_k c_k·T_k(τ)` at a single point (`G_tail(τ)`),
   - `at_tau_grid(const G& grid)` — full `GridExpansionTau<…, G, Fermionic>`
     over an arbitrary supported τ grid (`G`),
   - `tail_order()` — `P` (size of the coefficient axis),
   - `set_coeff(size_t k, data_type c)` / `coeff(size_t k)` — index into the
     coefficient axis, for extraction (test 5 below).

8. Implement `TailCorrectedInverseFourierTransform<FromExp, ToExp,
   DataType>` — the single-call fermionic pipeline. Takes, in addition to
   the two expansions and a `save` flag, a `FermionicTail`, the left and
   right stencil sizes `M_left`, `M_right`; and an optional `P` override
   (defaults to the tail's stored `tail_order()`). `operator()` returns a
   `ToExp`. Each pipeline step (stencil fit, coefficient extraction, tail
   evaluation at Matsubara, plain inverse, tail evaluation at τ) is exposed
   as a separate method on a class of its own, so tests can call any of them
   independently.

9. Add tests for all of the above in a dedicated source file
   (`src/inverse_fourier_tests.cpp`) and wire them into the main test runner
   (`run_inverse_fourier_tests()` in `src/tests.hpp`, called from `run_tests()`
   in `src/tests.cpp`), following the convention used in Story 03.

## Design

### New types

```cpp
// Statistics-parameterised τ expansion (the new dimension S on the existing
// GridExpansionTau).  S_from == S_to is the statistics-match check.
template <FloatingPoint data_type, ImaginaryTimeGrid G, StatisticsTag S>
class GridExpansionTau
  : /* existing members */;

// Plain inverse: Matsubara -> tau.  Signature mirrors the forward transform
// exactly: <FromExp, ToExp, DataType>.  S_from == S_to is enforced by
// `static_assert` in the primary template (Objective 2).
template <detail::FromGridExpansionMatsubara FromExp,
          detail::ToGridExpansionTau         ToExp,
          ComplexFloatingPoint               DataType>
class InverseFourierTransform {
  // FromExp := GridExpansionMatsubara<DataType, S>      (input)
  // ToExp   := GridExpansionTau<DataType, G, S>         (output)
  // DataType == typename ToExp::scalar_type == typename FromExp::scalar_type

  bool save;
  // save == true  : kernel K[m,t] stored and reused via `gemm`
  // save == false : kernel computed element-by-element on the fly
  ToExp operator()(const FromExp& in) const;
};

// Fermionic tail: stores c_1..c_P as a tensor (spatial x P, P fastest); owns
// β (needed for the T_k polynomials).  Templated over the data type AND the
// target τ grid G, because at_tau_grid returns a GridExpansionTau<G, S>.
// (Fermionic by construction: the T_k polynomials are Fermi-Matsubara-specific
// and this Story is fermionic-only for the tail.)
template <FloatingPoint data_type, ImaginaryTimeGrid G>
class FermionicTail {
  Tensor<data_type, Executor::Host> c_;  // c_[spatial…, k]  (k = 0..P-1, P fastest)
  InverseTemperature beta_;
  size_t P_;                             // P = tail_order()
public:
  // Accessors mirroring the ChebyshevExpansion* interface.
  size_t size() const;                    // == P_  (size of the coefficient axis)
  size_t P()    const;                    // alias for size()
  size_t tail_order() const;              // == P_
  const Tensor<data_type, Executor::Host>& data() const;
  Tensor<data_type, Executor::Host>&       data();
  // The i-th coefficient c_{i+1}.
  data_type operator()(size_t i) const;   // c_{i+1}
  void set_coeff(size_t i, data_type c);  // set c_{i+1}

  // Evaluation.
  data_type at_matsubara(double iw)   const;  // sum_k c_{k+1}/(iw)^(k+1)
  data_type at_tau(double tau)        const;  // sum_k c_{k+1} * T_{k+1}(tau)
  // Full grid evaluation (used by TailCorrectedInverseFourierTransform).
  GridExpansionTau<data_type, G, Fermionic>
      at_tau_grid(const G& grid) const;
};

// One-sided polynomial fit (left or right stencil).  Templated over the real
// scalar type so the Lagrange-basis arithmetic can be either `double` or a
// multiprecision real type (per feedback 4).
template <RealFloatingPoint real_type>
class Stencil {
  std::vector<real_type>  x_;            // abscissas (monotonic), size M
  std::vector<real_type>  y_;            // values at x_ (same real type)
  size_t  M_;                            // M = x_.size()
public:
  Stencil(std::vector<real_type> x, std::vector<real_type> y);
  // k-th derivative of the unique degree < M polynomial through (x_i, y_i),
  // evaluated at the stencil-endpoint (0+ or beta-).   k = 0..M-1.
  real_type derivative(size_t k, double evaluation_point) const;
  std::vector<real_type> derivatives(size_t K, double evaluation_point) const;
};

// Two-stencil boundary object (independent).  Carries the tensor-valued
// endpoint derivatives (they are themselves a Tensor) plus the coefficient-
// extraction helper (the c_{k+1} formula).
template <RealFloatingPoint real_type, class cplx>
class TauBoundaryStencils {
  Stencil<real_type>  left_;             // stencils near 0+ (smallest τ)
  Stencil<real_type>  right_;            // stencils near β- (largest τ)
  // Tensor-valued derivatives.   Lk_[spatial…, k] = f^(k)(0+),
  //                              Rk_[spatial…, k] = f^(k)(β-),    k = 0..K.
  Tensor<real_type /* or cplx */, Executor::Host>  Lk_;
  Tensor<real_type /* or cplx */, Executor::Host>  Rk_;
public:
  const auto& L() const;                  // f^(k)(0+)  tensor
  const auto& R() const;                  // f^(k)(β-)  tensor
  // Extract c_1..c_P using  c_{k+1} = (-1)^{k+1} (f^(k)(0+) + f^(k)(β-)).
  std::vector<cplx> extract_coeffs(size_t P) const;
};

// Composite fermionic pipeline (Objective 8).  Takes, in addition to the two
// expansion types and the save flag, a FermionicTail, M_left, M_right,
// and an optional P override (defaults to tail.tail_order()).   operator()
// returns a ToExp.   Each pipeline step is exposed separately:
template <detail::FromGridExpansionMatsubara FromExp,
          detail::ToGridExpansionTau         ToExp,
          ComplexFloatingPoint               DataType>
class TailCorrectedInverseFourierTransform<FromExp, ToExp, DataType> {
  InverseFourierTransform<FromExp, ToExp, DataType> plain_;  // plain inverse
  // Configuration: M_left, M_right, P (optional), save
  size_t M_left_, M_right_;
  size_t P_ = /* defaults to tail.tail_order() */;
public:
  TailCorrectedInverseFourierTransform(FromExp in_grid, ToExp out_grid,
                                       bool save, size_t M_left, size_t M_right);
  // Full pipeline: r = G - tail,  apply plain inverse,  add tail(τ).
  ToExp operator()(const FromExp& in, const FermionicTail<...>& tail) const;

  // Exposed pipeline steps (each a separate public method, for testing):
  // Step 1:  two-stencil boundary fit  (f^(k)(0+), f^(k)(β-))  for k = 0..P-1
  TauBoundaryStencils<...> stencil_fit(const /* τ-grid expansion */& src,
                                       size_t M_left, size_t M_right) const;
  // Step 2:  extract c_1..c_P from the stencil fit
  std::vector<cplx> extract_coeffs(const TauBoundaryStencils<...>& st, size_t P) const;
  // Step 3:  build the FermionicTail
  FermionicTail<...> make_tail(const std::vector<cplx>& c,
                               InverseTemperature beta, size_t P) const;
  // Step 4:  r(iω) = G(iω) - tail(iω)   (element-wise on the Matsubara axis)
  FromExp  subtract_tail(const FromExp& in,
                         const FermionicTail<...>& tail) const;
  // Step 5:  plain inverse on r   (delegates to plain_)
  ToExp    inverse_tau(const FromExp& r) const;
  // Step 6:  add tail(τ)  (delegates to tail.at_tau_grid)
  ToExp    add_tail(const ToExp& resid, const FermionicTail<...>& tail) const;
};
```

(Here `cplx` is the complex scalar type and `TensorT` is a matching Tensor
specialization; the actual API will be templated over both. The sketch is
for the *shape* of the types, not for compilation.)

These types should be placed in headers that mirror the existing layout:
`include/grid/` (the statistics parameter on `GridExpansionTau` fits next to
the existing definition) and `include/tail.hpp` (or similar), for the tail
object and the two-stencil derivative extractor.

### Inverse (Matsubara→τ) transform

The inverse is a finite Matsubara sum, not a quadrature: there is no source of
weights — the factor is `1/β` — so *no* `Quadrature` constraint applies to the
*target* τ point set. (Mirror image of the forward transform, where the
*source* τ grid is the quadrature; here the output is one of the *supported* τ
grids (uniform, Gauss–Legendre, Chebyshev-node) and we evaluate the exact
inverse Matsubara sum *at that grid's points* — the kernel has no quadrature
error of its own, so any of the supported grids is a valid output target.)

```text
K(m, t) = (1/β) · e^{-i ω_m τ_t}
```

with `m` indexing the Matsubara side (input, contracted) and `t` the τ side
(output), keeping the τ axis fastest in the output tensor. This is exactly the
time-reversed counterpart of the forward kernel `K(m,t) = w_t e^{+iω_m τ_t}`;
only the sign convention and normalization differ, and there is no source of
quadrature weights on the Matsubara side.

**Fermionic case, tail-corrected** (this is what this Story adds on top): the
`1/ω^k` tail terms diverge at `ω=0` (relevant only for the bosonic zero mode),
so the zero-mode convention `G_tail(0) := 0` — already motivated by the
singular `1/ω` term — is reused, and the pipeline is:

1. `r(iω_n) = G(iω_n) - G_tail(iω_n)` on the Matsubara side (element-wise;
   at `ω=0` this is just `G(0) - 0`).
2. Apply the inverse kernel `K(m,t) = (1/β)·e^{-iω_m τ_t}` to `r` to obtain
   `G_residual(τ_t)`. This is exactly the bosonic-style kernel above, so the
   `save`/`gemm` machinery is shared between the two cases.
3. Add `G_tail(τ_t)` element-wise along the τ axis to get `G(τ_t)`.

### The two one-sided polynomial fits

For a τ-grid `τ_1 < τ_2 < … < τ_N`, the *left* stencil is a contiguous
prefix `τ_1, …, τ_M` and the *right* stencil is a contiguous suffix
`τ_{N-M+1}, …, τ_N`. For each:

1. Compute the Lagrange basis polynomials `ℓ_i(τ)` of degree `M-1`,
   for `i = 1..M`, that satisfy `ℓ_i(τ_j) = δ_ij` on the stencil.
   On a general (nonuniform) stencil, this amounts to
   `ℓ_i(τ) = ∏_{j≠i} (τ - τ_j)/(τ_i - τ_j)`.
2. Compute `ℓ_i^{(k)}` analytically (these are polynomials; their derivatives
   are polynomials; the k-th derivative of a monomial is closed form). At the
   *left* boundary, `k = 0..M-1`: `f^{(k)}(0+) ≈ Σ_i y_i · ℓ_i^{(k)}(0)`.
   At the *right* boundary: `f^{(k)}(β-) ≈ Σ_i y_i · ℓ_i^{(k)}(β)`.
3. Return `f^{(k)}(0+)` and `f^{(k)}(β-)`.

This construction is exact for `f` a polynomial of degree `< M`, on any τ-grid.
When the τ-grid has a known analytical form (uniform midpoint, Gauss–Legendre),
the stencil can be precomputed and its derivative coefficients cached.

**Numerical note on high `k` / close points.** The Lagrange basis derivatives
`ℓ_i^{(k)}` at an endpoint are bounded above by `O(k!·M^k / h^k)` where `h` is
the smallest gap. For `M ~ 8`, `k` up to `7`, and typical β/τ scales (β ~ 10,
h ~ 1), the magnitudes stay well within `double`. For higher `M` and `k`,
the multiprecision machinery in `include/multiprecision.hpp` is available if
the tensor scalar type is promoted — the API should not preclude that.

**Why not a single wrapped-boundary fit?** A single polynomial through the
`M` nearest points to the *joined* boundary (straddling the τ=0/τ=β seam)
would implicitly assume the function is continuous and smooth at the seam,
which is exactly the assumption that fails for Green's functions with a jump
at the boundary. The two-domain stencil is correct for any single-sided
behavior of `f` at each endpoint, and is the standard choice in the
Green's-function literature (e.g. tail subtraction in fermionic GFs,
`G^(k)(0+)` and `G^(k)(β−)` enter separately).

### Tail extraction and the coefficient formula

Given `f^(k)(0+)` and `f^(k)(β-)` (independent of k, for `k = 0..P-1`),
define

```text
c_{k+1} = (-1)^{k+1} ( f^(k)(0+) + f^(k)(β-) )
```

and build `FermionicTail{β, [c_1,...,c_P]}`. For a *noninteracting* Green's
function (see tests below), the exact tail satisfies
`c_{k+1} = ε^k` regardless of β; the code should reproduce that.

For the bosonic case, no tail is required by this Story: the bosonic inverse
is just the finite Matsubara sum. If a bosonic tail were needed in a future
Story, same machinery applies (just with bosonic Matsubara frequencies —
the formula changes because the zero mode is included and `e^{i ω β} = +1`,
so some of the endpoint terms drop or double up). This is explicitly out of
scope here.

## Tests

All tests go in a new source file `src/inverse_fourier_tests.cpp` with a
`run_inverse_fourier_tests()` entry point (declared in `src/tests.hpp` and
called from `run_tests()` in `src/tests.cpp`), following the wiring used by
`src/fourier_tests.cpp`. The existing `{{"sp",1}}` scalar spatial dimension
convention is used where a tensor-valued input is needed (test 8 uses a
larger spatial dim to exercise the tensor-valued path).

1. **Two-stencil polynomial fit: exactness for polynomials.** On
   `UniformImaginaryTimeGrid`, `GaussLegendreImaginaryTimeGrid`, and
   `ChebyshevNodeImaginaryTimeGrid`, sample a random polynomial of degree
   `d ≤ M-1` (`f(τ) = a_0 + a_1 τ + … + a_d τ^d`) and verify the extracted
   derivatives `f^(k)(0+)` and `f^(k)(β-)` agree with the exact derivatives
   to numerical precision (≤ 1e-14). Test `M` in {4, 8, 16} and `d` in `0..M-1`.

2. **Two-stencil: derivative order independence.** For a fixed `f`,
   requesting `k = 0..M-1` in one call must return all the correct
   derivatives simultaneously, with the same stencil.

3. **FermionicTail evaluation (Matsubara) for a manually set coefficient set.**
   Pick `c = [c_1..c_P]` of random magnitudes; for several Matsubara
   frequencies `iω` (positive and negative, **including `ω = 0` for the
   bosonic case**, where `G_tail(0) := 0` by convention but would be singular
   if `c_1 ≠ 0`), verify `FermionicTail::at_matsubara(iω)` against
   `Σ c_k / (iω)^k` computed in closed form (watch out for the `ω = 0`
   convention explicitly).

4. **FermionicTail evaluation (τ) for a manually set coefficient set, plus T_k consistency.**
   Same c set; for several τ in (0, β), verify `FermionicTail::at_tau(τ)`
   against `Σ c_k·T_k(τ)` evaluated with the exact `T_k` polynomials (the `T_k`
   used by `FermionicTail`). Include multiple values of β. Also, for `k = 1..6`
   and several `(β, τ)`, verify numerically that the explicit Matsubara sum
   `T_k(τ) = (1/β) Σ_n e^{-iω_n τ} (iω_n)^{-k}` approaches the analytic `T_k(τ)`
   (a reference test — confirms the `T_k` formulas are correct and the
   convention matches this code's forward kernel). The Matsubara sum converges
   as `O(N^{-k})` (the `1/ω_n^k` term), so `k = 1` converges slowly — the test
   should allow a correspondingly looser tolerance for small k.

5. **One-level Green's function: end-to-end.** The noninteracting fermionic
   one-particle GF:
   - `f(τ) = -(1-n_F)·e^{-ε τ}` with `n_F = 1/(e^{βε}+1)`;
   - `G(iω) = 1/(iω - ε)`, so the exact tail satisfies `c_k = ε^{k-1}`.
   Verify, step by step (each step calls its own class, no pipeline needed):
   - `FermionicTail::at_matsubara(iω)` agrees with `1/(iω)` at several Matsubara
     frequencies for `c_1 = 1` and `P = 1` (sanity check on the coefficient).
   - `Stencil` fit on the τ samples of `f` gives `f^(k)(0+)` and `f^(k)(β−)`
     matching the exact `-(1-n_F)·(-ε)^k` and `-(1-n_F)·(-ε)^k·e^{-ε β}`
     to numerical precision.
   - `TauBoundaryStencils::extract_coeffs(P)` reproduces `c_k = ε^{k-1}` to
     numerical precision for several `ε` and `β`.
   - The *existing* (forward) Matsubara transform of `f` agrees with
     `G(iω_n) = 1/(iω_n − ε)` on the Matsubara grid.
   - The full pipeline — `TailCorrectedInverseFourierTransform::operator()`
     — recovers `f(τ)` to numerical precision for several τ, multiple `ε`, β,
     and tail order `P`; `P ≤ 6` in this story (only explicit formulas
     `T_1..T_6` are provided).

6. **Inverse transform convergence (fermionic, tail-corrected vs plain).**
   Sample the noninteracting fermionic GF (case 5) on a τ grid, transform
   forward to Matsubara. For each Matsubara cutoff `Nω`, compute the max-τ
   error for both `InverseFourierTransform` (plain) and
   `TailCorrectedInverseFourierTransform` (tail-corrected, `P = 4` and
   `P = 6`). Verify:
   - the plain error `e_plain` decreases (weakly) as `Nω` grows;
   - the tail-corrected error `e_tailcorr` decreases at least as fast as
     `O(Nω^{-P})` for the chosen `P`, and is monotonically decreasing in `Nω`;
   - `e_tailcorr < e_plain` for all `Nω ≥` some threshold — the point is that
     the tail-corrected transform converges *substantially faster* than the
     plain one, and the improvement is visible near the imaginary-time
     boundaries.

7. **Inverse transform convergence (bosonic).** Take a bosonic function with
   endpoint structure (e.g. a bosonic GF or any function continuous on [0, β]
   with `f(0) ≠ f(β)`); the *plain* (no-tail) inverse sum must converge to the
   correct τ-space value — the tail machinery is only a *convergence
   accelerator*, not a requirement, for the bosonic case. The convergence rate
   is controlled by the first non-vanishing endpoint difference in the
   sequence `f(0) − f(β)`, `f'(0) − f'(β)`, `f''(0) − f''(β)`, …:
   - a jump in the function value `f(0) ≠ f(β)` → error O(Nω^{-1});
   - a jump only in the first derivative → error O(Nω^{-2});
   - all endpoint differences up to order m vanish → error O(Nω^{-(m+1)}) or
     faster (spectral if all vanish and f is C^∞ at the seam).
   Verify numerically, with concrete functions that isolate each regime:
   - `f(τ) = 1 − τ/β` (value jump `f(0) − f(β) = 1`): the inverse-sum error
     decays at ≈ O(Nω^{-1});
   - `f(τ) = cos(2π τ/β)` (`f(0) = f(β)`, and all endpoint differences vanish,
     so the periodic extension is C^∞): the inverse-sum error is spectral
     (≈ 1e-14 at `Nω = 16` or similar, depending on the grid), **not** just
     O(Nω^{-1}).
   In both cases the *convergence to the correct value* is the key assertion —
   the tail machinery is a convergence accelerator, not a correctness
   requirement, for the bosonic case. (Bosonic convergence is not the focus of
   this story; the test merely confirms the framework works uniformly for
   both statistics.)

8. **Tensor-valued behavior.** Construct a τ-grid sample that is tensor-valued
   (e.g. with one spatial index, plus a second spatial index to exercise the
   non-scalar path). Verify:
   - `FermionicTail` stores its coefficients as a tensor whose spatial dims
     match the input (`c_` is a Tensor with the tail-coefficient axis fastest,
     size P, and matching spatial dims).
   - `Stencil` fit, `TauBoundaryStencils::extract_coeffs(P)`, and
     `TailCorrectedInverseFourierTransform::operator()` each run
     element-wise over the tensor (only the τ / Matsubara axis changes), and
     the reconstructed τ samples agree with the true tensor-valued function.
   - The extracted coefficients, for a non-scalar test (e.g. an identity-matrix
     GF), match the analytic values tensor-by-component.

9. **Statistics-mismatch rejection.** Attempt to use a `Fermionic` τ expansion
   with a `Bosonic` Matsubara expansion (or vice versa) inside
   `InverseFourierTransform` or `TailCorrectedInverseFourierTransform`. Since
   the two sides' statistics (`S`) are part of the expansion *types* in both
   templates, a mismatch is a **compile-time failure** (template-instantiation
   error or `static_assert` from the statistics-match check). The test is
   therefore to verify that a code snippet using a mismatched pair *does not
   compile*. If the implementation opts for a runtime check instead (a
   non-template statistics field), the equivalent test is a runtime
   `std::runtime_error`. Either way the acceptance criterion is: mismatched
   sides are rejected, never silently accepted.

## Definition of done

1. `GridExpansionTau` is parameterised on a `StatisticsTag` (the new `S`
   parameter); `GridExpansionMatsubara` already does this. The inverse
   transform's two sides are statistics-matched and the check is enforced
   (compile-time `static_assert` in the primary template, per Objective 2).

2. `InverseFourierTransform<FromExp, ToExp, DataType>` is implemented and
   follows the same API pattern as the forward `FourierTransform` (save mode,
   `gemm` path, inline path, axis/size checks).

3. `FermionicTail<data_type, G>` is implemented; it owns a coefficient tensor
   (`c_1, …, c_P` per spatial index) and provides `at_matsubara(iω)`,
   `at_tau(τ)`, and `at_tau_grid(grid)`.

4. `Stencil<real_type>` (one one-sided polynomial fit, templated over
   `RealFloatingPoint`) and `TauBoundaryStencils` (both fits, plus
   `extract_coeffs(P)`) are implemented; the internal Lagrange-basis
   arithmetic is templated over `RealFloatingPoint` so it can be instantiated
   with higher precision (Objective 7, feedback 4).

5. `TailCorrectedInverseFourierTransform<FromExp, ToExp, DataType>` is
   implemented; its `operator()` runs the pipeline in one call, and each
   pipeline step (stencil fit, coefficient extraction, `FermionicTail` build,
   `subtract_tail`, plain inverse, `add_tau`) is a separate public method for
   testing (Objective 8, feedback 3).

6. Tests 1–9 above pass.

7. The existing forward `FourierTransform` and all Story 03 tests continue to
   pass unchanged.

8. The new source file, and any new headers, compile cleanly in the existing
   CMake build.

## Open questions / resolved decisions

All design decisions previously listed as open have been resolved:

1. **Naming (resolved).**
   - `GridExpansionTau<data_type, G, S>` — the existing class, with the new `S`
     (statistics) parameter.
   - `InverseFourierTransform<FromExp, ToExp, DataType>` — the plain inverse,
     signature mirroring the forward `FourierTransform`.
   - `FermionicTail<data_type, G>` — coefficient tensor owning `c_1..c_P`;
     follows the *Expansion* mantra (owns a coefficient tensor, exposes
     `size()`, `data()`, `operator()(i)`).
   - `Stencil<real_type>` / `TauBoundaryStencils` — the two one-sided
     polynomial fits (independent), templated over `RealFloatingPoint` for
     the internal arithmetic.
   - `TailCorrectedInverseFourierTransform<FromExp, ToExp, DataType>` — the
     single public call for the fermionic pipeline.
2. **Tail object shape (resolved).** A standalone type, `FermionicTail`, that
   owns a coefficient tensor (the *Expansion* mantra) rather than being folded
   into the `GridExpansion*` family — its "grid" is a coefficient vector,
   not a set of points. `FermionicTail` is used *by*
   `TailCorrectedInverseFourierTransform` (feedback 2).
3. **Pipeline vs steps (resolved).** `TailCorrectedInverseFourierTransform`
   exposes a simple `operator()` that runs the whole pipeline; *each step*
   (stencil fit, coefficient extraction, tail evaluation at Matsubara, plain
   inverse, tail evaluation at τ) is a separate public method on a class of
   its own, so tests can call any of them independently (feedback 3).
4. **Internal arithmetic precision (resolved).** `Stencil` is templated over
   `RealFloatingPoint` so the Lagrange-basis arithmetic can be instantiated
   with either `double` or a multiprecision real type. The `FermionicTail`
   coefficient tensor and the transforms themselves remain templated over the
   `DataType` (complex) scalar of the expansions (feedback 4).
5. **Bosonic tails (still out of scope).** This is *only* in the sense that
   the *tail machinery* is fermionic; the *bosonic inverse* (`plain
   InverseFourierTransform` on a bosonic expansion) is in scope and is the
   test-7 case. A future Story may add a `BosonicTail` analog; this Story
   does not implement one.

Remaining open questions:

1. **Precision (practical).** For large `P` (say `P > 10`) the Lagrange-basis
   derivatives grow factorially and the extracted `c_k` can be ill-conditioned.
   For the intended use (`P` small, `M ~ 8`, `β ~ O(10)`), `double` is
   sufficient. If a future story needs very high `P`, `Stencil` will be
   instantiated with a multiprecision real type (already supported by the
   `RealFloatingPoint` templating of feedback 4); the API is already written
   so the caller can choose precision without a code change.

2. **`M = 8, keep variable` (carried forward from the original Story).**
   The stencil size `M_left` and `M_right` are runtime arguments (not
   compile-time template parameters); the production default is 8, and the
   tests verify several values (Objective 5).
