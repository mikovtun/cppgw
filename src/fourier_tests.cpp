// ============================================================================
//  Story 03 — Fourier-transform convergence tests (analytic references)
//
//  Validates the EXISTING production FourierTransform (tau -> Matsubara)
//  against three functions with known analytic transforms; it does not touch
//  the algorithm and does not add anything to the production API.
//
//  Convention implemented by the production code (grid/fourier.hpp + tau.hpp):
//
//      F(iω_m) = ∫_0^β  e^{i ω_m τ} f(τ) dτ
//
//  approximated by the source grid's quadrature rule.  Two tau grids exist
//  that are both Quadratures (and hence usable via FourierTransform's
//  `Quadrature` static_assert):
//
//    * UniformImaginaryTimeGrid       τ_t = (t+½)β/T,  w_t = β/T    (midpoint)
//    * GaussLegendreImaginaryTimeGrid n Gauss–Legendre nodes on (0,β)
//
//  Note: adding these tests exposed a real defect (not in FourierTransform,
//  which simply applies the provided rule): the grid's naive NR-style Newton
//  root-finder, near the middle roots of even-degree P_n (where P_n'(0)=0),
//  crossed the origin and converged to the WRONG root, corrupting nodes and
//  weights. It was fixed with a bracketed Newton (bisection fallback) in
//  include/grid/tau.hpp; the tests below then pass for the right reason.
//  (This is a grid implementation bug, which is why the Fourier transform
//  itself was left untouched.)
//
//  Both are constructed identically as  Grid(beta, n)  with n nodes; "grid
//  refinement" means increasing n (spectral for GL, ~O(n^-2) for midpoint).
//  ChebyshevNodeImaginaryTimeGrid has no weights (it is only a Grid, not a
//  Quadrature), so it is NOT usable with FourierTransform and is out of scope;
//  its DCT integration path is a different algorithm.
//
//  Matsubara frequencies come from MatsubaraGrid<Fermionic>:
//  ω = (±1, ±3, ±5, ...)·(π/β).  All analytic references below are exact at
//  Fermionic frequencies, where e^{iωβ} = -1:
//
//    1. f(τ) = 1
//         F(iω_n) = (e^{iω_n β} - 1)/(iω_n) = -2/(iω_n)
//    2. f(τ) = e^{-a τ}
//         F(iω_n) = (e^{β(iω_n - a)} - 1)/(iω_n - a)         [exact, any ω]
//    3. g(τ) = -(1 - n_F(ε)) e^{-ε τ},  n_F = 1/(e^{βε} + 1)
//         G(iω_n) = (1 - n_F)·(e^{-βε} + 1)/(iω_n - ε)
//                   = 1/(iω_n - ε)
//
//  For each (grid, function, β) the tests verify:
//    A. consistency          : numerical result close to the analytic one even
//                              on the coarsest grid;
//    B. tau-grid refinement  : refining the node count strictly decreases the
//                              max error over the tested Matsubara points;
//    C. Matsubara refinement : a frequency already present in a finer Matsubara
//                              grid transforms to the SAME value (the transform
//                              at a fixed frequency is independent of how many
//                              more frequencies the grid carries);
//    D. dense grid           : a very dense tau + Matsubara grid lands the max
//                              error below a tight absolute tolerance.
//
//  "Scalar-valued" input: GridExpansionTau requires at least one spatial dim,
//  so a single size-1 "sp" axis is carried through; the output is [matsu, sp].
// ============================================================================

#include "main.hpp"
#include "tests.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cppgw {
namespace {

using cplx    = std::complex<double>;
using FreqFn  = std::function<cplx(double)>;      // F(iω) at frequency ω
using TauFn   = std::function<double(double)>;    // f(τ)

// How the error is expected to behave under tau-node refinement (the
// "expected numerical behavior" of each discretization):
//   StrictDecreasing    : e.g. midpoint rule, O(n^-2) tail -> strictly
//                         decreasing error at every refinement level.
//   SpectrallySaturated : e.g. Gauss–Legendre, error at the ~1e-12..1e-15
//                         machine floor from n >= 16 -> assert it stays near
//                         machine precision at EVERY level (the strongest
//                         possible "convergence").
enum class Convergence { StrictDecreasing, SpectrallySaturated };

size_t mgrid_size(double beta, size_t halfN) {
  MatsubaraGrid<Fermionic> g(halfN, InverseTemperature(beta));
  return g.size();
}

// ----- numerics ------------------------------------------------------------
// Max |numerical - analytic| error over ALL Matsubara points of the target
// grid (frequencies are read from the grid, never assumed), for a tau grid G.
template <class G>
    requires (Quadrature<G> && ImaginaryTimeGrid<G>)   // usable by FourierTransform
double max_error(double         beta,
                 size_t         n,
                 size_t         halfN,
                 const TauFn&   f,
                 const FreqFn&  Fexact,
                 cplx*          result = nullptr)     // optional, for C.
{
  using InExp   = GridExpansionTau<double, G>;
  using OutExp  = GridExpansionMatsubara<cplx, Fermionic>;
  using FT      = FourierTransform<InExp, OutExp, cplx>;

  const InverseTemperature     bt    (beta);
  const G                      tau   (bt, n);
  const MatsubaraGrid<Fermionic> mgrid(halfN, bt);

  // Scalar input: spatial {"sp",1}; grid axis is added as the fastest.
  Tensor<double, Executor::Host> spatial({{"sp", 1}});
  InExp in(spatial, tau);
  {
    const size_t T = tau.size();
    double*      d = in.data().data();         // [tau (fast), sp=1]
    for (size_t t = 0; t < T; ++t)
      d[t] = f(tau(t).value);
  }

  FT       xform(tau, mgrid, /*save=*/true);
  const OutExp out = xform(in);                // data: [matsu (fast), sp=1]

  const size_t  M = mgrid.size();
  const cplx*   o = out.data().data();

  if (result)
    std::copy(o, o + M, result);

  double maxerr = 0.0;
  for (size_t m = 0; m < M; ++m)
    maxerr = std::max(maxerr, std::abs(o[m] - Fexact(mgrid(m).value)));
  return maxerr;
}

// Loose scale of |F(iω)| on the grid (sanity only, never a convergence criterion).
double analytic_scale(double beta, size_t halfN, const FreqFn& Fexact) {
  const InverseTemperature     bt    (beta);
  const MatsubaraGrid<Fermionic> mgrid(halfN, bt);
  double s = 0.0;
  for (size_t m = 0; m < mgrid.size(); ++m)
    s = std::max(s, std::abs(Fexact(mgrid(m).value)));
  return s;
}

// ----- one function, one β, one tau grid -----------------------------------
template <class G>
    requires (Quadrature<G> && ImaginaryTimeGrid<G>)
void one_case(const std::string& name,
              double             beta,
              const TauFn&       f,
              const FreqFn&      Fexact,
              const std::vector<size_t>& n_levels,    // increasing node counts
              size_t             dense_n,             // very dense tau grid
              size_t             dense_halfN,         // very dense Matsubara grid
              double             dense_tol,           // absolute error tolerance
              Convergence        conv,
              double             saturated_tol = 1e-9)  // near-machine-precision bound
{
  const size_t conv_halfN = std::max<size_t>(4, n_levels.front() / 4);
  const double scale      = analytic_scale(beta, conv_halfN, Fexact);

  // A. + B.  convergence under tau-node refinement (fixed Matsubara grid)
  std::vector<double> errs;
  errs.reserve(n_levels.size());
  for (size_t n : n_levels)
    errs.push_back(max_error<G>(beta, n, conv_halfN, f, Fexact));

  const double e0 = errs.front();
  if (!(std::isfinite(e0) && e0 < 5.0 * scale))
    throw std::runtime_error(std::string(name)
      + ": coarsest case not consistent with analytic ("
      + "e=" + std::to_string(e0) + ", |F|~" + std::to_string(scale) + ")");

  if (conv == Convergence::StrictDecreasing) {
    for (size_t k = 1; k < errs.size(); ++k)
      if (!(errs[k] < errs[k-1]))
        throw std::runtime_error(std::string(name)
          + ": error did not decrease under tau-grid refinement"
          + " (e=" + std::to_string(errs[k-1]) + " -> " + std::to_string(errs[k]) + ")");
  } else {  // SpectrallySaturated: at/around machine precision at every level
    for (size_t k = 0; k < errs.size(); ++k)
      if (!(errs[k] < saturated_tol))
        throw std::runtime_error(std::string(name)
          + ": expected a spectrally accurate (near machine precision) rule, but"
          + " error at n=" + std::to_string(n_levels[k]) + " is "
          + std::to_string(errs[k]) + " (bound " + std::to_string(saturated_tol) + ")");
  }

  // C.  Matsubara refinement: shared frequencies must be unchanged.
  {
    const size_t n = std::max<size_t>(128, n_levels.back());
    const size_t small = 2, large = 2 * small;
    std::vector<cplx> vs(mgrid_size(beta, small)), vl(mgrid_size(beta, large));
    max_error<G>(beta, n, small, f, Fexact, vs.data());
    max_error<G>(beta, n, large, f, Fexact, vl.data());

    const InverseTemperature    bt (beta);
    const MatsubaraGrid<Fermionic> gs(small, bt), gl(large, bt);
    for (size_t ms = 0; ms < gs.size(); ++ms)
    {
      const double w = gs(ms).value;
      bool found = false;
      for (size_t ml = 0; ml < gl.size(); ++ml)
        if (gl(ml).value == w)
        {
          if (!(std::abs(vs[ms] - vl[ml]) < 1e-9 * std::abs(vs[ms])))
            throw std::runtime_error(std::string(name)
              + ": Matsubara refinement changed an already-present frequency (w="
              + std::to_string(w) + ")");
          found = true;
          break;
        }
      if (!found)
        throw std::runtime_error(std::string(name)
          + ": frequency of the small Matsubara grid missing from the larger one");
    }
  }

  // D.  dense taus: densest grid must be very close to analytic.
  const double e_dense = max_error<G>(beta, dense_n, dense_halfN, f, Fexact);
  if (!(e_dense < dense_tol))
    throw std::runtime_error(std::string(name)
      + ": dense-grid result not close to analytic (e="
      + std::to_string(e_dense) + " >= tol " + std::to_string(dense_tol) + ")");
  if (conv == Convergence::StrictDecreasing && !(e_dense < e0))
    throw std::runtime_error(std::string(name)
      + ": dense-grid result not better than the coarsest one");

  std::cout << "  " << name
            << "  beta=" << beta
            << "  e[n=16]=" << e0
            << "  e[n=128]=" << errs.back()
            << "  e[dense n=" << dense_n << ", halfN=" << dense_halfN
            << " M=" << mgrid_size(beta, dense_halfN) << "]=" << e_dense
            << (e_dense < dense_tol ? "  OK" : "  ** FAIL **") << std::endl;
}

// ----- one tau grid over all functions / β ---------------------------------
template <class G>
    requires (Quadrature<G> && ImaginaryTimeGrid<G>)
void run_all_functions(const char* grid_name, Convergence conv,
                       double saturated_tol = 1e-9)
{
  const double a   = 0.3;    // decay rate of the exponential (small, positive)
  const double eps = 0.2;    // orbital energy of the 1-particle GF (small, positive)

  const std::vector<double>     betas    = {2.0, 10.0, 20.0};
  const std::vector<size_t>     n_levels = {16, 32, 64, 128, 256};
  constexpr size_t              dense_n    = 4096;   // very dense tau grid
  constexpr size_t              dense_halfN = 32;    // very dense Matsubara grid (64 pt)
  constexpr double              dense_tol  = 1e-4;   // absolute max error, all frequencies

  for (double beta : betas)
  {
    // 1. Constant  f(τ)=1  <-->  F(iω) = -2/(iω)
    {
      const TauFn  f  = [](double) -> double         { return 1.0; };
      const FreqFn Fe = [](double w) -> cplx         { return -2.0 / cplx(0.0, w); };
      one_case<G>(std::string(grid_name) + ": const", beta, f, Fe, n_levels,
                  dense_n, dense_halfN, dense_tol, conv, saturated_tol);
    }

    // 2. Exponential  f(τ)=e^{-aτ}  <-->  F(iω) = (e^{β(-a+iω)}-1)/(iω-a)
    {
      const double b = beta;
      const TauFn  f  = [a](double tau) -> double    { return std::exp(-a * tau); };
      const FreqFn Fe = [a, b](double w) -> cplx {
        return (std::exp(cplx(-a * b, w * b)) - 1.0) / cplx(-a, w);
      };
      one_case<G>(std::string(grid_name) + ": exp(a=" + std::to_string(a) + ")", beta,
                  f, Fe, n_levels, dense_n, dense_halfN, dense_tol, conv, saturated_tol);
    }

    // 3. 1-particle GF  g(τ)=-(1-n_F)e^{-ετ}  <-->  G(iω) = 1/(iω-ε)
    {
      const double b = beta;
      const double nF = 1.0 / (std::exp(eps * b) + 1.0);
      const TauFn  f  = [nF, eps](double tau) -> double {
        return -(1.0 - nF) * std::exp(-eps * tau);
      };
      const FreqFn Fe = [eps](double w) -> cplx {
        return 1.0 / cplx(-eps, w);
      };
      one_case<G>(std::string(grid_name) + std::string(": GF(eps=") + std::to_string(eps) + ")",
                  beta, f, Fe, n_levels, dense_n, dense_halfN, dense_tol, conv, saturated_tol);
    }
  }
}

} // namespace

int run_fourier_convergence_tests()
{
  std::cout << "\n--- Fourier-transform convergence tests (analytic references) ---" << std::endl;
  std::cout << "  tau grids: all usable Quadratures (Uniform midpoint, Gauss-Legendre)."
            << "  (ChebyshevNode has no weights -> not a Quadrature -> not usable; skipped.)"
            << std::endl;

  // 1. Uniform midpoint tau grid (existing Story 03 coverage, preserved).
  //    Midpoint rule: O(n^-2) error -> strict decrease under refinement.
  run_all_functions<UniformImaginaryTimeGrid>("uniform",
      Convergence::StrictDecreasing);

  // 2. Gauss-Legendre tau grid (all remaining usable Quadrature grids).
  //    Spectrally accurate: ~1e-12..1e-15 (machine floor) at n >= 16 ->
  //    assert near-machine precision at EVERY level (the strongest
  //    convergence a quadrature rule can show).
  run_all_functions<GaussLegendreImaginaryTimeGrid>("gauss-legendre",
      Convergence::SpectrallySaturated, /*saturated_tol=*/1e-9);

  return 0;
}

} // namespace cppgw
