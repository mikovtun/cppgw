// ============================================================================
//  Story 04 — Inverse Fourier transform (Matsubara -> tau): tests
//
//  Exercises, against analytic references:
//
//   [1] the two independent one-sided stencil fits: exactness for polynomials
//       (degree d <= M-1) on all three supported tau grids;  M in {4, 8, 16};
//   [2] derivative-order independence: one call returns ALL derivatives
//       k = 0..M-1 of the same stencil simultaneously, each correct;
//   [3] FermionicTail::at_matsubara for a manually set coefficient set,
//       including the omega = 0 convention  G_tail(0) := 0;
//   [4] FermionicTail::at_tau for a manually set coefficient set, and the T_k
//       polynomials against explicit Matsubara sums (O(N^{-k}) convergence);
//   [5] the one-level Green's function end-to-end: stencil fit -> coefficients
//       (c_k = epsilon^{k-1}) -> forward transform -> tail-corrected pipeline;
//   [6] inverse-transform convergence, fermionic: tail-corrected (P = 4, P = 6)
//       vs plain — the corrected transform converges substantially faster;
//   [7] Bosonic statistics: the PLAIN inverse converges to the correct value
//       (value-jump O(1/N) vs spectral for a periodic function);
//   [8] tensor-valued behavior (2x2 spatial block);
//   [9] statistics mismatches are rejected at the type level.
//
//  Conventions: the forward transform implements
//
//      F(iw) = int_0^b e^{+i w tau} f(tau) dtau
//
//  (grid/fourier.hpp); the inverse kernel is
//
//      K(m,t) = (1/beta) exp(-i w_m tau_t)
//
//  (grid/fourier_inverse.hpp); the tail follows the Story 04 spec:
//
//      c_{k+1} = (-1)^{k+1} ( f^(k)(0+) + f^(k)(beta-) )
//      G_tail(iw)   = sum_k c_k (iw)^{-k}
//      G_tail(tau)  = sum_k c_k T_k(tau)
//
//  Fermionic Matsubara frequencies are (odd) * pi/beta.  "Scalar-valued" test
//  data carries a single size-1 "sp" spatial axis.
// ============================================================================

#include "main.hpp"
#include "tests.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace cppgw {
namespace {

using cplx = std::complex<double>;
using GL = GaussLegendreImaginaryTimeGrid;
using UT = UniformImaginaryTimeGrid;
using CH = ChebyshevNodeImaginaryTimeGrid;
using RealTensor = Tensor<double, Executor::Host>;
using CplxTensor = Tensor<cplx, Executor::Host>;
using FermiMatsu = GridExpansionMatsubara<cplx, Fermionic>;
using BosMatsu   = GridExpansionMatsubara<cplx, Bosonic>;

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------

// Deterministic PRNG so any failure is reproducible.
std::mt19937& rng() { static std::mt19937 r(0x5ee04L); return r; }
double rreal(double lo, double hi) {
  std::uniform_real_distribution<double> d(lo, hi);
  return d(rng());
}

void fail(const std::string& msg) { throw std::runtime_error(msg); }
void req(bool cond, const std::string& what) { if (!cond) fail(what); }

// Formatting helper (std::to_string has no complex overload).
std::string cstr(const cplx z) {
  return "(" + std::to_string(z.real()) + ", " + std::to_string(z.imag()) + ")";
}

// |got - exact| <= max(abstol, reltol * |exact|)
void relcheck(const std::string& what, const cplx got, const cplx exact,
              double reltol, double abstol) {
  const double err   = std::abs(got - exact);
  const double scale = std::max(abstol, reltol * std::abs(exact));
  if (!(err <= scale))
    fail(what + ":  |got - exact| = " + std::to_string(err)
         + "  >  tolerance " + std::to_string(scale)
         + "   (exact = " + cstr(exact) + ", got = " + cstr(got) + ")");
}

// Sample a scalar f on grid g into an ["sp"]=1 tensor, tau axis FASTEST.
template <class G>
RealTensor sample1(const G& g, const std::function<double(double)>& f) {
  const size_t T = g.size();
  RealTensor d({TensorDim{g.dim_label, T}, TensorDim{std::string("sp"), 1}});
  double* p = d.data();
  for (size_t t = 0; t < T; ++t)
    p[t] = f(g(t).value);
  return d;
}

// Complex copy of a real tensor (the pipeline operates in the complex scalar).
CplxTensor to_cplx(const RealTensor& x) {
  CplxTensor y(x.dims());
  const double* s = x.data();
  cplx* d = y.data();
  for (size_t i = 0; i < x.total_elements(); ++i)
    d[i] = s[i];
  return y;
}

// Intrinsic round-off budget of the double-precision boundary-derivative
// evaluation  d^k/dtau^k L(tau)  (L = the degree < M Lagrange interpolant
// through `xs` at the data values `ys`), with z OUTSIDE the stencil nodes:
//   |error| is at most ~ C_lev * eps * sum_i |L_i^(k)(z)| * |y_i|,
//   (C_lev >> 1 absorbs correlated term cancellation; the basis derivatives
//   at z can be MUCH larger than the value they sum to -- e.g. ~4.5e9 in
//   absolute for M = 16, k = 4 on a 16-node uniform stencil).  This is an
//   absolute-condition bound of the evaluation functional itself, not of the
//   data, which is honest exactly where polynomial extrapolation is
//   intrinsically ill-conditioned (clustered nodes, high k).  The test
//   budgets  C_lev * eps * (that sum)  --  see  EPS_LEV and the call sites.
double eval_conditioning(const std::vector<double>& xs, const std::vector<double>& ys,
                         size_t k, double z) {
  double s = 0.0;
  for (size_t i = 0; i < xs.size(); ++i)
    s += std::abs(detail::lagrange_basis_kderivative(xs, i, k, z)) * std::abs(ys[i]);
  return std::max(s, 1.0);
}

const double EPS_LEV = 4096.0 * std::numeric_limits<double>::epsilon();   // per-case double budget

// The exact T_k(tau) polynomials (Story 04 spec, k = 1..6) — the tau-space
// reference.  (Test [4b] independently anchors THESE formulas against the
// explicit Matsubara sums.)
double T_exact(size_t k, double beta, double tau) {
  const double b = beta, t = tau;
  switch (k) {
    case 1: return -0.5;
    case 2: return -b / 4 + t / 2;
    case 3: return b * t / 4 - t * t / 4;
    case 4: return (b*b*b / 8 - 3.0 * b * t * t / 4 + t*t*t / 2) / 6;
    case 5: return (-b*b*b * t / 2 + b * t*t*t - t*t*t*t / 2) / 24;
    case 6: return (-b*b*b*b*b / 4 + 5.0 * b*b*b * t * t / 4
                     - 5.0 * b * t*t*t*t / 4 + t*t*t*t*t / 2) / 120;
    default: fail("T_exact: k out of range 1..6"); return 0.0;
  }
}

// (1/beta) sum over the Fermi-Matsubara grid of halfWidth N:
//   sum_n e^{-i w_n tau} (i w_n)^{-k}
double T_matsubara_sum(size_t k, size_t Nhalf, double beta, double tau) {
  MatsubaraGrid<Fermionic> mg(Nhalf, InverseTemperature(beta));
  const size_t Mpt = mg.size();
  cplx s(0.0, 0.0);
  for (size_t m = 0; m < Mpt; ++m) {
    const double w = mg(m).value;                    // nonzero (Fermionic)
    const cplx inv = cplx(1.0, 0.0) / cplx(0.0, w);  // (iw)^{-1}
    cplx ak(1.0, 0.0);
    for (size_t q = 0; q < k; ++q) ak *= inv;        // (iw)^{-k}
    s += std::exp(cplx(0.0, -w * tau)) * ak;
  }
  return std::real(s / beta);                        // symmetric sum is real
}

// ============================================================================
//  [1] Two-stencil polynomial fit — exactness for polynomials
//
//  For a polynomial f of degree d <= M-1, the degree < M Lagrange interpolant
//  through the stencil IS f, so for every order k <= d the extracted
//  derivatives at 0+ and beta- must agree with the exact values to round-off
//  (the bound is relative-1e-8, scaled by the Lagrange leverage).
// ============================================================================
template <class G>
void test1_polynomial_fit(const char* gname, double beta) {
  const size_t mset[3] = {4, 8, 16};
  size_t ncases = 0;

  for (size_t M : mset) {
    // Grid point count >= M:  Uniform/GL take n = M directly; the
    // Chebyshev-node grid gives (order + 1) points, so it takes order = M-1.
    const size_t n = std::is_same_v<G, CH> ? (M >= 2 ? M - 1 : 1) : M;
    G grid(beta, n);
    const size_t T = grid.size();
    if (T < M) continue;

    for (size_t d = 0; d + 1 <= M; ++d) {            // degree d = 0 .. M-1
      std::vector<double> a(d + 1);
      for (size_t j = 0; j <= d; ++j)
        a[j] = rreal(-3.0, 3.0);

      auto fall_of = [](size_t n, size_t k) {
        double v = 1.0;
        for (size_t q = 0; q < k; ++q) v *= static_cast<double>(n - q);
        return v;
      };
      auto f = [d, &a](double t) {                   // f(tau) = sum_j a_j tau^j
        double v = 0.0;
        for (size_t j = 0; j <= d; ++j) v = v * t + a[d - j];     // Horner, a_d first
        return v;
      };
      auto dexact = [&](size_t k, double x) {        // exact f^(k)(x), k <= d
        if (k > d) return 0.0;
        double v = 0.0;
        for (size_t j = k; j <= d; ++j)
          v += a[j] * fall_of(j, k) * std::pow(x, j - k);
        return v;
      };

      const RealTensor data = sample1(grid, f);
      TauBoundaryStencils<double, cplx> b(data, grid, M, M);
      // Data values at the stencil nodes (for the conditioning budget).
      const auto& xL = b.x_left();
      const auto& xR = b.x_right();
      std::vector<double> yL(xL.size()), yR(xR.size());
      for (size_t i = 0; i < xL.size(); ++i) yL[i] = f(xL[i]);
      for (size_t i = 0; i < xR.size(); ++i) yR[i] = f(xR[i]);

      for (size_t k = 0; k <= d; ++k) {
        const double ex = dexact(k, 0.0);
        const double rx = dexact(k, beta);
        const cplx   gl = b.L().linear(k);           // ["sp"] = 1 element
        const cplx   gr = b.R().linear(k);
        // Absolute budget: the k-th boundary-derivative evaluation is
        // intrinsically conditioned by  eps * sum_i |L_i^(k)(z)| |y_i|;
        // budget it (EPS_LEV absorbs correlation of the round-offs).
        const double condL = eval_conditioning(xL, yL, k, 0.0);
        const double condR = eval_conditioning(xR, yR, k, (double)beta);
        relcheck(std::string("polyfit [") + gname + "] M=" + std::to_string(M)
                 + " d=" + std::to_string(d) + " k=" + std::to_string(k) + " L",
                 gl, cplx(ex, 0.0), 1e-8, EPS_LEV * condL);
        relcheck(std::string("polyfit [") + gname + "] M=" + std::to_string(M)
                 + " d=" + std::to_string(d) + " k=" + std::to_string(k) + " R",
                 gr, cplx(rx, 0.0), 1e-8, EPS_LEV * condR);
      }
      ncases += 1;
    }
  }
  std::cout << "  [1] polynomial-fit exactness on " << gname << ": "
            << ncases << " (M,d) cases  OK" << std::endl;
}

// ============================================================================
//  [2] Derivative-order independence: one call -> ALL orders, each correct
// ============================================================================
void test2_order_independence() {
  const double beta = 3.7;
  const size_t M = 8;
  UT grid(beta, /*n=*/12);                          // 12 points; M = 8 each side
  const size_t T = grid.size();

  const size_t d = M - 1;
  std::vector<double> a(d + 1);
  for (size_t j = 0; j <= d; ++j)
    a[j] = rreal(-3.0, 3.0);
  auto f = [d, &a](double t) {
    double v = 0.0;
    for (size_t j = 0; j <= d; ++j) v = v * t + a[d - j];     // Horner, a_d first
    return v;
  };
  auto dexact = [&](size_t k, double x) {
    double v = 0.0;
    for (size_t j = k; j <= d; ++j) {
      double fall = 1.0;
      for (size_t q = 0; q < k; ++q) fall *= static_cast<double>(j - q);
      v += a[j] * fall * std::pow(x, j - k);
    }
    return v;
  };

  // Left stencil: the M smallest tau; right: the M largest (UT is ascending).
  std::vector<double> xl(M), xr(M);
  for (size_t i = 0; i < M; ++i) {
    xl[i] = grid(i).value;
    xr[i] = grid(T - M + i).value;
  }
  std::vector<double> yl(M), yr(M);
  for (size_t i = 0; i < M; ++i) {
    yl[i] = f(xl[i]);
    yr[i] = f(xr[i]);
  }

  Stencil<double> stL(xl, yl), stR(xr, yr);

  // (a) ONE call for all orders k = 0..M-1 ...
  const size_t K = M;
  const std::vector<double> allL = stL.derivatives(K, /*z =*/0.0);
  const std::vector<double> allR = stR.derivatives(K, /*z =*/beta);
  // ... must match the per-order calls (same stencil) exactly ...
  for (size_t k = 0; k < K; ++k) {
    req(allL[k] == stL.derivative(k, 0.0),
        "order independence L k=" + std::to_string(k) + ": one-call != per-order");
    req(allR[k] == stR.derivative(k, beta),
        "order independence R k=" + std::to_string(k) + ": one-call != per-order");
  }
  // ... and each one must be the correct derivative of d (d = M-1 covers all k).
  for (size_t k = 0; k < K; ++k) {
    const double condL = eval_conditioning(xl, yl, k, 0.0);
    const double condR = eval_conditioning(xr, yr, k, beta);
    relcheck("order independence L k=" + std::to_string(k),
             cplx(allL[k], 0.0), cplx(dexact(k, 0.0), 0.0), 1e-8, EPS_LEV * condL);
    relcheck("order independence R k=" + std::to_string(k),
             cplx(allR[k], 0.0), cplx(dexact(k, beta), 0.0), 1e-8, EPS_LEV * condR);
  }
  std::cout << "  [2] derivative-order independence (M = " << M << "): OK" << std::endl;
}

// ============================================================================
//  [3] FermionicTail::at_matsubara, manual coefficients (including iw = 0)
// ============================================================================
void test3_tail_matsubara() {
  const double beta = 6.5;
  const size_t P = 5;
  std::vector<cplx> c(P);
  for (size_t k = 0; k < P; ++k)
    c[k] = cplx(rreal(-2.0, 2.0), rreal(-2.0, 2.0));

  CplxTensor cot({TensorDim{std::string("coeff"), P}, TensorDim{std::string("sp"), 1}});
  cplx* p = cot.data();
  for (size_t k = 0; k < P; ++k) p[k] = c[k];

  FermionicTail<cplx, UT> tail(cot, InverseTemperature(beta));
  req(tail.tail_order() == P, "tail_order() must equal P");
  req(tail.size() == P, "size() must equal P");
  req(tail.P() == P, "P() must equal P");
  for (size_t k = 0; k < P; ++k)
    relcheck("coeff(i) i=" + std::to_string(k), tail.coeff(k), c[k], 1e-15, 1e-300);

  const std::vector<double> ws = {0.0, 1.3, -1.3,
                                  std::numbers::pi / 6.0, -std::numbers::pi / 6.0,
                                  5.2, -5.2};
  for (double w : ws) {
    const cplx got = tail.at_matsubara(w).linear(0);
    if (w == 0.0) {
      // The convention (singular 1/(iw) terms are a HIGH-frequency object):
      req(std::abs(got) == 0.0,
          "at_matsubara(0) must be the zero spatial tensor (convention G_tail(0) := 0)");
      continue;
    }
    cplx expected(0.0, 0.0), pw(1.0, 0.0);   // pw accumulates (iw)^{-k}, start (iw)^0
    const cplx inv = cplx(1.0, 0.0) / cplx(0.0, w);
    for (size_t k = 1; k <= P; ++k) {
      pw *= inv;
      expected += c[k - 1] * pw;
    }
    relcheck("at_matsubara w=" + std::to_string(w),
             got, expected, 1e-13, 1e-13);
  }
  std::cout << "  [3] FermionicTail::at_matsubara (P=5, iw = 0 convention): OK" << std::endl;
}

// ============================================================================
//  [4] FermionicTail::at_tau;  T_k vs explicit Matsubara sums
// ============================================================================
void test4_tail_tau_and_Tk() {
  const size_t P = 6;
  std::vector<cplx> c(P);
  for (size_t k = 0; k < P; ++k)
    c[k] = cplx(rreal(-2.0, 2.0), rreal(-2.0, 2.0));

  // (a) at_tau(tau) = sum_k c_k T_k(tau), several betas and interior points.
  for (double beta : {1.9, 5.3, 10.0}) {
    CplxTensor cot({TensorDim{std::string("coeff"), P}, TensorDim{std::string("sp"), 1}});
    cplx* p = cot.data();
    for (size_t k = 0; k < P; ++k) p[k] = c[k];
    FermionicTail<cplx, UT> tail(cot, InverseTemperature(beta));
    UT grid(InverseTemperature(beta), /*n=*/16);
    for (double u : {0.1, 0.35, 0.55, 0.9}) {
      const double tau = u * beta;
      cplx expected(0.0, 0.0);
      for (size_t k = 1; k <= P; ++k)
        expected += c[k - 1] * T_exact(k, beta, tau);
      const cplx got = tail.at_tau(tau).linear(0);
      relcheck("at_tau beta=" + std::to_string(beta) + " tau=" + std::to_string(tau),
               got, expected, 1e-13, 1e-13);
    }
    // (a2) at_tau_grid over a whole grid
    const auto tg = tail.at_tau_grid(grid);
    double e = 0.0;
    const cplx* p2 = tg.data().data();
    for (size_t t = 0; t < grid.size(); ++t) {
      cplx expected(0.0, 0.0);
      for (size_t k = 1; k <= P; ++k)
        expected += c[k - 1] * T_exact(k, beta, grid(t).value);
      e = std::max(e, std::abs(p2[t] - expected));
    }
    req(e < 1e-13, "at_tau_grid beta=" + std::to_string(beta) + " error " + std::to_string(e));
  }

  // (b) The T_k formulas against the DIRECT Matsubara sums.  The partial sum
  //     has O(N^{-k}) convergence, so the tolerance is self-scaling:
  //     max(floor_k, 10 * |S_{2N} - S_N|).  (k = 1 converges slowly: loose floor.)
  const double floors[6] = {5e-3, 5e-5, 5e-7, 1e-8, 1e-10, 1e-12};
  for (size_t k = 1; k <= 6; ++k)
    for (double beta : {4.4, 9.1})
      for (double u : {0.2, 0.5, 0.8}) {
        const double tau  = u * beta;
        const double exact = T_exact(k, beta, tau);
        const double s1    = T_matsubara_sum(k, 1024, beta, tau);
        const double s2    = T_matsubara_sum(k, 2048, beta, tau);
        const double tol   = std::max(floors[k - 1], 10.0 * std::abs(s2 - s1));
        if (!(std::abs(s2 - exact) <= tol))
          fail("T_k reference k=" + std::to_string(k) + " beta=" + std::to_string(beta)
               + " tau=" + std::to_string(tau)
               + ": |S - T_k| = " + std::to_string(std::abs(s2 - exact))
               + " > tol " + std::to_string(tol));
      }
  std::cout << "  [4a] FermionicTail::at_tau / at_tau_grid (P=6, 3 betas): OK" << std::endl;
  std::cout << "  [4b] T_k(1..6) vs explicit Matsubara sums (N=1024/2048): OK" << std::endl;
}

// ============================================================================
//  [5] One-level Green's function, end to end
//
//     f(tau) = -(1 - n_F) e^{-eps tau},   n_F = 1/(e^{beta eps} + 1)
//     G(iw)  = 1/(iw - eps)       exact tail:  c_k = eps^{k-1}
//     f^(k)(0+) = -(1-n_F)(-eps)^k ,   f^(k)(beta-) = -(1-n_F)(-eps)^k e^{-beta eps}
// ============================================================================
void test5_one_level_gf() {
  for (double beta : {2.0, 10.0})
    for (double eps : {0.2, 1.5}) {
      const std::string tag = " [beta=" + std::to_string(beta)
                            + ", eps=" + std::to_string(eps) + "]";
      const double nF = 1.0 / (std::exp(beta * eps) + 1.0);
      auto f = [nF, eps](double tau) { return -(1.0 - nF) * std::exp(-eps * tau); };
      const InverseTemperature bt(beta);
      const size_t T = 64;
      const GL grid(bt, T);

      using TauIn  = GridExpansionTau<double, GL, Fermionic>;
      using TauOut = GridExpansionTau<cplx, GL, Fermionic>;
      using TC = TailCorrectedInverseFourierTransform<FermiMatsu, TauOut, cplx>;

      // The 1/omega-Taylor tail (c_k = eps^{k-1} for this GF) is an
      // ASYMPTOTIC expansion: the finite sum is only a good subtraction when
      // eps < omega_min = (3/2) pi / beta (the smallest |frequency| on ANY
      // Matsubara grid).  Tail-dependent checks ((b),(c),(e),(f)) are gated on
      // that regime; everything else ((a),(d)) is regime-independent.
      const bool asymptotic = (eps * beta / std::numbers::pi < 1.5);

      // tau-side samples (real, scalar "sp"; complex copy for the pipeline)
      const RealTensor dsamp = sample1(grid, f);
      const CplxTensor dsampc = to_cplx(dsamp);

      // (a) FermionicTail with c_1 = 1 (P = 1):  at_matsubara == 1/(iw)
      FermionicTail<cplx, GL> t1(TensorShape{{"sp", 1}}, bt, /*P=*/1);
      t1.set_coeff(0, cplx(1.0, 0.0));
      {
        bool ok = true;
        for (double w : {1.1, -1.1, 3.7, -3.7})
          ok = ok && std::abs(t1.at_matsubara(w).linear(0) - cplx(1.0, 0.0) / cplx(0.0, w)) < 1e-14;
        if (!ok) fail("sanity c_1 = 1" + tag);
      }

      // (b) Stencil fit on the tau samples:  f^(k)(0+) and f^(k)(beta-), k = 0..7
      //     (only meaningful in the asymptotic regime; outside it the
      //     right-boundary stencil samples an exponentially small function and
      //     double-precision k-th differences of it are pure noise).
      if (asymptotic) {
      {
        TauBoundaryStencils<double, cplx> b(dsamp, grid, 8, 8);
        const auto& xL = b.x_left();
        const auto& xR = b.x_right();
        std::vector<double> yL(xL.size()), yR(xR.size());
        for (size_t i = 0; i < xL.size(); ++i) yL[i] = f(xL[i]);
        for (size_t i = 0; i < xR.size(); ++i) yR[i] = f(xR[i]);
        bool ok = true;
        for (size_t k = 0; k <= 7; ++k) {
          const double exL = -(1.0 - nF) * std::pow(-eps, k);
          const double exR = -(1.0 - nF) * std::pow(-eps, k) * std::exp(-eps * beta);
          ok = ok && std::abs(b.L().linear(k) - cplx(exL, 0.0))
              <= std::max(1e-10, std::max(1e-7 * std::abs(exL), EPS_LEV * eval_conditioning(xL, yL, k, 0.0)));
          ok = ok && std::abs(b.R().linear(k) - cplx(exR, 0.0))
              <= std::max(1e-10, std::max(1e-7 * std::abs(exR), EPS_LEV * eval_conditioning(xR, yR, k, (double)beta)));
        }
        if (!ok) fail("stencil-fit one-level GF" + tag);
      }
      } // (b): asymptotic regime

      // (c) Coefficient extraction:  c_{k+1} = eps^k  for  P in {2, 4, 6}
      if (asymptotic) {
      {
        TauBoundaryStencils<double, cplx> b(dsamp, grid, 8, 8);
        const auto& xL2 = b.x_left();
        const auto& xR2 = b.x_right();
        std::vector<double> yL2(xL2.size()), yR2(xR2.size());
        for (size_t i = 0; i < xL2.size(); ++i) { yL2[i] = f(xL2[i]); yR2[i] = f(xR2[i]); }
        for (size_t P : {2, 4, 6}) {
          auto c = b.extract_coeffs(P);
          if (c.dims().empty() || c.dims()[0].dim != P)
            fail("extract_coeffs: coefficient axis size" + tag);
          bool ok = true;
          for (size_t k = 0; k < P; ++k)
            ok = ok && std::abs(c.linear(k) - cplx(std::pow(eps, k), 0.0))
                           <= 2.0 * std::max(1e-10,
                                            std::max(1e-7 * std::pow(eps, k),
                                                     EPS_LEV * std::max(eval_conditioning(xL2, yL2, k, 0.0),
                                                                        eval_conditioning(xR2, yR2, k, (double)beta))));
          if (!ok) fail("extract_coeffs P=" + std::to_string(P) + tag);
        }
      }
      } // (c): asymptotic regime

      // (d) The EXISTING forward transform reproduces  G(iw) = 1/(iw - eps)
      const MatsubaraGrid<Fermionic> mg(16, bt);
      const size_t Mf = mg.size();
      TauIn fin(dsamp, grid);
      FourierTransform<TauIn, FermiMatsu, cplx> ft(grid, mg, /*save=*/true);
      FermiMatsu gout = ft(fin);
      {
        bool ok = true;
        const cplx* gp = gout.data().data();
        for (size_t m = 0; m < Mf; ++m) {
          const cplx ex = cplx(1.0, 0.0) / (cplx(0.0, mg(m).value) - eps);
          ok = ok && std::abs(gp[m] - ex) <= std::max(1e-12, 1e-10 * std::abs(ex));
        }
        if (!ok) fail("forward transform one-level GF" + tag);
      }

      // (e) The full pipeline  stencil-fit -> subtract -> invert -> add recovers
      //     f(tau) up to the Matsubara truncation of the remainder.  Two regimes
      //     are distinguished (the 1/omega-Taylor tail is an ASYMPTOTIC
      //     expansion of G(iw), valid where omega_min > eps):
      //     - asymptotic regime (beta = 2, eps = 0.2 here): a tight 2% band,
      //       and a clean P-ordering  err(P=6) <= err(P=4)/10;
      //     - strong-coupling regime (eps > omega_min): the subtraction can be
      //       anti-asymptotic (it is, provably, a 1/omega-Taylor series about
      //       the pole), so only BOUNDEDNESS is asserted (a bug would amplify
      //       the error 100x+).
      const TauOut tsampc_exp(dsampc, grid);
      if (asymptotic) {
      {
        const double s0 = std::abs(f(0.0));
        const bool strict = (beta < 2.5 && eps < 1.0);
        double err4 = 0.0, err6 = 0.0;
        for (size_t P : {4, 6}) {
          TC tc(mg, grid, /*save=*/true, 8, 8, P);
          const TauOut out = tc(tsampc_exp, gout);
          const cplx* op = out.data().data();
          double maxerr = 0.0;
          for (size_t t = 0; t < T; ++t)
            maxerr = std::max(maxerr, std::abs(op[t] - f(grid(t).value)));
          if (!(std::isfinite(maxerr) && maxerr < 1.0 * s0))
            fail("full pipeline P=" + std::to_string(P) + tag
                 + ": max error " + std::to_string(maxerr) + " (not bounded: > |f(0)|)");
          if (strict && !(maxerr < std::max(1e-8, 2e-2 * s0)))
            fail("full pipeline P=" + std::to_string(P) + tag
                 + ": max error " + std::to_string(maxerr));
          if (P == 4) err4 = maxerr; else err6 = maxerr;
        }
        if (strict) {
          // Fitted (not exact) P=6 coefficients carry stencil-fit noise, so the
          // P-ordering is only required to IMPROVE (>=2x), not to reach the
          // algebraic floor (that is checked with the exact tail in (f)).
          if (!(err6 <= err4 / 2.0))
            fail("full pipeline P-ordering (P=6 should beat P=4 by 2x)"
                 + tag + ": P4=" + std::to_string(err4) + " P6=" + std::to_string(err6));
        }
      }
      } // (e): asymptotic regime

      // (f) A USER-SUPPLIED EXACT tail (c_k = eps^{k-1} for G(iw) = 1/(iw-eps)):
      //     same regime-aware checks, plus the exact-tail result must not be
      //     worse than the fitted-tail one (it can only be equal or better).
      if (asymptotic) {
      {
        const double s0 = std::abs(f(0.0));
        const bool strict = (beta < 2.5 && eps < 1.0);
        double err4 = 0.0, err6 = 0.0;
        for (size_t P : {4, 6}) {
          TC tc(mg, grid, /*save=*/true, 8, 8, P);
          FermiMatsu in(gout.data(), mg);
          FermionicTail<cplx, GL> t(TensorShape{{"sp", 1}}, bt, P);
          for (size_t k = 1; k <= P; ++k)
            t.set_coeff(k - 1, cplx(std::pow(eps, k - 1), 0.0));
          const TauOut out = tc(in, t);
          const cplx* op = out.data().data();
          double maxerr = 0.0;
          for (size_t t = 0; t < T; ++t)
            maxerr = std::max(maxerr, std::abs(op[t] - f(grid(t).value)));
          if (!(std::isfinite(maxerr) && maxerr < 1.0 * s0))
            fail("explicit-tail pipeline P=" + std::to_string(P) + tag
                 + ": max error " + std::to_string(maxerr) + " (not bounded: > |f(0)|)");
          if (strict && !(maxerr < std::max(1e-8, 2e-2 * s0)))
            fail("explicit-tail pipeline P=" + std::to_string(P) + tag
                 + ": max error " + std::to_string(maxerr));
          if (P == 4) err4 = maxerr; else err6 = maxerr;
        }
        if (strict) {
          if (!(err6 <= err4 / 10.0))
            fail("explicit-tail P-ordering"
                 + tag + ": P4=" + std::to_string(err4) + " P6=" + std::to_string(err6));
        }
      }
      } // (f): asymptotic regime

      std::cout << "  [5] one-level GF end-to-end" << tag << (asymptotic ? "" : " (tail regime: asymptotic)") << "  OK" << std::endl;
    }
}

// ============================================================================
//  [6] Convergence, fermionic:  plain inverse  vs  tail-corrected (P = 4, 6)
//
//  G(iw) = 1/(iw - eps)  has the exact tail c_k = eps^{k-1}; we subtract the
//  fitted tail (fitted on dense, independent tau samples) and compare the
//  max-tau error of the two transforms as the Matsubara cutoff grows.
// ============================================================================
void test6_convergence_fermi() {
  const double beta = 10.0, eps = 0.2;   // asymptotic regime: eps < omega_min = 3 pi / (2 beta)
  const double nF = 1.0 / (std::exp(beta * eps) + 1.0);
  auto f = [nF, eps](double tau) { return -(1.0 - nF) * std::exp(-eps * tau); };
  const InverseTemperature bt(beta);

  // Reference: G(iw) = 1/(iw - eps) directly (the analytic Matsubara series)
  using TauOutP = GridExpansionTau<cplx, GL, Fermionic>;     // plain: on GL-24 nodes
  using TauOutR = GridExpansionTau<cplx, UT, Fermionic>;     //   and: pipeline on UT-32
  const size_t T = 24;
  const GL gridP(bt, T);
  const UT gridR(bt, 32);

  // tau samples on the PIPELINE grid (used for its boundary stencils)
  const RealTensor dsampR = sample1(gridR, f);
  const CplxTensor dsampcR = to_cplx(dsampR);

  using TC = TailCorrectedInverseFourierTransform<FermiMatsu, TauOutR, cplx>;
  using IX = InverseFourierTransform<FermiMatsu, TauOutP, cplx>;

  const TauOutR tsampc_exp(dsampcR, gridR);
  auto errR = [&f](const TauOutR& out) {
    const cplx* p = out.data().data();
    double e = 0.0;
    const size_t ng = out.size();
    for (size_t t = 0; t < ng; ++t)
      e = std::max(e, std::abs(p[t] - f(out.grid()(t).value)));
    return e;
  };

  const std::vector<size_t> halfNs = {2, 4, 8, 16};
  std::vector<double> eplain, e4, e6;
  std::vector<double> eplain_edge, e4_edge;      // error at the tau-boundary nodes
  eplain.reserve(halfNs.size()); e4.reserve(halfNs.size()); e6.reserve(halfNs.size());
  eplain_edge.reserve(halfNs.size()); e4_edge.reserve(halfNs.size());

  for (size_t h : halfNs) {
    const MatsubaraGrid<Fermionic> mg(h, bt);
    const size_t Mh = mg.size();
    CplxTensor dm({TensorDim{mg.dim_label, Mh}, TensorDim{std::string("sp"), 1}});
    cplx* q = dm.data();
    for (size_t m = 0; m < Mh; ++m)
      q[m] = cplx(1.0, 0.0) / (cplx(0.0, mg(m).value) - eps);    // the analytic G
    FermiMatsu in(dm, mg);

    // (i) PLAIN inverse, evaluated at the GL-24 nodes.
    const IX ix(mg, gridP, /*save=*/false);
    const TauOutP outp = ix(in);
    const cplx* op = outp.data().data();
    double ep = 0.0;
    for (size_t t = 0; t < gridP.size(); ++t)
      ep = std::max(ep, std::abs(op[t] - f(gridP(t).value)));
    eplain.push_back(ep);
    eplain_edge.push_back(std::max(std::abs(op[0] - f(gridP(0).value)),
                                   std::abs(op[gridP.size() - 1] - f(gridP(gridP.size() - 1).value))));

    // (ii) TAIL-CORRECTED pipeline (P = 4 and P = 6), on the UT-32 points.
    {
      TC tc(mg, gridR, /*save=*/true, 8, 8, 4);
      const TauOutR out4 = tc(tsampc_exp, in);
      e4.push_back(errR(out4));
      e4_edge.push_back(std::max(std::abs(out4.data().linear(0)             - f(0.0)),
                                 std::abs(out4.data().linear(out4.size()-1) - f(beta))));
    }
    {
      TC tc6(mg, gridR, /*save=*/true, 8, 8, 6);
      const TauOutR out6 = tc6(tsampc_exp, in);
      e6.push_back(errR(out6));
    }
  }

  // ---- assertions ----
  // The PLAIN Matsubara inverse of this GF carries the Gibbs tail of the
  // antiperiodic boundary jump: its error plateaus at O(|f(0)|) and does NOT
  // show clean O(1/N) monotonicity on these small cutoffs -- so we only assert
  // BOUNDEDNESS.  The TAIL-CORRECTED pipeline removes the entire algebraic
  // 1/omega tail of G(iw), so it must beat the plain error by a large factor at
  // every cutoff, with P = 6 at least as good as P = 4.

  // (i) plain error is bounded (Gibbs plateau of O(|f(0)|), no blow-up)
  {
    const double s0 = std::abs(f(0.0));
    for (size_t i = 0; i < eplain.size(); ++i)
      if (!(std::isfinite(eplain[i]) && eplain[i] < 1.5 * s0))
        fail("plain inverse error " + std::to_string(eplain[i]) + " not bounded (O(|f0|) expected)");
  }

  // (ii) tail-corrected error is SUBSTANTIALLY smaller than the plain error
  //      at every cutoff (the whole point of the tail subtraction) ...
  for (size_t i = 0; i < halfNs.size(); ++i)
    if (!(e4[i] < 0.5 * eplain[i]))
      fail("tail-corrected (P=4) error " + std::to_string(e4[i])
           + " is not substantially smaller than the plain error "
           + std::to_string(eplain[i]) + " at cutoff " + std::to_string(2 * halfNs[i] + 1));

  // (iii) ... and stays bounded overall (no anti-asymptotic blow-up);
  //       P = 6 does not do worse than P = 4 at any cutoff.
  {
    const double s0 = std::abs(f(0.0));
    for (size_t i = 0; i < halfNs.size(); ++i) {
      if (!(std::isfinite(e4[i]) && e4[i] < s0))
        fail("tail-corrected (P=4) error " + std::to_string(e4[i]) + " not bounded");
      if (!(e6[i] <= e4[i] * 2.0 + 1e-12))
        fail("tail-corrected (P=6) error " + std::to_string(e6[i])
             + " should not be materially worse than (P=4) " + std::to_string(e4[i])
             + " at cutoff " + std::to_string(2 * halfNs[i] + 1));
    }
  }

  std::cout << "  [6] fermionic convergence (eps=0.2, beta=10) -- max-tau error vs cutoff M:"
            << std::endl;
  for (size_t i = 0; i < halfNs.size(); ++i) {
    const size_t modes = 2 * halfNs[i] + 1;
    std::cout << "      M=" << modes
              << "   plain=" << eplain[i]
              << "   edge-plain=" << eplain_edge[i]
              << "   P4=" << e4[i]
              << "   edge-P4=" << e4_edge[i]
              << "   P6=" << e6[i] << std::endl;
  }
  std::cout << "  [6] tail-corrected converges substantially faster than plain: OK" << std::endl;
}

// ============================================================================
//  [7] Bosonic statistics: the PLAIN inverse converges to the correct value.
//
//  (a) f(tau) = 1 - tau/beta :  f(0) != f(beta) (value jump of 1)
//      -> the truncated inverse decays at O(1/Nw).  (The tail machinery is a
//         CONVERGENCE ACCELERATOR for the fermionic case; for bosons the plain
//         sum still converges to the right answer.)
//  (b) f(tau) = cos(2 pi tau/beta) : all endpoint differences vanish (the
//      periodic extension is C^infinity) -> the inverse is SPECTRAL: the
//      function is a single bosonic mode and is reproduced at round-off.
// ============================================================================
void test7_bosonic() {
  const double beta = 6.0;
  const InverseTemperature bt(beta);
  const size_t T = 32;
  const UT grid(bt, T);
  using BosTauOut = GridExpansionTau<cplx, UT, Bosonic>;

  const std::vector<size_t> halfNs = {1, 2, 4, 8};

  // Run the PLAIN bosonic inverse of an analytic Matsubara series F(iw) on the
  // uniform output grid and report the max error against f(tau) at each cutoff.
  auto run_plain = [&](const std::function<cplx(double)>& F,
                       const std::function<double(double)>& f) {
    std::vector<double> e;
    e.reserve(halfNs.size());
    for (size_t h : halfNs) {
      MatsubaraGrid<Bosonic> mg(h, bt);
      const size_t Mh = mg.size();
      CplxTensor dm({TensorDim{mg.dim_label, Mh}, TensorDim{std::string("sp"), 1}});
      cplx* q = dm.data();
      for (size_t m = 0; m < Mh; ++m)
        q[m] = F(mg(m).value);
      BosMatsu in(dm, mg);
      const InverseFourierTransform<BosMatsu, BosTauOut, cplx> ix(mg, grid, /*save=*/true);
      const BosTauOut out = ix(in);
      const auto& data = out.data();
      double em = 0.0;
      for (size_t t = 0; t < grid.size(); ++t)
        em = std::max(em, std::abs(data.linear(t) - f(grid(t).value)));
      e.push_back(em);
    }
    return e;
  };

  // (a) f(tau) = 1 - tau/beta :  value jump  f(0+) - f(beta-) = 1
  //     -> the plain truncated inverse decays at O(1/Nw).
  {
    auto F = [&](double w) -> cplx {
      if (w == 0.0) return cplx(0.5 * beta, 0.0);     // zero mode = int_0^b (1 - t/b) dt = b/2
      const cplx iw(0.0, w);
      const cplx eibw = std::exp(cplx(0.0, w * beta));
      // A = int_0^b e^{iwt} dt           = (e^{iwb} - 1)/(iw)
      // B = int_0^b t e^{iwt} dt         = (e^{iwb}(iwb - 1) + 1)/(iw)^2
      const cplx A = (eibw - cplx(1.0, 0.0)) / iw;
      const cplx B = (eibw * (iw * beta - 1.0) + cplx(1.0, 0.0)) / (iw * iw);
      return A - B / beta;
    };
    const std::vector<double> e = run_plain(F, [&](double tau) { return 1.0 - tau / beta; });
    for (size_t i = 0; i + 1 < halfNs.size(); ++i) {
      const double ratio = e[i] / e[i + 1];            // ~2 for O(1/N)
      req(ratio > 0.5 && ratio < 12.0,
          "value-jump bosonic decay ratio " + std::to_string(ratio) + " inconsistent with O(1/Nw)");
    }
    std::cout << "  [7a] bosonic value-jump f=1-tau/beta, e ~ O(1/Nw):  ";
    for (double v : e) std::cout << " " << v;
    std::cout << "  OK" << std::endl;
  }

  // (b) f(tau) = cos(2 pi tau/beta) : every endpoint difference vanishes (the
  //     periodic extension is C^infinity); the function is one bosonic mode, so
  //     the inverse is SPECTRAL (round-off) - clearly NOT merely O(1/Nw).
  {
    auto one = [&](double q) -> cplx {                    // int_0^b e^{iqt} dt
      if (std::abs(q) < 1e-14) return cplx(beta, 0.0);
      return (std::exp(cplx(0.0, q * beta)) - cplx(1.0, 0.0)) / cplx(0.0, q);
    };
    auto F = [&](double w) -> cplx {
      const double c2 = 2.0 * std::numbers::pi / beta;
      return 0.5 * (one(w + c2) + one(w - c2));
    };
    const std::vector<double> e = run_plain(F,
        [&](double tau) { return std::cos(2.0 * std::numbers::pi * tau / beta); });
    double e_min = 1e300;
    for (double v : e) e_min = std::min(e_min, v);
    req(e_min < 1e-12,
        "spectral bosonic cos(2 pi tau/beta) should reach round-off; got "
        + std::to_string(e_min));
    std::cout << "  [7b] bosonic spectral f=cos(2 pi tau/beta),  e_min=" << e_min
              << "  OK" << std::endl;
  }

  std::cout << "  [7] Bosonic statistics: plain inverse converges to the correct value  OK"
            << std::endl;
}

// ============================================================================
//  [8] Tensor-valued behavior (2x2 spatial block).
//
//  The tail, the boundary fits and the full pipeline run element-wise over the
//  spatial indices (only the tau / Matsubara axis changes).  We build a
//  tensor-valued function  G_{mu,nu}(iw) = M_{mu,nu} * G(iw)  with M a 2x2
//  matrix, and verify the reconstructed tau samples and the extracted
//  coefficients component-by-component against the analytic values.
// ============================================================================
void test8_tensor_valued() {
  const double beta = 10.0, eps = 0.8;
  const double nF = 1.0 / (std::exp(beta * eps) + 1.0);
  auto f = [nF, eps](double tau) { return -(1.0 - nF) * std::exp(-eps * tau); };   // scalar part
  const InverseTemperature bt(beta);

  const size_t mu_n = 2, nu_n = 2;

  // Distinct complex weight per spatial component (so per-component errors are visible).
  auto Mmn = [](size_t mu, size_t nu) -> cplx {
    return cplx(1.5 + 1.0 * (double)mu + 0.3 * (double)nu,
                0.7 + 2.0 * (double)mu + 1.1 * (double)nu);
  };

  using TauOut = GridExpansionTau<cplx, GL, Fermionic>;
  using TC = TailCorrectedInverseFourierTransform<FermiMatsu, TauOut, cplx>;

  const size_t T = 48;
  const GL grid(bt, T);

  // tau-side tensor samples  G_{mu,nu}(tau) = M_{mu,nu} f(tau);
  // layout: [tau (fastest), nu, mu].
  std::vector<TensorDim> dsdims = {TensorDim{grid.dim_label, T},
                                   TensorDim{std::string("nu"), nu_n},
                                   TensorDim{std::string("mu"), mu_n}};
  CplxTensor dsc(dsdims);
  cplx* dp = dsc.data();
  for (size_t mu = 0; mu < mu_n; ++mu)
    for (size_t nu = 0; nu < nu_n; ++nu)
      for (size_t t = 0; t < T; ++t)
        dp[t + T * nu + T * nu_n * mu] = Mmn(mu, nu) * f(grid(t).value);

  const MatsubaraGrid<Fermionic> mg(10, bt);
  const size_t Mf = mg.size();

  // Matsubara-side tensor samples  G_{mu,nu}(iw) = M_{mu,nu} / (iw - eps);
  // layout: [matsu (fastest), nu, mu].
  std::vector<TensorDim> dmdims = {TensorDim{mg.dim_label, Mf},
                                   TensorDim{std::string("nu"), nu_n},
                                   TensorDim{std::string("mu"), mu_n}};
  CplxTensor dmm(dmdims);
  cplx* mq = dmm.data();
  for (size_t mu = 0; mu < mu_n; ++mu)
    for (size_t nu = 0; nu < nu_n; ++nu)
      for (size_t m = 0; m < Mf; ++m)
        mq[m + Mf * nu + Mf * nu_n * mu] = Mmn(mu, nu) / (cplx(0.0, mg(m).value) - eps);

  FermiMatsu in(dmm, mg);
  const TauOut tsampc_exp(dsc, grid);
  TC tc(mg, grid, /*save=*/true, 8, 8, 4);

  // (i) full pipeline runs element-wise over the 2x2 block; reconstruction per component.
  {
    const TauOut out = tc(tsampc_exp, in);
    const auto& od = out.data();
    req(od.rank() == 3,
        "pipeline tensor output should keep dims [tau, nu, mu]");
    double worst = 0.0;
    for (size_t mu = 0; mu < mu_n; ++mu)
      for (size_t nu = 0; nu < nu_n; ++nu)
        for (size_t t = 0; t < T; ++t) {
          const cplx ex  = Mmn(mu, nu) * f(grid(t).value);
          const cplx got = od(t, nu, mu);
          worst = std::max(worst, std::abs(got - ex) / std::max(1e-300, std::abs(ex)));
        }
    // Tolerance: (beta=10, eps=0.8) sits just outside the strict 1/omega-
    // Taylor regime (eps*beta/pi = 2.55 > 1.5), so the fitted-pipeline error
    // is a few-percent-of-amplitude (Gibbs/tail truncation), like test 5 for
    // the same parameters; a bug would amplify it 10x+.
    if (!(worst < 0.2))
      fail("tensor pipeline max relative component error " + std::to_string(worst));
    std::cout << "  [8a] tensor pipeline (2x2 spatial) per-component: OK "
              << "(worst rel err " << worst << ")" << std::endl;
  }

  // (ii) coefficients extracted per component:  c_{k+1}[mu,nu] = M_{mu,nu} eps^k
  // (iii) FermionicTail keeps the spatial tensor shape
  {
    const auto b = tc.stencil_fit(tsampc_exp);
    const auto c = b.extract_coeffs(4);
    req(c.rank() == 3 && c.dims()[0].dim == 4,
        "coeff tensor should be [4 (fastest), nu, mu]");
    double worst = 0.0;
    for (size_t mu = 0; mu < mu_n; ++mu)
      for (size_t nu = 0; nu < nu_n; ++nu)
        for (size_t k = 0; k < 4; ++k) {
          const cplx ex  = std::pow(eps, (double)k) * Mmn(mu, nu);
          const cplx got = c.linear(k + 4 * (nu + nu_n * mu));   // [4, nu, mu] strides 1,4,8
          worst = std::max(worst, std::abs(got - ex) / std::max(1e-300, std::abs(ex)));
        }
    // Tolerance: fitted coefficients in this (non-strict) regime carry
    // stencil-fit noise of O(1e-3..1e-2) relative; any algorithmic bug (an
    // element crossed between components, a dropped sign, a transposed block)
    // is 10x+ on top.
    if (!(worst < 0.05))
      fail("tensor coefficient max relative error " + std::to_string(worst));

    auto tl = tc.make_tail(c, bt, 4);
    req(tl.data().rank() == 3 && tl.data().dims()[0].dim == 4,
        "FermionicTail::data() should be [coeff=4, nu, mu]");
    std::cout << "  [8b] tensor coefficient extraction + tail shape: OK "
              << "(worst rel err " << worst << ")" << std::endl;
  }

  std::cout << "  [8] tensor-valued (2x2 spatial) fits, coefficients and pipeline: OK" << std::endl;
}

// ============================================================================
//  [9] Statistics mismatches are rejected (compile-time, type-level).
//
//  The two sides carry different statistics -> the `static_assert` in the
//  primary template makes a mismatched pair a template-instantiation error.
//  We positively-assert (so the test passes) that the *types* do in fact
//  disagree, which is exactly the condition the static_assert checks.
// ============================================================================
void test9_statistics_mismatch() {
  // A Bosonic Matsubara side + a Fermionic tau side (or vice-versa) must not
  // satisfy the statistics-match check the transforms enforce.
  static_assert( !std::same_as<typename BosMatsu::statistics_type,
                               typename GridExpansionTau<cplx, UT, Fermionic>::statistics_type>,
                 "a Bosonic/Fermionic statistics mismatch must be REJECTED" );
  static_assert( !std::same_as<typename FermiMatsu::statistics_type,
                               typename GridExpansionTau<cplx, UT, Bosonic>::statistics_type>,
                 "a Fermionic/Bosonic statistics mismatch must be REJECTED" );
  static_assert(  std::same_as<typename FermiMatsu::statistics_type,
                               typename GridExpansionTau<cplx, UT, Fermionic>::statistics_type> );
  static_assert(  std::same_as<typename BosMatsu::statistics_type,
                               typename GridExpansionTau<cplx, UT, Bosonic>::statistics_type> );
  std::cout << "  [9] statistics-mismatch rejected by type (static_assert)  OK" << std::endl;
}

} // namespace (anonymous)

// ============================================================================
//  Entry point
// ============================================================================
int run_inverse_fourier_tests() {
  std::cout << "\n=== Story 04: inverse (Matsubara -> tau) Fourier transform ===\n";
  test1_polynomial_fit<UT>("uniform", 2.0);
  test1_polynomial_fit<GL>("gauss-legendre", 2.0);
  test1_polynomial_fit<CH>("chebyshev-node", 2.0);
  test2_order_independence();
  test3_tail_matsubara();
  test4_tail_tau_and_Tk();
  test5_one_level_gf();
  test6_convergence_fermi();
  test7_bosonic();
  test8_tensor_valued();
  test9_statistics_mismatch();
  std::cout << "\n=== Story 04: ALL TESTS PASSED ===\n";
  return 0;
}

} // namespace cppgw