#pragma once
// ============================================================================
//  Story 04 — Fermionic tail subtraction for the inverse (Matsubara -> tau)
//  Fourier transform.
//
//  A fermionic Green's function on [0, beta] is antiperiodic, G(tau+beta) = -G(tau),
//  and Green's functions of physical interest (e.g. G(iw) = 1/(iw - eps)) carry a
//  high-frequency expansion  G(iw) ~ c_1/(iw) + c_2/(iw)^2 + ...  whose 1/(iw)
//  term makes the plain truncated inverse sum converge at only O(1/w_max).
//
//  The fix (this file):
//
//    1.  Stencil<real_type>               - one one-sided polynomial fit on an
//                                           arbitrary (nonuniform) stencil, with
//                                           analytic endpoint derivatives.
//    2.  TauBoundaryStencils<real_type, cplx>
//                                            - the TWO INDEPENDENT one-sided fits
//                                           (M_left points near 0+, M_right points
//                                           near beta-, disjoint sets of grid
//                                           points; NO wrapped-boundary fit), plus
//                                           the coefficient extraction
//                                           c_{k+1} = (-1)^{k+1} ( f^(k)(0+) + f^(k)(beta-) ).
//    3.  FermionicTail<data_type, G>      - coefficient-carrying tail object
//                                           (owns c_1..c_P per spatial index, and
//                                           beta); evaluation at Matsubara
//                                           frequencies and on tau grids.
//    4.  TailCorrectedInverseFourierTransform<FromExp, ToExp, DataType>
//                                            - the single-call fermionic pipeline
//                                           ( r = G - tail;  plain inverse;  + tail(tau) ),
//                                           with every step also exposed separately.
//
//  The bosonic inverse needs no tail (the plain sum is well conditioned); it is
//  served by InverseFourierTransform directly (grid/fourier_inverse.hpp).
// ============================================================================
#include "expansion.hpp"
#include "grid/fourier_inverse.hpp"   // InverseFourierTransform + detail::Inv{From,To}GridExpansion
#include "types.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cppgw {

namespace detail {
  // --------------------------------------------------------------------------
  //  Lagrange-basis endpoint derivative.
  //
  //  For distinct nodes x_0..x_{M-1}, the Lagrange basis polynomial l_i is
  //
  //      l_i(t) = p_i(t) / w_i ,     p_i(t) = prod_{j != i} (t - x_j),
  //                                    w_i  = prod_{j != i} (x_i - x_j)
  //
  //  p_i^(k)(z) is built by multiplying the factors (t - x_j) one at a time
  //  with the exact derivative propagation
  //      (P (t-a))^(m)  =  P^(m) (t-a) + m P^(m-1)      (m = 0..k),
  //
  //  which stays at the intrinsic magnitude of p_i^(m)(z).  (A monomial
  //  expansion evaluated OUTSIDE the node span would involve terms of size
  //  O(M! z^M) that cancel to O(z^0) and is catastrophically unstable.)
  //
  //  All arithmetic is in `R` (RealFloatingPoint), so fits can run in either
  //  `double` or a boost multiprecision real (Objective 7 / feedback 4).
  // --------------------------------------------------------------------------
  template <RealFloatingPoint R>
  R lagrange_basis_kderivative(const std::vector<R>& x, size_t i, size_t k, R z) {
    const size_t M = x.size();
    if (M == 0 || i >= M)
      throw std::out_of_range("lagrange_basis_kderivative: node index out of range");
    if (k >= M)
      throw std::invalid_argument("lagrange_basis_kderivative: k must be < M (degree < M interpolation)");

    // d[m] = P^(m)(z) for the factors multiplied so far;  start P = 1.
    std::vector<R> d(k + 1, R(0.0));
    d[0] = R(1.0);
    for (size_t j = 0; j < M; ++j) {
      if (j == i) continue;
      const R za = z - x[j];
      for (size_t m = k; m >= 1; --m)
        d[m] = d[m] * za + R(m) * d[m - 1];
      d[0] = d[0] * za;
    }

    R w = R(1.0);                                // w_i = prod_{j != i} (x_i - x_j)
    for (size_t j = 0; j < M; ++j)
      if (j != i) w *= (x[i] - x[j]);

    return d[k] / w;
  }

  // --------------------------------------------------------------------------
  //  T_k(tau)  =  (1/beta) sum_n  e^{-i w_n tau} (i w_n)^{-k}   (Fermi Matsubara),
  //
  //  the tau-space counterpart of the tail's k-th power (i w_n)^{-k}.  These are
  //  Euler-Bernoulli polynomials in (tau/beta); the first six (verbatim from the
  //  Story 04 spec, k = 1..6):
  //
  //    T_1 = -1/2
  //    T_2 = -1/4 beta + 1/2 tau
  //    T_3 = 1/2 (1/2 beta*tau - 1/2 tau^2)
  //    T_4 = 1/6 (1/8 beta^3 - 3/4 beta*tau^2 + 1/2 tau^3)
  //    T_5 = 1/24 (-1/2 beta^3*tau + beta*tau^3 - 1/2 tau^4)
  //    T_6 = 1/120 (-1/4 beta^5 + 5/4 beta^3*tau^2 - 5/4 beta*tau^4 + 1/2 tau^5)
  //
  //  (Test 4 verifies these against the explicit Matsubara sums.)
  // --------------------------------------------------------------------------
  template <RealFloatingPoint R>
  R tau_Tk(size_t k, R beta, R tau) {
    const R b = beta, t = tau;
    switch (k) {
      case 1: return -R(1.0) / 2;
      case 2: return -b / 4 + t / 2;
      case 3: return (b * t) / 4 - (t * t) / 4;
      case 4: return ( b*b*b / 8 - 3.0 * b * t * t / 4 + t * t * t / 2 ) / 6;
      case 5: return ( -b*b*b * t / 2 + b * t * t * t - t * t * t * t / 2 ) / 24;
      case 6: return ( -b*b*b*b*b / 4 + 5.0 * b*b*b * t * t / 4
                     - 5.0 * b * t * t * t * t / 4 + t * t * t * t * t / 2 ) / 120;
      default:
        throw std::invalid_argument("tau_Tk: only T_1..T_6 are implemented (Story 04; a higher tail order is a future Story)");
    }
  }

  // The real part type of a (possibly complex) scalar: std::complex<R> -> R,
  // a native floating-point type -> itself, anything else (e.g. boost
  // complex) -> double.  Used to choose the real arithmetic type of the stencil fits.
  template <class C> struct RealOf { using type = double; };
  template <class R> struct RealOf<std::complex<R>> { using type = R; };
  template <std::floating_point R> struct RealOf<R> { using type = R; };
} // namespace detail

// ============================================================================
//  Stencil : one one-sided polynomial fit
//
//  Owns M abscissas x (the stencil points on tau) and values y = f(x).
//  The interpolant is the unique degree < M polynomial through (x_i, y_i);
//  `derivative(k, z)` reads off its k-th derivative at an arbitrary point z
//  (the boundary endpoint: 0 for the left stencil, beta for the right) ANALYTICALLY
//  from the Lagrange basis (Objective 5).  Exact, up to round-off, when f is a
//  polynomial of degree < M.
//
//  Templated over the REAL scalar so the Lagrange-basis arithmetic can be
//  instantiated with either `double` or a multiprecision real (Objective 7,
//  feedback 4).
// ============================================================================
template <RealFloatingPoint real_type>
class Stencil {
public:
  // x: the stencil abscissas (distinct, any order), y: the values f(x_i)
  //    (same length).  M = x.size(); the interpolating polynomial has degree M-1.
  Stencil(std::vector<real_type> x, std::vector<real_type> y)
    : x_(std::move(x)), y_(std::move(y))
  {
    if (x_.empty())
      throw std::invalid_argument("Stencil: need at least 1 abscissa");
    if (x_.size() != y_.size())
      throw std::invalid_argument("Stencil: x and y must have equal sizes");
    for (size_t i = 0; i < x_.size(); ++i)
      for (size_t j = i + 1; j < x_.size(); ++j)
        if (x_[i] == x_[j])
          throw std::invalid_argument("Stencil: duplicate abscissas");
  }

  size_t M() const { return x_.size(); }
  size_t size() const { return x_.size(); }
  const std::vector<real_type>& x() const { return x_; }
  const std::vector<real_type>& y() const { return y_; }

  // k-th derivative of the unique degree < M polynomial through (x_i, y_i),
  // evaluated at `evaluation_point` (z = 0 for the left boundary, z = beta for
  // the right).  k must be <= M - 1.
  real_type derivative(size_t k, double evaluation_point) const {
    if (k >= x_.size())
      throw std::out_of_range("Stencil::derivative: order k must be < M (the interpolant is degree M-1)");
    const real_type z = real_type(evaluation_point);
    real_type acc = real_type(0.0);
    for (size_t i = 0; i < x_.size(); ++i)
      acc += y_[i] * detail::lagrange_basis_kderivative(x_, i, k, z);
    return acc;
  }

  // Derivatives of orders 0..K-1 together (K must be <= M).
  std::vector<real_type> derivatives(size_t K, double evaluation_point) const {
    if (K > x_.size())
      throw std::invalid_argument("Stencil::derivatives: K must be <= M");
    std::vector<real_type> out(K);
    for (size_t k = 0; k < K; ++k)
      out[k] = derivative(k, evaluation_point);
    return out;
  }

private:
  std::vector<real_type> x_;     // abscissas, size M
  std::vector<real_type> y_;     // values f(x_i)
};

// ============================================================================
//  TauBoundaryStencils : the two INDEPENDENT one-sided boundary fits
//
//  Left  : the M_left  tau-grid points with the SMALLEST tau -> f^(k)(0+),    k = 0..M_left  - 1
//  Right : the M_right tau-grid points with the LARGEST tau -> f^(k)(beta-),  k = 0..M_right - 1
//
//  The two domains are disjoint sets of grid points and are fitted separately
//  (NO single wrapped fit across the beta/0 seam -- that would assume a
//  continuous, smooth function at the boundary, exactly what a fermionic GF
//  with a jump at the endpoints is NOT).  This holds the resulting tensor-
//  valued endpoint derivatives (one row per order, one column per spatial
//  index) and provides the tail-coefficient extraction (Objective 6):
//
//      c_{k+1} = (-1)^{k+1} ( f^(k)(0+) + f^(k)(beta-) ),   k = 0..P-1.
//
//  Templated over the REAL arithmetic type `real_type` and the complex data
//  scalar `cplx`; real- and complex-valued function data are both supported
//  (complex data is fit via paired real/imag Stencils).
// ============================================================================
template <RealFloatingPoint real_type, ComplexFloatingPoint cplx>
class TauBoundaryStencils {
  static_assert( requires { cplx(real_type(1.0), real_type(2.0)); },
    "TauBoundaryStencils: cplx must be constructible from (real_type, real_type)");
  using TensorT = Tensor<cplx, Executor::Host>;

public:
  size_t M_left()  const { return M_left_; }
  size_t M_right() const { return M_right_; }
  real_type beta() const { return beta_; }
  const std::vector<double>& x_left()  const { return x_left_; }
  const std::vector<double>& x_right() const { return x_right_; }

  // f^(k)(0+)   per spatial index;  dims: [k = 0..M_left-1 (fastest)], spatial...
  const TensorT& L() const { return Lk_; }
  // f^(k)(beta-) per spatial index;  dims: [k = 0..M_right-1 (fastest)], spatial...
  const TensorT& R() const { return Rk_; }

  // Extract the tail coefficients c_1..c_P  (P must be <= min(M_left, M_right)),
  // as a tensor [P (fastest), spatial...]:
  //   c_{k+1} = (-1)^{k+1} ( f^(k)(0+) + f^(k)(beta-) ).
  TensorT extract_coeffs(size_t P) const {
    if (P == 0)
      throw std::invalid_argument("TauBoundaryStencils::extract_coeffs: P must be >= 1");
    if (P > std::min(M_left_, M_right_))
      throw std::invalid_argument("TauBoundaryStencils::extract_coeffs: P must be <= min(M_left, M_right) ("
          + std::to_string(std::min(M_left_, M_right_)) + ")");
    const std::vector<TensorDim> sp_dims(Lk_.dims().begin() + 1, Lk_.dims().end());
    std::vector<TensorDim> dims;
    dims.push_back(TensorDim{TensorDimLabel("coeff"), P});
    dims.insert(dims.end(), sp_dims.begin(), sp_dims.end());
    TensorT out(dims);
    const size_t S = out.total_elements() / P;
    for (size_t k = 0; k < P; ++k) {
      const cplx sign = (k % 2 == 0) ? cplx(-1.0, 0.0) : cplx(1.0, 0.0);   // (-1)^{k+1}
      for (size_t s = 0; s < S; ++s)
        out.linear(k + P * s) = sign * (Lk_.linear(k + M_left_ * s) + Rk_.linear(k + M_right_ * s));
    }
    return out;
  }

  // Fit both stencils from `tau_data` (a tau grid-expansion tensor: tau axis
  // FASTEST, size == grid.size(); scalar type V real or complex) on grid `g`.
  template <FloatingPoint V, ImaginaryTimeGrid G>
  explicit TauBoundaryStencils(const Tensor<V, Executor::Host>& tau_data, const G& grid,
                               size_t M_left, size_t M_right)
    : M_left_(M_left), M_right_(M_right), beta_(real_type(grid.beta().value))
  {
    const size_t T = grid.size();
    if (M_left_ == 0 || M_right_ == 0)
      throw std::invalid_argument("TauBoundaryStencils: stencil sizes must be >= 1");
    if (M_left_ > T || M_right_ > T)
      throw std::invalid_argument("TauBoundaryStencils: a stencil size exceeds the tau grid size ("
          + std::to_string(T) + ")");
    const size_t tau_idx = tau_data.label_index(G::dim_label);
    if (tau_idx != 0)
      throw std::invalid_argument("TauBoundaryStencils: the tau axis of tau_data must be the fastest axis");
    if (tau_data.dims()[0].dim != T)
      throw std::invalid_argument("TauBoundaryStencils: tau_data size ("
          + std::to_string(tau_data.dims()[0].dim)
          + ") does not match the tau grid (" + std::to_string(T) + ")");

    const Tensor<V, Executor::Host>& d = tau_data;
    const size_t S = d.total_elements() / T;       // spatial elements (tau block = fastest)

    // Work in ASCENDING tau order (the input grid's storage order is not
    // guaranteed to be ascending -- e.g. the Chebyshev node grid lists points
    // from beta down to 0):  left = M_left smallest tau, right = M_right largest.
    std::vector<size_t> idx(T);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) { return grid(a).value < grid(b).value; });

    std::vector<real_type>  x_left(M_left_), x_right(M_right_);
    x_left_.resize(M_left_);   x_right_.resize(M_right_);
    std::vector<std::vector<V>> yl(S), yr(S);
    for (size_t i = 0; i < M_left_; ++i) {
      const double xv = grid(idx[i]).value;
      x_left[i]  = real_type(xv);
      x_left_[i] = xv;
      for (size_t s = 0; s < S; ++s) yl[s].push_back(d.linear(idx[i] + T * s));
    }
    for (size_t i = 0; i < M_right_; ++i) {
      const size_t gi = idx[T - M_right_ + i];
      const double xv = grid(gi).value;
      x_right[i]  = real_type(xv);
      x_right_[i] = xv;
      for (size_t s = 0; s < S; ++s) yr[s].push_back(d.linear(gi + T * s));
    }

    // Spatial dim list of the derivative tensors (everything after the tau axis).
    const std::vector<TensorDim> sp_dims(d.dims().begin() + 1, d.dims().end());

    auto build_row = [&](const char* lbl, size_t K, const std::vector<real_type>& xs,
                         const std::vector<std::vector<V>>& ys, double z) -> TensorT {
      std::vector<TensorDim> dims;
      dims.push_back(TensorDim{TensorDimLabel(lbl), K});
      dims.insert(dims.end(), sp_dims.begin(), sp_dims.end());
      TensorT out(dims);
      const auto re = real_parts(ys);
      for (size_t s = 0; s < S; ++s) {
        const Stencil<real_type> re_stencil(xs, re[s]);
        if constexpr (RealFloatingPoint<V>) {
          for (size_t k = 0; k < K; ++k)
            out.linear(k + K * s) = cplx(re_stencil.derivative(k, z), 0.0);
        } else {
          const auto im = imag_parts(ys);
          const Stencil<real_type> im_stencil(xs, im[s]);
          for (size_t k = 0; k < K; ++k)
            out.linear(k + K * s) = cplx(re_stencil.derivative(k, z), im_stencil.derivative(k, z));
        }
      }
      return out;
    };

    Lk_ = build_row("L", M_left_,  x_left,  yl, /*z =*/0.0);            // at 0+
    Rk_ = build_row("R", M_right_, x_right, yr, /*z =*/double(beta_));  // at beta-
  }

private:
  size_t M_left_, M_right_;
  real_type beta_;
  std::vector<double> x_left_, x_right_;
  TensorT Lk_, Rk_;

  // Real / imaginary parts of a block of (possibly real) data, in real_type.
  template <FloatingPoint V>
  static std::vector<std::vector<real_type>> real_parts(const std::vector<std::vector<V>>& v) {
    std::vector<std::vector<real_type>> o(v.size());
    for (size_t s = 0; s < v.size(); ++s) {
      o[s].resize(v[s].size());
      for (size_t i = 0; i < v[s].size(); ++i)
        if constexpr (RealFloatingPoint<V>) o[s][i] = real_type(v[s][i]);
        else                                o[s][i] = real_type(v[s][i].real());
    }
    return o;
  }
  template <FloatingPoint V>
  static std::vector<std::vector<real_type>> imag_parts(const std::vector<std::vector<V>>& v) {
    std::vector<std::vector<real_type>> o(v.size());
    for (size_t s = 0; s < v.size(); ++s) {
      o[s].resize(v[s].size());
      for (size_t i = 0; i < v[s].size(); ++i)
        if constexpr (RealFloatingPoint<V>) o[s][i] = real_type(0.0);   // real data: no imaginary part
        else                                o[s][i] = real_type(v[s][i].imag());
    }
    return o;
  }
};

// ============================================================================
//  FermionicTail : coefficient-carrying high-frequency tail (Objective 7)
//
//  Owns, per spatial index, the tail coefficients  c_ = [c_1..c_P]  with the
//  coefficient axis the FASTEST:    c_( spatial..., <coeff> ),
//  following the ChebyshevExpansion* mantra (owns a coefficient tensor, exposes
//  size() / data() / operator()(i)), plus beta (needed for the T_k polynomials).
//
//  Evaluation (fermionic by construction -- a BosonicTail would be a future Story):
//      at_matsubara(iw) :  sum_k c_k (iw)^{-k}          (G_tail(iw) at one frequency;
//                        the singular 1/(iw) terms are NOT evaluated at iw = 0 --
//                        by the Story 04 convention  G_tail(0) := 0  -- and the
//                        method simply returns zero there)
//      at_tau(tau)      :  sum_k c_k T_k(tau)           (G_tail(tau) at one point)
//      at_tau_grid(g)   :  the full GridExpansionTau<..., G, Fermionic> on g
// ============================================================================
template <FloatingPoint data_type, ImaginaryTimeGrid G>
class FermionicTail {
  using TensorT = Tensor<data_type, Executor::Host>;
  using rtype   = detail::RealOf<data_type>::type;   // real arithmetic for the T_k polynomials

  TensorT            c_;       // [coeff (fastest, size P_)], spatial...
  InverseTemperature beta_;
  size_t             P_ = 0;
public:
  using TensorType      = TensorT;
  using statistics_type = Fermionic;      // the tail mechanism is fermionic by construction

  FermionicTail() = delete;

  // Adopt a coefficient tensor  [coeff (fastest, size P), spatial...].
  explicit FermionicTail(TensorT data, InverseTemperature beta)
    : c_(std::move(data)), beta_(beta)
  {
    if (c_.rank() == 0)
      throw std::invalid_argument("FermionicTail: the coefficient tensor needs at least the (fastest) coefficient axis");
    P_ = c_.dims()[0].dim;
    if (P_ == 0)
      throw std::invalid_argument("FermionicTail: P (the coefficient-axis size) must be >= 1");
  }

  // A fresh, all-zero tail over a spatial shape with P coefficients.
  explicit FermionicTail(TensorShape spatial, InverseTemperature beta, size_t P)
    : beta_(beta), P_(P)
  {
    if (P == 0)
      throw std::invalid_argument("FermionicTail: P must be >= 1");
    if (spatial.size() == 0)
      throw std::invalid_argument("FermionicTail: the spatial shape must have at least one dim");
    spatial.insert_fast(TensorDim{TensorDimLabel("coeff"), P});   // coeff axis = fastest
    c_ = TensorT(spatial);
  }

  // ----- Accessors (mirroring the ChebyshevExpansion* interface) -----
  size_t size() const        { return P_; }   // size of the coefficient axis
  size_t P() const           { return P_; }   // alias
  size_t tail_order() const  { return P_; }   // == P_
  InverseTemperature beta() const             { return beta_; }

  TensorT&       data()       { return c_; }
  const TensorT& data() const { return c_; }

  // The coefficient c_{i+1}, i = 0..P-1 (i indexes the coefficient axis).
  // For a single spatial element (the common scalar case; total == P).
  data_type coeff(size_t i) const {
    if (i >= P_)
      throw std::out_of_range("FermionicTail::coeff: i must be < P");
    if (c_.total_elements() != P_)
      throw std::invalid_argument("FermionicTail::coeff: c_{i+1} is a spatial tensor (more than one spatial"
                                  " element); use coeff_slice(i) for the full spatial block");
    return c_.linear(i);
  }
  // The whole spatial block of c_{i+1} (a tensor with the spatial dims of c_).
  TensorT coeff_slice(size_t i) const {
    if (i >= P_)
      throw std::out_of_range("FermionicTail::coeff_slice: i must be < P");
    const std::vector<TensorDim> sd = spatial_dims_of(c_);
    TensorT out(sd);
    const size_t S = (sd.empty() ? size_t(1) : size_t(out.total_elements()));
    for (size_t s = 0; s < S; ++s)
      out.linear(s) = c_.linear(i + P_ * s);
    return out;
  }
  // Set c_{i+1} for EVERY spatial component (a broadcast).
  void set_coeff(size_t i, data_type value) {
    if (i >= P_)
      throw std::out_of_range("FermionicTail::set_coeff: i must be < P");
    const size_t S = c_.total_elements() / P_;
    for (size_t s = 0; s < S; ++s)
      c_.linear(i + P_ * s) = value;
  }
  // This tail has no spatial element to index; kept for API symmetry with
  // GridExpansionTau::operator()(i) (index into the coefficient axis).
  data_type operator()(size_t i) const { return coeff(i); }

  // ----- Evaluation -----
private:
  static std::vector<TensorDim> spatial_dims_of(const TensorT& t) {
    if (t.rank() <= 1) return {};
    return std::vector<TensorDim>(t.dims().begin() + 1, t.dims().end());
  }

public:
  // G_tail(iw) = sum_{k=1..P} c_k (iw)^{-k},   (iw = i*omega).
  // By the Story 04 convention (G_tail(0) := 0, because the tail is a
  // HIGH-frequency object and its 1/(iw) terms diverge at iw = 0), this returns
  // the (spatial) zero tensor when iw == 0.  The result is a plain tensor with
  // one value per spatial index.
  // (Only viable for a complex data type; a real scalar cannot represent 1/(iw).)
  TensorT at_matsubara(double iw) const requires (requires { data_type(0.0, 1.0); }) {
    if (iw == 0.0)
      return TensorT(spatial_dims_of(c_));   // zero spatial tensor (convention: G_tail(0) := 0)
    const std::vector<TensorDim> sd = spatial_dims_of(c_);
    TensorT out(sd);
    const size_t S = (sd.empty() ? size_t(1) : size_t(out.total_elements()));
    const data_type iwC(0.0, iw);                 // i * omega
    const data_type inv = data_type(1.0) / iwC;   // (iw)^{-1}
    for (size_t s = 0; s < S; ++s) {
      data_type acc = data_type(0.0), pw = inv;
      for (size_t k = 1; k <= P_; ++k) {          // c_k * (iw)^{-k}
        acc += c_.linear((k - 1) + P_ * s) * pw;
        pw *= inv;
      }
      out.linear(s) = acc;
    }
    return out;
  }

  // G_tail(tau) = sum_{k=1..P} c_k T_k(tau),  T_k the tau-space polynomial
  // (detail::tau_Tk; orders 1..6 in this Story).
  TensorT at_tau(double tau) const {
    const std::vector<TensorDim> sd = spatial_dims_of(c_);
    TensorT out(sd);
    const size_t S = (sd.empty() ? size_t(1) : size_t(out.total_elements()));
    const rtype b = rtype(beta_.value), t = rtype(tau);
    for (size_t s = 0; s < S; ++s) {
      data_type acc = data_type(0.0);
      for (size_t k = 1; k <= P_; ++k)
        acc += c_.linear((k - 1) + P_ * s) * detail::tau_Tk<rtype>(k, b, t);
      out.linear(s) = acc;
    }
    return out;
  }

  // The full tail on a tau grid: a GridExpansionTau on (g) with the tail's
  // OWNED beta.  For direct composition in TailCorrectedInverseFourierTransform.
  GridExpansionTau<data_type, G, Fermionic> at_tau_grid(const G& grid) const {
    if (std::abs(grid.beta().value - beta_.value)
        > 1e-12 * std::max(1.0, std::abs(beta_.value)))
      throw std::invalid_argument("FermionicTail::at_tau_grid: the grid beta does not match the tail beta");
    const std::vector<TensorDim> sd = spatial_dims_of(c_);
    const size_t T = grid.size();
    std::vector<TensorDim> dims;
    dims.push_back(TensorDim{G::dim_label, T});    // tau = fastest
    dims.insert(dims.end(), sd.begin(), sd.end());
    TensorT out(dims);
    const size_t S = (sd.empty() ? size_t(1) : size_t(out.total_elements() / T));
    const rtype b = rtype(beta_.value);
    for (size_t s = 0; s < S; ++s)
      for (size_t t = 0; t < T; ++t) {
        data_type acc = data_type(0.0);
        for (size_t k = 1; k <= P_; ++k)
          acc += c_.linear((k - 1) + P_ * s) * detail::tau_Tk<rtype>(k, b, rtype(grid(t).value));
        out.linear(t + T * s) = acc;
      }
    return GridExpansionTau<data_type, G, Fermionic>(std::move(out), grid);
  }
};

// ============================================================================
//  TailCorrectedInverseFourierTransform : the single-call fermionic pipeline
//  (Objective 8)
//
//      out   =  plain_inverse( G(iw) - G_tail(iw) )  +  G_tail(tau)
//
//  i.e.    G(tau) = (1/beta) sum_n e^{-i w_n tau} [G(iw_n) - G_tail(iw_n)] + G_tail(tau)
//
//  The coefficients come EITHER from the (independently fitted) one-sided
//  boundary stencils of the tau-side samples  (operator()(tau_samples, in)),
//  OR from a user-supplied FermionicTail (operator()(in, tail)).  Every step
//  is ALSO exposed as its own public method (stencil_fit / extract_coeffs /
//  make_tail / subtract_tail / inverse_tau / add_tau), so tests can drive any
//  of them independently (Objective 8, feedback 3).
//
//      FromExp := GridExpansionMatsubara<DataType, S>   (input, S = Fermionic in practice)
//      ToExp   := GridExpansionTau<DataType, G, S>      (output)
// ============================================================================
template <detail::InvFromGridExpansion FromExp,
          detail::InvToGridExpansion   ToExp,
          ComplexFloatingPoint         DataType>
class TailCorrectedInverseFourierTransform {
  using TauG      = typename ToExp::grid_type;
  using MatsuGrid = typename FromExp::grid_type;
  using cplx      = DataType;
  using real_type = detail::RealOf<cplx>::type;
  static_assert( RealFloatingPoint<real_type>,
    "TailCorrectedInverseFourierTransform: could not determine a real arithmetic type for the stencil fits");

  static_assert(std::same_as<typename FromExp::statistics_type, typename ToExp::statistics_type>,
    "TailCorrectedInverseFourierTransform: the Matsubara and tau sides carry different statistics (Fermionic/Bosonic)");
public:
  using Boundaries = TauBoundaryStencils<real_type, cplx>;
  using Tail       = FermionicTail<cplx, TauG>;
  using plain_type = InverseFourierTransform<FromExp, ToExp, DataType>;

  // Grids + the pipeline configuration.
  //   save    : stored-kernel (gemm) vs on-the-fly plain inverse (Objective 2).
  //   M_left  : left-boundary (0+) stencil size (nearest M_left tau points).
  //   M_right : right-boundary (beta-) stencil size (nearest M_right tau points).
  //   P       : tail order (# of coefficients; P = 0 -> min(M_left, M_right));
  //             must be <= min(M_left, M_right) and <= 6 in this Story (T_1..T_6).
  TailCorrectedInverseFourierTransform(MatsuGrid matsu, TauG tau,
                                       bool save = true,
                                       size_t M_left = 8, size_t M_right = 8,
                                       size_t P = 0)
    : plain_(std::move(matsu), std::move(tau), save),
      M_left_(M_left), M_right_(M_right),
      P_(P == 0 ? std::min(M_left, M_right) : P)
  {
    if (M_left_ == 0 || M_right_ == 0)
      throw std::invalid_argument("TailCorrectedInverseFourierTransform: M_left and M_right must be >= 1");
    if (P_ > std::min(M_left_, M_right_))
      throw std::invalid_argument("TailCorrectedInverseFourierTransform: P must be <= min(M_left, M_right)");
    if (P_ > 6)
      throw std::invalid_argument("TailCorrectedInverseFourierTransform: P > 6 in this Story (only T_1..T_6 are provided)");
  }

  // ----- Configuration accessors -----
  bool     saved()   const { return plain_.saved(); }
  size_t   M_left()  const { return M_left_; }
  size_t   M_right() const { return M_right_; }
  size_t   P()       const { return P_; }
  const plain_type& plain() const { return plain_; }

  // =====================================================================
  //  Pipeline steps (each a separate public method; tests may call any of
  //  them independently -- Objective 8).
  // =====================================================================

  // STEP 1: the two independent one-sided stencil fits on the tau samples.
  //   (V may be a real or complex scalar: complex data fits via the real and
  //   imaginary parts.)  The returned Boundaries owns the tensor-valued
  //   endpoint derivatives  L() = f^(k)(0+)  and  R() = f^(k)(beta-).
  template <FloatingPoint V>
  Boundaries stencil_fit(const Tensor<V, Executor::Host>& tau_data, size_t M_left = 0, size_t M_right = 0) const {
    const size_t ml = (M_left  == 0) ? M_left_  : M_left;
    const size_t mr = (M_right == 0) ? M_right_ : M_right;
    return Boundaries(tau_data, plain_.tau_grid(), ml, mr);
  }
  // Convenience: the output-type tau expansion itself.
  Boundaries stencil_fit(const ToExp& src, size_t M_left = 0, size_t M_right = 0) const {
    return stencil_fit(src.data(), M_left, M_right);
  }

  // STEP 2: c_1..c_P from an already-fitted boundary object (P = 0 -> this
  // object's configured P).
  Tensor<cplx, Executor::Host> extract_coeffs(const Boundaries& b, size_t P = 0) const {
    const size_t p = (P == 0) ? P_ : P;
    return b.extract_coeffs(p);
  }

  // STEP 3: build the coefficient-carrying tail from the extracted coefficients.
  Tail make_tail(const Tensor<cplx, Executor::Host>& coeff, InverseTemperature beta, size_t P) const {
    if (coeff.dims().empty() || coeff.dims()[0].dim != P)
      throw std::invalid_argument("TailCorrectedInverseFourierTransform::make_tail: the (fastest) coefficient axis must have size P");
    if (P > 6)
      throw std::invalid_argument("TailCorrectedInverseFourierTransform::make_tail: P > 6 in this Story (only T_1..T_6 are provided)");
    return Tail(coeff, beta);
  }

  // STEP 4: r(iw_m) = G(iw_m) - G_tail(iw_m), element-wise on the Matsubara axis.
  FromExp subtract_tail(const FromExp& in, const Tail& tail) const {
    const auto& d = in.data();
    const size_t M = plain_.matsu_grid().size();
    const size_t idx = d.label_index(FromExp::dim_label);
    if (idx != 0)
      throw std::invalid_argument("TailCorrectedInverseFourierTransform::subtract_tail: the Matsubara axis must be the fastest");
    if (d.dims()[0].dim != M)
      throw std::invalid_argument("TailCorrectedInverseFourierTransform::subtract_tail: Matsubara size mismatch");
    const size_t S = d.total_elements() / M;
    Tensor<cplx, Executor::Host> out(d.dims());
    // Precompute the tail at each Matsubara frequency (one spatial tensor each;
    // at the Bosonic zero mode this is just the convention zero).
    std::vector<Tensor<cplx, Executor::Host>> tv(M);
    for (size_t m = 0; m < M; ++m)
      tv[m] = tail.at_matsubara(plain_.matsu_grid()(m).value);
    for (size_t sp = 0; sp < S; ++sp)
      for (size_t m = 0; m < M; ++m)
        out.linear(m + M * sp) = d.linear(m + M * sp) - tv[m].linear(sp);
    return FromExp(std::move(out), plain_.matsu_grid());
  }

  // STEP 5: the plain inverse transform of `resid` (delegates to the stored
  // InverseFourierTransform; gemm path iff save).
  ToExp inverse_tau(const FromExp& resid) const {
    return plain_(resid);
  }

  // STEP 6: add the tau-space tail, element-wise along the tau axis.
  ToExp add_tau(const ToExp& resid, const Tail& tail) const {
    const auto& r = resid.data();
    const size_t T = plain_.tau_grid().size();
    const size_t idx = r.label_index(ToExp::dim_label);
    if (idx != 0)
      throw std::invalid_argument("TailCorrectedInverseFourierTransform::add_tau: the tau axis must be the fastest");
    if (r.dims()[0].dim != T)
      throw std::invalid_argument("TailCorrectedInverseFourierTransform::add_tau: tau size mismatch");
    const size_t S = r.total_elements() / T;
    const auto tail_grid = tail.at_tau_grid(plain_.tau_grid());
    const auto& tg  = tail_grid.data();
    Tensor<cplx, Executor::Host> out(r.dims());
    for (size_t sp = 0; sp < S; ++sp)
      for (size_t t = 0; t < T; ++t)
        out.linear(t + T * sp) = r.linear(t + T * sp) + tg.linear(t + T * sp);
    return ToExp(std::move(out), plain_.tau_grid());
  }

  // =====================================================================
  //  Single-call pipelines (Objective 8).
  // =====================================================================

  // FULL pipeline: fit the two boundary stencils on `tau_samples`, extract
  // c_1..c_P (P = configured), build the tail,  r = G - tail(iw),  invert,  + tail(tau).
  ToExp operator()(const ToExp& tau_samples, const FromExp& in) const {
    auto st = stencil_fit(tau_samples);                        // step 1
    auto c  = extract_coeffs(st);                             // step 2
    auto tl = make_tail(c, plain_.tau_grid().beta(), P_);     // step 3
    auto r  = subtract_tail(in, tl);                          // step 4
    auto rt = inverse_tau(r);                                 // step 5
    return add_tau(rt, tl);                                   // step 6
  }

  // Pipeline with a USER-SUPPLIED tail (no stencil fit needed).
  ToExp operator()(const FromExp& in, const Tail& tail) const {
    auto r  = subtract_tail(in, tail);                        // step 4
    auto rt = inverse_tau(r);                                 // step 5
    return add_tau(rt, tail);                                 // step 6
  }

private:
  plain_type plain_;      // the plain (residual) inverse transform
  const size_t M_left_{};   // left-boundary stencil size
  const size_t M_right_{};  // right-boundary stencil size
  const size_t P_{};        // tail order (number of coefficients)
};

}    // namespace cppgw
