#pragma once
#include "expansion.hpp"        // GridExpansionTau, GridExpansionMatsubara (+ scalar_type / grid_type / statistics_type aliases)
#include "grid.hpp"             // Quadrature, Grid, ...
#include "types.hpp"
#include "linalgbackend.hpp"    // gemm (Executor)

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cppgw {

namespace detail {
  // Inverse-geometry concepts: the INPUT is a grid expansion on the Matsubara
  // (ImaginaryFrequency) space and the OUTPUT one on the ImaginaryTime space;
  // both must expose the standard interface, plus the statistics type S so a
  // Fermionic/Bosonic mismatch is a compile-time error (Objective 2 of Story 04).
  template <class T>
  concept InvFromGridExpansion =
    HasFunctionSpace<T>
    && std::same_as<typename T::space, ImaginaryFrequencySpace>
    && requires { typename T::grid_type; typename T::scalar_type; typename T::DimLabel; typename T::statistics_type; };

  template <class T>
  concept InvToGridExpansion =
    HasFunctionSpace<T>
    && std::same_as<typename T::space, ImaginaryTimeSpace>
    && requires { typename T::grid_type; typename T::scalar_type; typename T::DimLabel; typename T::statistics_type; };
} // namespace detail

// ============================================================================
//  InverseFourierTransform :  Matsubara (ImaginaryFrequency)  ->  tau (ImaginaryTime)
//
//      out[s, t] = sum_m  K[m, t] . in[s, m]
//      K[m, t]   =  (1 / beta) . exp( -i . omega_m . tau_t )
//
//        m : Matsubara-frequency index        (contracted, input axis)
//        t : tau output point index           (fastest axis of `out`)
//        s : spatial indices, unchanged
//
//  This is the plain truncated inverse Matsubara sum of Story 04 (Objectives
//  3-4): for Bosonic statistics it is exactly  G(tau) = (1/beta) sum_n e^(-i w_n tau) G(i w_n)
//  (the zero mode contributes its full weight), and the same finite sum is the
//  "residual" step of the fermionic tail-corrected pipeline.
//
//  Unlike the forward (quadrature) transform there is NO Quadrature constraint
//  on the TARGET tau point set: the kernel has no quadrature of its own, so the
//  output may be any supported tau grid (uniform, Gauss-Legendre, or
//  Chebyshev-node).
//
//  Signature mirrors the forward FourierTransform exactly:
//      InverseFourierTransform<FromExp, ToExp, DataType>
//        FromExp := GridExpansionMatsubara<DataType, S>     (input)
//        ToExp   := GridExpansionTau<DataType, G, S>        (output)
//  The statistics S of the two sides MUST agree: it is part of the types, and
//  the static_assert below makes a mismatch a compile-time error (Objective 2).
//
//  `save` (const, at construction time) selects the two modes, exactly as in
//  the forward transform:
//    save == true  (default): the kernel K is built once in the constructor,
//                             STORED, and every operator() reuses it via gemm.
//    save == false           : the kernel is NEVER stored; operator() computes
//                             each K[m,t] element-by-element on the fly.
// ============================================================================
template <detail::InvFromGridExpansion FromExp,
          detail::InvToGridExpansion   ToExp,
          ComplexFloatingPoint         DataType>
class InverseFourierTransform {
  // Statistics-match check (Objective 2): a Fermionic/Bosonic mismatch between
  // the two sides is a template-instantiation error, checked at compile time.
  static_assert( std::same_as<typename FromExp::statistics_type, typename ToExp::statistics_type>,
    "InverseFourierTransform: the Matsubara (input) and tau (output) sides carry different statistics (Fermionic vs Bosonic)");
  static_assert( std::same_as<DataType, typename ToExp::scalar_type>,
    "InverseFourierTransform: DataType must equal the output expansion's scalar type");
  static_assert( std::same_as<DataType, typename FromExp::scalar_type>,
    "InverseFourierTransform: DataType must equal the input expansion's scalar type");

  using MatsuGrid = typename FromExp::grid_type;
  using TauG      = typename ToExp::grid_type;
  using TensorOut = Tensor<DataType, Executor::Host>;

public:
  using InputType  = FromExp;
  using OutputType = ToExp;
  using Scalar     = DataType;

  // Construct a transform between the two grids. `save` decides the two modes:
  //   save == true  : the kernel K is built once, STORED, and reused by every call.
  //   save == false : the kernel is NEVER stored (saves the matrix memory);
  //                   operator() computes each K[m,t] on the fly into the output.
  InverseFourierTransform(MatsuGrid matsu, TauG tau, bool save = true)
    : matsu_grid_(std::move(matsu)), tau_grid_(std::move(tau)), save_(save)
  {
    if (tau_grid_.size() == 0)
      throw std::invalid_argument("InverseFourierTransform: the target tau grid must have at least 1 point");
    if (matsu_grid_.size() == 0)
      throw std::invalid_argument("InverseFourierTransform: the Matsubara grid must have at least 1 point");
    // The 1/beta prefactor comes from the Matsubara grid; require the tau grid
    // to agree so the two sides describe the same interval [0, beta].
    if (std::abs(tau_grid_.beta().value - matsu_grid_.beta().value)
        > 1e-12 * std::max(1.0, std::abs(matsu_grid_.beta().value)))
      throw std::invalid_argument("InverseFourierTransform: the tau grid beta does not match the Matsubara grid beta");
    if (save_)
      kernel_ = build_kernel();
  }

  // ----- Accessors -----
  const MatsuGrid& matsu_grid() const { return matsu_grid_; }
  const TauG&      tau_grid()   const { return tau_grid_; }
  const TensorOut& kernel()     const { return kernel_; }   // the stored kernel (valid iff saved())
  bool             saved()      const { return save_; }

  // The (m, t) element of the inverse transformation kernel, computed directly
  // from the two grids:   K(m, t) = (1/beta) exp(-i omega_m tau_t).
  // (m: Matsubara, t: tau.)  The Bosonic zero mode (omega = 0) is a normal
  // summand here: K(0, t) = 1/beta.
  DataType K(size_t m, size_t t) const {
    const double inv_beta = 1.0 / matsu_grid_.beta().value;
    const double arg = matsu_grid_(m).value * tau_grid_(t).value;
    return DataType(inv_beta * std::cos(arg), -inv_beta * std::sin(arg));
  }

  // Apply the inverse transform to a Matsubara-represented function; return
  // the tau expansion G(tau_t) = sum_m K[m,t] G(i omega_m).
  OutputType operator()(const InputType& input) const {
    const auto& in = input.data();   // The Tensor data
    const size_t M = matsu_grid_.size();
    const size_t T = tau_grid_.size();

    // The input's Matsubara axis must be the fastest (innermost) and match the grid.
    const size_t m_idx = in.label_index(FromExp::dim_label);
    if (m_idx != 0)
      throw std::invalid_argument("InverseFourierTransform: the input Matsubara axis must be the fastest axis");
    if (in.dims()[0].dim != M)
      throw std::invalid_argument("InverseFourierTransform: input Matsubara size ("
          + std::to_string(in.dims()[0].dim)
          + ") does not match the Matsubara grid (" + std::to_string(M) + ")");

    if (save_) {
      // Stored-kernel path: gemm contracts the (stored) kernel against the input.
      // X = kernel [tau (fast), matsu (slow)], Y = input [matsu (fast), spatial...];
      // the shared (contracted) axis is the Matsubara label; matsu must be X's
      // SLOWEST and Y's FASTEST dim (as arranged by build_kernel / the input).
      TensorOut out;
      gemm(kernel_, in, out, FromExp::dim_label, FromExp::dim_label);
      return OutputType(std::move(out), tau_grid_);
    }

    // save == false: no stored kernel. Allocate the output and compute each
    // K[m, t] element-by-element on the fly, straight into the output.
    std::vector<TensorDim> out_dims;
    out_dims.reserve(in.dims().size());
    out_dims.push_back(TensorDim{ToExp::dim_label, T});      // tau = fastest
    for (size_t i = 1; i < in.dims().size(); ++i)
      out_dims.push_back(in.dims()[i]);                       // unchanged spatial dims
    TensorOut out(out_dims);

    const size_t S = in.total_elements() / M;                 // total number of spatial elements
    for (size_t sp = 0; sp < S; ++sp)
      for (size_t t = 0; t < T; ++t) {
        DataType sum{};
        for (size_t m = 0; m < M; ++m)
          // in: [matsu, <spatial...>]  matsu fastest (stride 1) -> offset (m + M*sp)
          sum += K(m, t) * in.linear(m + M * sp);
        // out: [tau, <spatial...>]     tau fastest (stride 1)   -> offset (t + T*sp)
        out.linear(t + T * sp) = sum;
      }
    return OutputType(std::move(out), tau_grid_);
  }

private:
  MatsuGrid   matsu_grid_;
  TauG        tau_grid_;
  const bool save_{};    // const: whether the kernel is stored; fixed at construction
  TensorOut kernel_;     // K[m, t]; allocated iff save_ is true

  // Build the kernel K[m,t] = (1/beta) exp(-i omega_m tau_t) into a
  // [tau (fastest = stride 1), matsu (slowest = stride T)]-labeled tensor
  // (this codebase lists dims fastest-to-slowest, so dim[0] = tau).  The flat
  // storage index of element (tau = t, matsu = m) is therefore  t + T * m.
  // Uses cos/sin on the real argument and the two-argument (real, imag)
  // constructor, so it works for any ComplexFloatingPoint scalar type.
  TensorOut build_kernel() const {
    const size_t M = matsu_grid_.size();
    const size_t T = tau_grid_.size();

    std::vector<TensorDim> k_dims = {
      TensorDim{ToExp::dim_label, T},    // tau   = fastest (stride 1)
      TensorDim{FromExp::dim_label, M}   // matsu = slowest (stride T)
    };
    TensorOut ker(k_dims);
    DataType* ker_buf = ker.data();

    for (size_t t = 0; t < T; ++t)
      for (size_t m = 0; m < M; ++m)
        ker_buf[t + T * m] = K(m, t);

    return ker;
  }
};

}
