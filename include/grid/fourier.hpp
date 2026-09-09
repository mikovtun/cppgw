#pragma once
#include "expansion.hpp"        // GridExpansionTau, GridExpansionMatsubara (+ scalar_type / grid_type aliases)
#include "grid.hpp"             // Quadrature, Grid, ...
#include "types.hpp"
#include "linalgbackend.hpp"    // Executor

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cppgw {

namespace detail {
  // The input must be a grid expansion living on the ImaginaryTime space
  // (i.e. a GridExpansionTau<...>), the output one on ImaginaryFrequency
  // (a GridExpansionMatsubara<...>), and both must expose the standard
  // interface (grid_type, scalar_type, DimLabel, dim_label, size(), data()).
  template <class T>
  concept FromGridExpansion =
    HasFunctionSpace<T>
    && std::same_as<typename T::space, ImaginaryTimeSpace>
    && requires { typename T::grid_type; typename T::scalar_type; typename T::DimLabel; };

  template <class T>
  concept ToGridExpansion =
    HasFunctionSpace<T>
    && std::same_as<typename T::space, ImaginaryFrequencySpace>
    && requires { typename T::grid_type; typename T::scalar_type; typename T::DimLabel; };

  // The product "kernel element * input element" must be well formed and
  // assignable to the kernel (output) type.
  template <class Kernel, class In>
  concept MultipliesInto =
    requires(const Kernel& k, const In& x) {
      { k * x } -> std::convertible_to<Kernel>;
    };
} // namespace detail


// ============================================================================
//  FourierTransform :  tau (ImaginaryTime)  ->  Matsubara (ImaginaryFrequency)
//
//      out[m, s] = sum_t  K[m, t] . in[s, t]
//      K[m, t]   =  w_t . exp( i . omega_m . tau_t )
//
//        m : target Matsubara-frequency index        (fastest axis)
//        t : source tau quadrature-point index       (contracted)
//        s : spatial indices, unchanged
//        w_t : quadrature weight at tau point t      (=> source grid must be a Quadrature)
//
//  The transform is fully defined by the source (tau) grid and the target
//  (Matsubara) grid, which are captured at construction time. `save` (const)
//  selects the two modes:
//    save == true  (default): the kernel K is built once in the constructor,
//                             STORED, and reused by every operator() call.
//    save == false           : the kernel is NEVER stored (this mode is for when
//                             storing the matrix would use too much memory);
//                             operator() computes each K[m,t] element-by-element
//                             on the fly, directly into the output tensor.
//
//    in    : an expansion on a tau grid  (FromExp  = GridExpansionTau<...>)
//    out   : an expansion on a Matsubara grid (ToExp = GridExpansionMatsubara<...>)
//    K / out scalar type: DataType (complex). DataType must equal ToExp's scalar
//    type so the result can be adopted into ToExp.
//
//  The two modes use different machinery:
//    save == true  -> gemm(...) (the mixed-type Tensor/LinAlgBackend matrix product)
//                     contracts the stored, complex kernel against the (possibly real)
//                     input; the arithmetic is done in the output (complex) type.
//    save == false -> no kernel is available (it is never stored), so each K[m,t] is
//                     computed on the fly and accumulated directly into the output.
// ============================================================================
template <detail::FromGridExpansion FromExp,
          detail::ToGridExpansion   ToExp,
          ComplexFloatingPoint       DataType>
class FourierTransform {
  static_assert( Quadrature<typename FromExp::grid_type>,
    "FourierTransform: the source (tau) grid must be a Quadrature (must expose weights)");
  static_assert( std::same_as<DataType, typename ToExp::scalar_type>,
    "FourierTransform: DataType must equal the output expansion's scalar type");
  static_assert( detail::MultipliesInto<DataType, typename FromExp::scalar_type>,
    "FourierTransform: a kernel element (DataType) times an input element (FromExp scalar) "
    "must be well formed and assignable to DataType");

  using InScalar   = typename FromExp::scalar_type;
  using TauGrid    = typename FromExp::grid_type;
  using MatsuGrid  = typename ToExp::grid_type;
  using TensorOut  = Tensor<DataType, Executor::Host>;

public:
  using InputType  = FromExp;
  using OutputType = ToExp;
  using Scalar     = DataType;

  // Construct a transform between the two grids. `save` decides the two modes:
  //   save == true  : the kernel K is built once, STORED, and reused by every call.
  //   save == false : the kernel is NEVER stored (saves the matrix memory);
  //                   operator() computes each K[m,t] on the fly into the output.
  // `save` is fixed at construction (const) and cannot change afterwards.
  FourierTransform(TauGrid tau, MatsuGrid matsu, bool save = true)
    : tau_grid_(std::move(tau)), matsu_grid_(std::move(matsu)), save_(save)
  {
    if (save_)
      kernel_ = build_kernel();
  }

  // ----- Accessors -----
  const TauGrid&   tau_grid()   const { return tau_grid_; }
  const MatsuGrid& matsu_grid() const { return matsu_grid_; }
  const TensorOut& kernel()     const { return kernel_; }   // the stored kernel (valid iff saved())
  bool             saved()      const { return save_; }

  // The (m, t) element of the transformation kernel, computed directly from the
  // two grids:   K(m, t) = w_t . exp(i omega_m tau_t).   (m: Matsubara, t: tau.)
  DataType K(size_t m, size_t t) const {
    const double w   = tau_grid_.weights(t);
    const double arg = tau_grid_(t).value * matsu_grid_(m).value;
    return DataType(w * std::cos(arg), w * std::sin(arg));
  }

  // Apply the transform to a tau-represented function; return the Matsubara expansion.
  OutputType operator()(const InputType& input) const {
    const auto& in = input.data();
    const size_t T = tau_grid_.size();
    const size_t M = matsu_grid_.size();

    // The input's tau axis must be the fastest (innermost) and match the source grid.
    const size_t tau_idx = in.label_index(FromExp::dim_label);
    if (tau_idx != 0)
      throw std::invalid_argument("FourierTransform: the input tau axis must be the fastest axis");
    if (in.dims()[0].dim != T)
      throw std::invalid_argument("FourierTransform: input tau size ("
          + std::to_string(in.dims()[0].dim)
          + ") does not match the source tau grid (" + std::to_string(T) + ")");

    if (save_) {
      // The kernel is stored: use the (mixed-type) gemm to form
      //   out[m, sp] = sum_t kernel[m, t] * in[sp, t]
      // with X = kernel [matsu, tau] (tau slowest) and Y = input [tau, spatial] (tau
      // fastest); the free gemm allocates the output tensor for us.
      TensorOut out;
      gemm(kernel_, in, out, FromExp::dim_label, FromExp::dim_label);
      return OutputType(std::move(out), matsu_grid_);
    }

    // save == false: no stored kernel. Allocate the output and compute each K[m, t]
    // element-by-element on the fly, straight into the output. This uses no kernel
    // memory, at the cost of recomputing K.
    std::vector<TensorDim> out_dims;
    out_dims.reserve(in.dims().size());
    out_dims.push_back(TensorDim{ToExp::dim_label, M});      // Matsubara = fastest
    for (size_t i = 1; i < in.dims().size(); ++i)
      out_dims.push_back(in.dims()[i]);                        // unchanged spatial dims
    TensorOut out(out_dims);

    const size_t    S       = in.total_elements() / T;        // total number of spatial elements
    const InScalar* in_buf  = in.data();                       // in[sp, t]  -> in[t + T*sp]
    DataType*       out_buf = out.data();                      // out[m, sp] -> out[m + M*sp]
    for (size_t sp = 0; sp < S; ++sp)
      for (size_t m = 0; m < M; ++m) {
        DataType sum{};
        for (size_t t = 0; t < T; ++t)
          sum += K(m, t) * in_buf[t + T * sp];
        out_buf[m + M * sp] = sum;
      }
    return OutputType(std::move(out), matsu_grid_);
  }

private:
  TauGrid   tau_grid_;
  MatsuGrid matsu_grid_;
  const bool save_{};    // const: whether the kernel is stored; fixed at construction
  TensorOut kernel_;     // K[m, t]; allocated iff save_ is true

  // Build the kernel K[m,t] = w_t exp(i omega_m tau_t) into a [matsu, tau]-labeled
  // tensor (matsu fastest). Uses cos/sin on the real argument and the two-argument
  // (real, imag) constructor, so it works for any ComplexFloatingPoint scalar type.
  TensorOut build_kernel() const {
    const size_t M = matsu_grid_.size();
    const size_t T = tau_grid_.size();

    std::vector<TensorDim> k_dims = {
      TensorDim{ToExp::dim_label, M},     // Matsubara = fastest
      TensorDim{FromExp::dim_label, T}    // tau       = slowest
    };
    TensorOut ker(k_dims);
    DataType* ker_buf = ker.data();

    for (size_t t = 0; t < T; ++t)
      for (size_t m = 0; m < M; ++m)
        ker_buf[m + M * t] = K(m, t);

    return ker;
  }
};

}
