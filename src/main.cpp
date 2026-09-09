#include "main.hpp"

using namespace cppgw;

// Verify one (statistics, save) configuration of the Fourier transform against
// a hand-computed reference:   out[m, mu, nu] = sum_t w_t exp(i omega_m tau_t) in[mu, nu, t]
template <StatisticsTag S>
void fourier_tau_to_matsu_check(bool save, const InverseTemperature& beta,
                                size_t T, size_t halfN, size_t mu_n, size_t nu_n) {
  UniformImaginaryTimeGrid utau(beta, T);
  MatsubaraGrid<S>         mgrid(halfN, beta);
  const size_t M = mgrid.size();

  auto in_val = [](size_t mu, size_t nu, size_t t) -> double {
    return (double)(mu+1) + 10.0*(nu+1) + 100.0*(t+1);
  };

  // Build the input expansion  in[tau, mu, nu]  (tau = fastest)
  std::vector<TensorDim> in_dims;
  in_dims.push_back(TensorDim{utau.dim_label, T});
  in_dims.push_back(TensorDim{std::string("mu"), mu_n});
  in_dims.push_back(TensorDim{std::string("nu"), nu_n});
  Tensor<double, Executor::Host> inT(in_dims);
  {
    double* d = inT.data();
    for (size_t nu = 0; nu < nu_n; ++nu)
      for (size_t mu = 0; mu < mu_n; ++mu)
        for (size_t t = 0; t < T; ++t)
          d[t + T*mu + T*mu_n*nu] = in_val(mu, nu, t);   // strides: tau=1, mu=T, nu=T*mu_n
  }
  GridExpansionTau<double, UniformImaginaryTimeGrid> in(inT, utau);

  using OutExp = GridExpansionMatsubara<std::complex<double>, S>;
  FourierTransform<GridExpansionTau<double, UniformImaginaryTimeGrid>, OutExp, std::complex<double>>
      FT(utau, mgrid, save);
  // Storage behavior must follow the const save flag (Story 1):
  //   save=true  -> the kernel IS stored;   save=false -> the kernel is NEVER stored.
  {
    const bool stored = FT.kernel().allocated();
    if (stored != save)
      throw std::runtime_error("FourierTransform: kernel stored " + std::to_string((int)stored)
          + " but save=" + std::to_string((int)save));
    if (FT.saved() != save)
      throw std::runtime_error("FourierTransform: saved()=" + std::to_string((int)FT.saved())
          + " but save=" + std::to_string((int)save));
  }
  OutExp out = FT(in);

  // Hand-computed reference
  const std::complex<double>* odata = out.data().data();
  double maxerr = 0.0;
  for (size_t nu = 0; nu < nu_n; ++nu)
    for (size_t mu = 0; mu < mu_n; ++mu)
      for (size_t m = 0; m < M; ++m) {
        std::complex<double> ref{};
        for (size_t t = 0; t < T; ++t) {
          const double arg = utau(t).value * mgrid(m).value;
          ref += std::complex<double>(utau.weights(t)*std::cos(arg),
                                      utau.weights(t)*std::sin(arg)) * in_val(mu, nu, t);
        }
        const std::complex<double> got = odata[m + M*mu + M*mu_n*nu];   // strides: matsu=1, mu=M, nu=M*mu_n
        maxerr = std::max(maxerr, std::abs(got - ref));
      }
  std::cout << "  " << (std::is_same_v<S, Fermionic> ? "Fermionic " : "Bosonic  ")
            << "save=" << (save ? "true " : "false")
            << ": rank=" << out.data().rank()
            << " [matsu=" << M << ", mu=" << mu_n << ", nu=" << nu_n
            << "]  max|err|=" << maxerr
            << (maxerr < 1e-12 ? "  OK" : "  ** MISMATCH **") << std::endl;
}

int main(int argc, char** argv) {
  static_assert(Grid<MatsubaraGrid<Fermionic>>);
  auto beta = InverseTemperature(10.0);
  MatsubaraGrid<Fermionic> fgrid(4, beta);
  MatsubaraGrid<Bosonic> bgrid(4, beta);
  std::cout << "Fermionic grid (size " << fgrid.size() << "):\n";
  for(auto& p : fgrid.points()) 
    std::cout << p.value << " ";
  std::cout << std::endl << "Bosonic grid (size " << bgrid.size() << "):\n";
  for(auto& p : bgrid.points()) 
    std::cout << p.value << " ";
  std::cout << std::endl << "Bosonic points_no_zero:\n";
  for(auto& p : bgrid.points_no_zero()) 
    std::cout << p.value << " ";
  std::cout << std::endl;
  // operator(size_t) indexes into the full set (zero mode in the middle for Bosonic)
  std::cout << "bgrid(0)=" << bgrid(0).value
            << "  bgrid(4) [zero mode]=" << bgrid(4).value
            << "  bgrid(8)=" << bgrid(8).value << std::endl;

  std::cout << "Tensor test" << std::endl;
  size_t N = 6;
  std::vector<TensorDim> dims{{"x",3},{"y", 9}};
  Tensor<double,Executor::Host>  tensor({{"a",N},{"b", N*3}, {"c", N}});
  Tensor<std::complex<double>, Executor::Host> tensor2(dims);
  std::cout << "Total elements: " << tensor.total_elements() << std::endl;
  
  Tensor<double, Executor::Host> matrix1({{"x", 2},{"y", 2}});
  Tensor<double, Executor::Host> matrix2({{"x", 2},{"y", 2}});
  matrix1.allocate();
  matrix2.allocate();
  for(int i=0; i<matrix1.total_elements(); i++)
    *(matrix1.data() + i) = i+1;
  for(int i=0; i<matrix2.total_elements(); i++)
    *(matrix2.data() + i) = i+1;
  matrix1.print();
  Tensor<double, Executor::Host> matrix3;
  gemm(matrix1, matrix2, matrix3, "y", "x");
  matrix3.print();

  // ---- Mixed-type gemm:  real (X) x complex (Y) -> complex (Z) ----
  {
    using cplx = std::complex<double>;
    // X: [a=2 fast, k=3 slow]  real          A[i,k] = 3i + k + 1
    Tensor<double, Executor::Host>    A({{"a", 2}, {"k", 3}});
    // Y: [k=3 fast, b=2 slow]  complex      B[k,j] = (2k+j+1) + i*(2k+j+1)
    Tensor<cplx, Executor::Host>      B({{"k", 3}, {"b", 2}});
    {
      double* a = A.data();
      for (size_t k = 0; k < 3; ++k)
        for (size_t i = 0; i < 2; ++i)
          a[i + k*2] = 3.0*i + k + 1.0;                 // A: i fast (stride 1), k (stride 2)
    }
    {
      cplx* b = B.data();
      for (size_t j = 0; j < 2; ++j)
        for (size_t k = 0; k < 3; ++k) {
          const double v = 2.0*k + j + 1.0;
          b[k + j*3] = cplx(v, v);                      // B: k fast (stride 1), j (stride 3)
        }
    }
    Tensor<cplx, Executor::Host> C;   // gemm will shape + allocate [a, b]
    gemm(A, B, C, "k", "k");
    // expected   C[i,j] = sum_k A[i,k] * B[k,j]
    //            C[0][0]=22+22i C[0][1]=28+28i
    //            C[1][0]=49+49i C[1][1]=64+64i
    const cplx Cexp[2][2] = {{ cplx(22,22), cplx(28,28) },
                             { cplx(49,49), cplx(64,64) }};
    const cplx* c = C.data();
    double maxerr = 0.0;
    for (size_t j = 0; j < 2; ++j)
      for (size_t i = 0; i < 2; ++i)
        maxerr = std::max(maxerr, std::abs(c[i + j*2] - Cexp[i][j]));   // C: i fast, j (stride 2)
    std::cout << "\nMixed-type gemm (real x complex -> complex):\n";
    std::cout << C << std::endl;
    std::cout << " max|err|=" << maxerr << (maxerr < 1e-12 ? "  OK" : "  ** MISMATCH **") << std::endl;
  }

  // Test chebyshev machinery
  ChebyshevBasisImpl<numerics::float50> CB(5);
  std::vector<float> coeffs {0.0, 1.0, 5.0, 0.0, 0.0};
  std::cout << CB.evaluate(coeffs, 0.5) << std::endl;

  TensorShape shape1{{"u", 2}, {"v", 2}};
  ChebyshevExpansionTau<float> CET(shape1, 8);
  std::cout << "ChebExpTau data: " << CET.data() << std::endl;
  ChebyshevExpansionTau<double> CET2(matrix3, 8);
  std::cout << "ChebExpTau matrix1: " << CET2.data() << std::endl;

  // Test the GridExpansionTau machinery (a spatial Green's function on a uniform tau grid)
  UniformImaginaryTimeGrid utau(beta, 4);   // 4 grid points in (0, beta)
  std::cout << "\n--- GridExpansionTau ---" << std::endl;
  std::cout << "grid: " << utau.size() << " points" << std::endl;

  // (1) Build from a spatial tensor -> the grid axis is added as the fastest, data broadcast
  Tensor<double, Executor::Host> spatial({{"mu", 2}, {"nu", 2}});
  for (int i = 0; i < spatial.total_elements(); ++i)
    spatial.data()[i] = i + 1;
  GridExpansionTau<double, UniformImaginaryTimeGrid> GE_tensor(spatial, utau);
  std::cout << "From spatial tensor: " << GE_tensor.data() << std::endl;
  std::cout << "  grid point at index 0: tau = " << GE_tensor(0).value << std::endl;

  // (2) Build from a spatial shape -> allocated all-zero
  GridExpansionTau<double, UniformImaginaryTimeGrid> GE_shape(
      TensorShape{{"mu", 2}, {"nu", 2}}, utau);
  std::cout << "From spatial shape:  " << GE_shape.data() << std::endl;

  // Test the unified interface on the Matsubara (ImaginaryFrequency) space
  MatsubaraGrid<Fermionic> mgrid(4, beta);   // 8 Matsubara points
  Tensor<double, Executor::Host> mspatial({{"mu", 2}, {"nu", 2}});
  for (int i = 0; i < mspatial.total_elements(); ++i)
    mspatial.data()[i] = i + 1;

  std::cout << "\n--- GridExpansionMatsubara ---" << std::endl;
  std::cout << "grid: " << mgrid.size() << " points" << std::endl;
  GridExpansionMatsubara<double, Fermionic> GEM_t(mspatial, mgrid);
  std::cout << "From spatial tensor: " << GEM_t.data() << std::endl;
  GridExpansionMatsubara<double, Fermionic> GEM_s(TensorShape{{"mu", 2}, {"nu", 2}}, mgrid);
  std::cout << "From spatial shape:  " << GEM_s.data() << std::endl;

  std::cout << "\n--- ChebyshevExpansionMatsubara ---" << std::endl;
  std::cout << "coeff axis: " << 9 << " (order 8)" << std::endl;
  ChebyshevExpansionMatsubara<double, Fermionic> CEM_s(shape1, 8);
  std::cout << "From spatial shape:  " << CEM_s.data() << std::endl;
  ChebyshevExpansionMatsubara<double, Fermionic> CEM_t(matrix3, 8);
  std::cout << "From spatial tensor: " << CEM_t.data() << std::endl;

  // ---- FourierTransform (tau -> Matsubara), verified against a hand computation ----
  std::cout << "\n--- FourierTransform (tau -> Matsubara) ---" << std::endl;
  {
    const size_t T = 4, halfN = 2, mu_n = 2, nu_n = 3;   // halfN=2 -> 4 Fermionic / 5 Bosonic points
    fourier_tau_to_matsu_check<Fermionic>(/*save=*/true,  beta, T, halfN, mu_n, nu_n);
    fourier_tau_to_matsu_check<Fermionic>(/*save=*/false, beta, T, halfN, mu_n, nu_n);
    fourier_tau_to_matsu_check<Bosonic>  (/*save=*/true,  beta, T, halfN, mu_n, nu_n);
    fourier_tau_to_matsu_check<Bosonic>  (/*save=*/false, beta, T, halfN, mu_n, nu_n);
  }

  return 0;
}
