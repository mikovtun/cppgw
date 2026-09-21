#include "main.hpp"
#include "tests.hpp"
#include "input.hpp"

#include <highfive/highfive.hpp>

#include <filesystem>
#include <cmath>
#include <cstdio>

using namespace cppgw;

// Tensor::operator() is host-only (device storage cannot be touched from the
// host). It is constrained to `Executor::Host`, so any call on a `Tensor<...,Device>`
// is a compile error (it is not a candidate). We can positively static_assert the
// host case below; the device-negative case is intentionally NOT asserted here, because
// forming that call is itself ill-formed (no viable `operator()`), which a
// `static_assert(!requires(...))` cannot express without making this TU a hard error.
static_assert( requires(Tensor<double, Executor::Host> h)   { h(0, 0); });

namespace {
// Exception-checking helpers: true iff f() throws exactly T (any other throw, or no
// throw, returns false).
template <typename T, typename F>
bool throws_as(F&& f) {
  try { f(); }
  catch (const T&) { return true; }
  catch (...) { /* a throw, but of the wrong type */ }
  return false;
}
template <typename F> bool throws_oor(F&& f) { return throws_as<std::out_of_range>(std::forward<F>(f)); }
template <typename F> bool throws_ia (F&& f) { return throws_as<std::invalid_argument>(std::forward<F>(f)); }
} // namespace

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

namespace cppgw {

int run_tests() {
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
      // Write through Tensor::operator() (arg #k indexes dim k: i is "a", k is "k").
      for (size_t k = 0; k < 3; ++k)
        for (size_t i = 0; i < 2; ++i)
          A(i, k) = 3.0*i + k + 1.0;                   // A: i fast (stride 1), k (stride 2)
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
    double maxerr = 0.0;
    for (size_t j = 0; j < 2; ++j)
      for (size_t i = 0; i < 2; ++i)
        maxerr = std::max(maxerr, std::abs(C(i, j) - Cexp[i][j]));   // read back via operator()
    std::cout << "\nMixed-type gemm (real x complex -> complex):\n";
    std::cout << C << std::endl;
    std::cout << " max|err|=" << maxerr << (maxerr < 1e-12 ? "  OK" : "  ** MISMATCH **") << std::endl;
  }

  // ---- Tensor::operator(): read/write, bounds checks (Story 3) ----
  {
    // X dims (fast->slow): [x0=2, x1=3, x2=4];  storage offset = i + 2*j + 6*k.
    Tensor<double, Executor::Host> X({{"x0", 2}, {"x1", 3}, {"x2", 4}});
    for (size_t k = 0; k < 4; ++k)
      for (size_t j = 0; j < 3; ++j)
        for (size_t i = 0; i < 2; ++i)
          X(i, j, k) = 1.0 + 10.0*j + 100.0*k;
    // Read the whole array back; must equal the values written through operator().
    double maxerr = 0.0;
    for (size_t k = 0; k < 4; ++k)
      for (size_t j = 0; j < 3; ++j)
        for (size_t i = 0; i < 2; ++i)
          maxerr = std::max(maxerr, std::abs(X(i, j, k) - (1.0 + 10.0*j + 100.0*k)));
    // Const overload (read-only).
    const Tensor<double, Executor::Host>& Xc = X;
    const bool const_ok = (Xc(1, 2, 3) == 321.0);       // 1 + 2*10 + 3*100
    // Out-of-bounds indices must throw std::out_of_range.
    const bool oor_eq  = throws_oor([&]{ X(2, 0, 0); });   // i  == dim size
    const bool oor_mid = throws_oor([&]{ X(0, 3, 0); });   // j  out of range
    const bool oor_far = throws_oor([&]{ X(0, 0, 99); });  // k  far out
    const bool oor_neg = throws_oor([&]{ X(-1, 0, 0); });  // negative (signed) arg
    // Wrong index count throws std::invalid_argument.
    const bool bad_arity1 = throws_ia([&]{ X(0, 0); });    // too few
    const bool bad_arity2 = throws_ia([&]{ X(0, 0, 0, 0); });  // too many
    const bool all_ok = (maxerr < 1e-12) && const_ok && oor_eq && oor_mid
                     && oor_far && oor_neg && bad_arity1 && bad_arity2;
    std::cout << "\n--- Tensor operator() ---\n";
    std::cout << "  X(1,2,3)   = " << X(1, 2, 3) << "   (expected 321)\n";
    std::cout << "  const X(1,2,3) = " << Xc(1, 2, 3) << "   (expected 321)\n";
    std::cout << "  full read/write sweep max|err| = " << maxerr
              << (maxerr < 1e-12 ? "  OK" : "  ** MISMATCH **") << std::endl;
    std::cout << "  out_of_range (i==dim size / j oob / k oob / negative): "
              << (oor_eq && oor_mid && oor_far && oor_neg
                  ? "  OK" : "  ** FAIL **") << std::endl;
    std::cout << "  wrong index count throws invalid_argument: "
              << (bad_arity1 && bad_arity2 ? "  OK" : "  ** FAIL **") << std::endl;
    if (!all_ok) throw std::runtime_error("Tensor::operator() checks failed");
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
  std::cout << "From spatial tensor:  " << CEM_t.data() << std::endl;

  // ---- FourierTransform (tau -> Matsubara), verified against a hand computation ----
  std::cout << "\n--- FourierTransform (tau -> Matsubara) ---" << std::endl;
  {
    const size_t T = 4, halfN = 2, mu_n = 2, nu_n = 3;   // halfN=2 -> 4 Fermionic / 5 Bosonic points
    fourier_tau_to_matsu_check<Fermionic>(/*save=*/true,  beta, T, halfN, mu_n, nu_n);
    fourier_tau_to_matsu_check<Fermionic>(/*save=*/false, beta, T, halfN, mu_n, nu_n);
    fourier_tau_to_matsu_check<Bosonic>  (/*save=*/true,  beta, T, halfN, mu_n, nu_n);
    fourier_tau_to_matsu_check<Bosonic>  (/*save=*/false, beta, T, halfN, mu_n, nu_n);
  }

  // ---- Story 03: analytic convergence tests for the Fourier transform ----
  run_fourier_convergence_tests();

  // ---- Story 04: inverse (Matsubara -> tau) Fourier transform ----
  run_inverse_fourier_tests();

  // ---- Story 06/06.1: index symmetry, TensorBackend seam, and transpose ---
  run_tensor_symmetry_tests();
  run_tensor_solve_tests();

  // ---- Story 05: InputCatalog, input parsing, and semantic GF2 loading ----
  run_input_data_loading_tests();

  return 0;
}

using cplx = std::complex<double>;

namespace {
// Exception-checking helper for logic_error-derived validation failures.
template <typename F>
bool throws_le(F&& f, const char* expected_substring = "") {
  try { f(); }
  catch (const std::logic_error& e) {
    if (expected_substring == nullptr)
      return true;
    return std::string(e.what()).find(expected_substring) != std::string::npos;
  }
  catch (...) { /* a throw, but of the wrong type */ }
  return false;
}

// Exact (or tightly bounded) elementwise comparison of two Tensors: same rank,
// same total size, and all linear elements equal within tol.
template <typename T>
bool tensor_allclose(const Tensor<T, Executor::Host>& a, const Tensor<T, Executor::Host>& b) {
  if (a.rank() != b.rank() || a.total_elements() != b.total_elements())
    return false;
  for (size_t n = 0; n < a.total_elements(); ++n)
    if (std::abs(a.linear(n) - b.linear(n)) > 1e-12)
      return false;
  return true;
}
} // namespace

// ============================================================================
//  Story 06 — tensor index symmetry, the TensorBackend seam, and the no-op
//  transpose contract.
// ============================================================================
int run_tensor_symmetry_tests() {

  // ------------------------------------------------------------------
  //  0. The TensorBackend contract is satisfied for host scalars: the
  //     seam's required ops (fill/zero/scale/conjugate + mixed-type gemm)
  //     are statically present. The frontend programs ONLY against this.
  // ------------------------------------------------------------------
  static_assert(TensorBackendOps<double, Executor::Host>);
  static_assert(TensorBackendOps<std::complex<double>, Executor::Host>);
  static_assert(TensorBackendOps<boost::multiprecision::cpp_complex_50, Executor::Host>);
  static_assert(!TensorBackendOps<int, Executor::Host>);

  // ------------------------------------------------------------------
  //  1. Construction & invariants
  // ------------------------------------------------------------------
  {
    using T2 = Tensor<double, Executor::Host>;
    // (a) Plain/legacy path: TensorDim{label,dim} (null symmetry) still
    //     constructs, and unique plain labels still hold.
    T2 t({TensorDim{"a", 2}, TensorDim{"b", 3}});
    if (!(t.rank() == 2 && null_symgroup(t.dims()[0].symmetry)
          && null_symgroup(t.dims()[1].symmetry)))
      throw std::runtime_error("symmetry tests: plain TensorDim lost its plain status");

    // (b) Two plain "ao" axes (no group) -> a label spanning groups is an error.
    if (!throws_ia([&]{ Tensor<double, Executor::Host>({{"ao", 50}, {"ao", 50}}); }))
      throw std::runtime_error("symmetry tests: duplicate plain label must throw");
    // Same label, TWO different groups -> error.
    {
      auto g1 = Symmetric();
      auto g2 = Symmetric();                        // distinct group, same kind
      if (!throws_ia([&]{ Tensor<double, Executor::Host>({TensorDim{"ao", 50, g1}, TensorDim{"ao", 50, g2}}); }))
        throw std::runtime_error("symmetry tests: label spanning two groups must throw");
      // One plain + one bound, same label -> error.
      if (!throws_ia([&]{ Tensor<double, Executor::Host>({TensorDim{"ao", 50}, TensorDim{"ao", 50, g1}}); }))
        throw std::runtime_error("symmetry tests: plain + bound duplicate label must throw");
    }
    // SymGroup members with mismatched sizes -> error.
    {
      auto g = Symmetric();
      if (!throws_ia([&]{ Tensor<double, Executor::Host>({TensorDim{"ao", 50, g}, TensorDim{"ao", 40, g}}); }))
        throw std::runtime_error("symmetry tests: matched-size rule violated without a throw");
      // Mismatched LABEL inside one group -> error.
      if (!throws_ia([&]{ Tensor<double, Executor::Host>({TensorDim{"ao", 50, g}, TensorDim{"oo", 50, g}}); }))
        throw std::runtime_error("symmetry tests: matched-label rule violated without a throw");
    }

    // (c) eri3: duplicate label WITH one shared SymGroup -> constructs, rank 3,
    //     the two ao axes carry the SAME handle.
    auto g3 = Symmetric();
    Tensor<double, Executor::Host> eri3(
        { {TensorDimLabel("ri"), 120 },
          { TensorDimLabel("ao"), 50, g3 },
          { TensorDimLabel("ao"), 50, g3 } });
    if (eri3.rank() != 3
        || !same_symgroup(eri3.dims()[1].symmetry, eri3.dims()[2].symmetry)
        || !same_symgroup(g3, eri3.dims()[1].symmetry))
      throw std::runtime_error("symmetry tests: eri3 twin handles are not identical");

    // (d) eri4: two INDEPENDENT symmetric pairs (g1, g2 distinct, same kind).
    auto g1 = Symmetric();
    auto g2 = Symmetric();
    Tensor<cplx, Executor::Host> eri4(
        { {TensorDimLabel("ao1"), 50, g1}, { TensorDimLabel("ao1"), 50, g1 },
          { TensorDimLabel("ao2"), 50, g2}, { TensorDimLabel("ao2"), 50, g2 } });
    if (same_symgroup(g1, g2))
      throw std::runtime_error("symmetry tests: distinct SymmetryGroup instances compared equal");
    if (g1->kind() != SymKind::Symmetric || g2->kind() != SymKind::Symmetric)
      throw std::runtime_error("symmetry tests: Symmetric() factory mislabeled");
    if (eri4.label_indices("ao1").size() != 2 || eri4.label_indices("ao2").size() != 2)
      throw std::runtime_error("symmetry tests: eri4 family cardinality wrong");

    // (e) Handle identity: a shared_ptr COPY of the same handle is the same
    //     group; distinct factories are not. Same-group copies bind the label.
    {
      auto h1 = Hermitian();
      auto h1copy = h1;                    // shared_ptr copy: same object
      if (!same_symgroup(h1, h1copy) || !null_symgroup(nullptr))
        throw std::runtime_error("symmetry tests: handle-identity copy rule broken");
      auto h2 = Hermitian();               // fresh factory: distinct group
      if (same_symgroup(h1, h2))
        throw std::runtime_error("symmetry tests: two Hermitian() factories are the same group");
      // And a copy can be used to construct a valid bound tensor.
      Tensor<double, Executor::Host> hc({TensorDim{"ao", 50, h1}, TensorDim{"ao", 50, h1copy}});
      (void)hc;
    }
  }

  // ------------------------------------------------------------------
  //  2. Symmetry queries: label indices and no-op transpose contracts
  // ------------------------------------------------------------------
  {
    // eri3-style duplicate label: label_indices / label_index / has_label.
    auto g = Symmetric();
    Tensor<double, Executor::Host> eri3(
        { {TensorDimLabel("ri"), 3 },
          { TensorDimLabel("ao"), 4, g }, { TensorDimLabel("ao"), 4, g } });
    {
      auto idx = eri3.label_indices(TensorDimLabel("ao"));
      if (idx.size() != 2 || idx[0] != 1 || idx[1] != 2)
        throw std::runtime_error("symmetry tests: label_indices('ao') != {1,2}");
      if (eri3.label_index(TensorDimLabel("ao")) != 1)
        throw std::runtime_error("symmetry tests: label_index('ao') != first position");
      if (!eri3.has_label(TensorDimLabel("ao")))
        throw std::runtime_error("symmetry tests: has_label('ao') is false");
      if (eri3.has_label(TensorDimLabel("nope")))
        throw std::runtime_error("symmetry tests: has_label('nope') is true");
    }

    // Symmetric (real) pair: transpose(1,2) is elementwise-identical, a copy
    // (not a permutation), produced via the identity path (no restride).
    {
      using T2 = Tensor<double, Executor::Host>;
      for (size_t k = 0; k < 4; ++k)
        for (size_t j = 0; j < 4; ++j)
          for (size_t i = 0; i < 3; ++i)
            eri3(i, j, k) = 1.0 + i + 10*j + 100*k;
      auto r12 = eri3.transpose(1, 2);
      auto r21 = eri3.transpose(2, 1);
      if (!tensor_allclose(eri3, r12) || !tensor_allclose(eri3, r21))
        throw std::runtime_error("symmetry tests: symmetric transpose != identity");
      // Symmetric in its arguments, and NOT aliasing the source:
      if (eri3.data() == r12.data())
        throw std::runtime_error("symmetry tests: transpose returned aliased storage");
      if (r12(0,0,0) != eri3(0,0,0) || r21(2,3,1) != eri3(2,3,1))
        throw std::runtime_error("symmetry tests: symmetric transpose value mismatch");
      r12(1, 1, 1) = -999.0;
      if (eri3(1, 1, 1) == -999.0)
        throw std::runtime_error("symmetry tests: transpose aliasing corrupted the source");
      // ri<->ao (0,1) is NOT a bound pair (ri is plain), so this is now a
      // physical transpose. The output order is [ao, ri, ao] and
      // out(j,i,k) == eri3(i,j,k).
      auto r01 = eri3.transpose(0, 1);
      if (r01.dims()[0].label != TensorDimLabel("ao")
          || r01.dims()[0].dim != 4
          || r01.dims()[1].label != TensorDimLabel("ri")
          || r01.dims()[1].dim != 3
          || r01.dims()[2].label != TensorDimLabel("ao")
          || r01.dims()[2].dim != 4)
        throw std::runtime_error("transpose tests: distinct-axis transpose metadata wrong");
      for (size_t k = 0; k < 4; ++k)
        for (size_t i = 0; i < 3; ++i)
          for (size_t j = 0; j < 4; ++j)
            if (r01(j, i, k) != eri3(i, j, k))
              throw std::runtime_error("transpose tests: distinct-axis transpose values wrong");
      // Out-of-range position remains an error.
      if (!throws_le([&]{ eri3.transpose(1, 9); }))
        throw std::runtime_error("symmetry tests: transpose oob did not throw");
    }

    // mo_coeff {ao, mo}: no bound pair -> a physical rectangular transpose.
    {
      Tensor<double, Executor::Host> mo({TensorDim{"ao", 3}, TensorDim{"mo", 2}});
      if (mo.label_indices("ao").size() != 1 || mo.label_indices("mo").size() != 1)
        throw std::runtime_error("symmetry tests: mo_coeff label indices wrong");
      for (size_t j = 0; j < 2; ++j)
        for (size_t i = 0; i < 3; ++i)
          mo(i, j) = 10.0 * i + j;
      auto mt = mo.transpose(0, 1);
      if (mt.dims()[0].label != TensorDimLabel("mo") || mt.dims()[0].dim != 2
          || mt.dims()[1].label != TensorDimLabel("ao") || mt.dims()[1].dim != 3)
        throw std::runtime_error("transpose tests: rectangular transpose metadata wrong");
      for (size_t j = 0; j < 2; ++j)
        for (size_t i = 0; i < 3; ++i)
          if (mt(j, i) != mo(i, j))
            throw std::runtime_error("transpose tests: rectangular transpose values wrong");
      if (mt.data() == mo.data())
        throw std::runtime_error("transpose tests: rectangular transpose aliased storage");
    }

    // General permutation: a non-self-inverse rank-three permutation must
    // follow permutation[new_axis] = old_axis, not the inverse convention.
    {
      Tensor<double, Executor::Host> src({TensorDim{"a", 2}, TensorDim{"b", 3}, TensorDim{"c", 4}});
      for (size_t c = 0; c < 4; ++c)
        for (size_t b = 0; b < 3; ++b)
          for (size_t a = 0; a < 2; ++a)
            src(a, b, c) = 100.0 * a + 10.0 * b + c;

      auto p = src.permute_axes({2, 0, 1});
      if (p.rank() != 3
          || p.dims()[0].label != TensorDimLabel("c") || p.dims()[0].dim != 4
          || p.dims()[1].label != TensorDimLabel("a") || p.dims()[1].dim != 2
          || p.dims()[2].label != TensorDimLabel("b") || p.dims()[2].dim != 3)
        throw std::runtime_error("transpose tests: general permutation metadata wrong");
      for (size_t b = 0; b < 3; ++b)
        for (size_t a = 0; a < 2; ++a)
          for (size_t c = 0; c < 4; ++c)
            if (p(c, a, b) != src(a, b, c))
              throw std::runtime_error("transpose tests: general permutation values wrong");

      // The inverse of [2,0,1] under the documented convention is [1,2,0].
      auto roundtrip = p.permute_axes({1, 2, 0});
      if (!tensor_allclose(src, roundtrip))
        throw std::runtime_error("transpose tests: permutation round trip failed");
      if (p.data() == src.data())
        throw std::runtime_error("transpose tests: general permutation aliased storage");
    }

    // The raw physical permutation is deliberately distinct from the
    // Story-06 semantic same-group transpose shortcut.
    {
      auto g = Hermitian();
      Tensor<cplx, Executor::Host> h(
          {TensorDim{"x", 2, g}, TensorDim{"x", 2, g}});
      h(0, 0) = cplx(1.0, 2.0);
      h(1, 0) = cplx(3.0, 4.0);
      h(0, 1) = cplx(5.0, 6.0);
      h(1, 1) = cplx(7.0, 8.0);
      auto semantic = h.transpose(0, 1);
      auto physical = h.permute_axes({1, 0});
      for (size_t n = 0; n < h.total_elements(); ++n) {
        if (semantic.linear(n) != std::conj(h.linear(n)))
          throw std::runtime_error("transpose tests: semantic Hermitian transpose changed");
      }
      if (physical(0, 1) != h(1, 0) || physical(1, 0) != h(0, 1))
        throw std::runtime_error("transpose tests: raw symmetry-bound permutation wrong");
      if (physical.linear(0) == semantic.linear(0)
          && physical.linear(1) == semantic.linear(1))
        throw std::runtime_error("transpose tests: raw permutation collapsed into semantic shortcut");
    }

    // Invalid permutations and edge cases.
    {
      Tensor<double, Executor::Host> src({TensorDim{"a", 2}, TensorDim{"b", 3}});
      if (!throws_ia([&]{ src.permute_axes({0}); }))
        throw std::runtime_error("transpose tests: short permutation did not throw");
      if (!throws_ia([&]{ src.permute_axes({0, 0}); }))
        throw std::runtime_error("transpose tests: duplicate permutation axis did not throw");
      if (!throws_le([&]{ src.permute_axes({0, 2}); }))
        throw std::runtime_error("transpose tests: out-of-range permutation axis did not throw");
      if (!throws_le([&]{ src.transpose(0, 2); }))
        throw std::runtime_error("transpose tests: out-of-range transpose axis did not throw");

      std::vector<TensorDim> scalar_dims;
      Tensor<double, Executor::Host> scalar(scalar_dims);
      scalar.linear(0) = 42.0;
      auto scalar_copy = scalar.permute_axes({});
      if (scalar_copy.rank() != 0 || scalar_copy.linear(0) != 42.0
          || scalar_copy.data() == scalar.data())
        throw std::runtime_error("transpose tests: rank-zero permutation failed");

      Tensor<double, Executor::Host> empty;
      auto empty_copy = empty.permute_axes({});
      if (empty_copy.rank() != 0 || empty_copy.total_elements() != 0
          || empty_copy.allocated())
        throw std::runtime_error("transpose tests: unallocated empty permutation failed");

      Tensor<double, Executor::Host> zero({TensorDim{"a", 0}, TensorDim{"b", 2}});
      auto zero_permuted = zero.permute_axes({1, 0});
      if (zero_permuted.dims()[0].label != TensorDimLabel("b")
          || zero_permuted.dims()[1].label != TensorDimLabel("a")
          || zero_permuted.total_elements() != 0
          || zero_permuted.allocated())
        throw std::runtime_error("transpose tests: zero-sized permutation failed");
    }

    // Antisymmetric pair (real): transpose(i,j) == -T elementwise.
    {
      using T2 = Tensor<double, Executor::Host>;
      auto a = Antisymmetric();
      T2 at({TensorDim{"x", 2, a}, TensorDim{"x", 2, a}});
      for (int i = 0; i < 4; ++i)  at.data()[i] = 1.0 + i;
      auto r = at.transpose(0, 1);
      // Direct check: all elements of r == -all elements of at (the sign flip).
      for (size_t n = 0; n < at.total_elements(); ++n)
        if (r.linear(n) != -at.linear(n))
          throw std::runtime_error("symmetry tests: antisymmetric transpose != -T");
      auto r2 = at.transpose(1, 0);
      for (size_t n = 0; n < at.total_elements(); ++n)
        if (r2.linear(n) != -at.linear(n))
          throw std::runtime_error("symmetry tests: antisymmetric transpose(1,0) != -T");
    }

    // Complex Hermitian pair: transpose(i,j) == conj(T) elementwise.
    {
      using T2 = Tensor<cplx, Executor::Host>;
      auto h = Hermitian();
      T2 ht({TensorDim{"ao", 3, h}, TensorDim{"ao", 3, h}});
      for (size_t n = 0; n < 9; ++n)
        ht.linear(n) = cplx(1.0*n, -2.0*n + 0.5);       // nonzero imag part
      auto r = ht.transpose(0, 1);
      if (r.total_elements() != 9)
        throw std::runtime_error("symmetry tests: hermitian transpose size wrong");
      for (size_t n = 0; n < 9; ++n) {
        if (r.linear(n) != std::conj(ht.linear(n)))
          throw std::runtime_error("symmetry tests: complex hermitian transpose != conj(T)");
      }
      auto r2 = ht.transpose(1, 0);
      // Symmetric in its arguments: transpose(1,0) equals transpose(0,1).
      for (size_t n = 0; n < 9; ++n)
        if (std::abs(r2.linear(n) - r.linear(n)) > 1e-15)
          throw std::runtime_error("symmetry tests: hermitian transpose not symmetric in args");
    }

    // Hermitian on a REAL scalar reduces to Symmetric: transpose == T.
    {
      using T2 = Tensor<double, Executor::Host>;
      auto h = Hermitian();
      T2 hcore({TensorDim{"ao", 4, h}, TensorDim{"ao", 4, h}});
      for (size_t n = 0; n < 16; ++n) hcore.linear(n) = 3.0*n - 1.0;
      auto r = hcore.transpose(0, 1);
      if (!tensor_allclose(hcore, r))
        throw std::runtime_error("symmetry tests: real hermitian transpose != T");
    }
  }

  // ------------------------------------------------------------------
  //  3. gemm on symmetric tensors (per-position bookkeeping of duplicate labels)
  // ------------------------------------------------------------------
  {
    // X (eri3-style):  {ri=2, ao=3, ao=3} with the two ao axes one Symmetric family.
    //                  X(i, j, k) = 1 + i + 10*j + 100*k
    // Y:               {ao=3, x=2};                Y(k, x) = 3 + 5*k + 7*x
    // C = X x_{ao}:    {ri=2, ao=3, x=2}
    //   C(i, j, x)   = Σ_k  X(i, j, k) * Y(k, x)
    {
      auto g = Symmetric();
      Tensor<double, Executor::Host> X(
          { TensorDim{"ri", 2}, TensorDim{"ao", 3, g}, TensorDim{"ao", 3, g } });
      for (size_t k = 0; k < 3; ++k)
        for (size_t j = 0; j < 3; ++j)
          for (size_t i = 0; i < 2; ++i)
            X(i, j, k) = 1.0 + i + 10.0*j + 100.0*k;

      Tensor<double, Executor::Host> Y({TensorDim{"ao", 3}, TensorDim{"x", 2}});
      for (size_t x = 0; x < 2; ++x)
        for (size_t k = 0; k < 3; ++k)
          Y(k, x) = 3.0 + 5.0*k + 7.0*x;

      Tensor<double, Executor::Host> C;                      // gemm shapes + allocates
      gemm(X, Y, C, "ao", "ao");

      // Output shape: X minus its last dim, then Y minus its first dim
      //   = {ri=2, ao=3, x=2};  surviving ao keeps its position (index 1) AND
      //   keeps its original SymGroup (per-position bookkeeping).
      if (C.rank() != 3)
        throw std::runtime_error("symmetry tests: gemm(X,Y) wrong rank");
      if (!(C.dims()[0].label == TensorDimLabel("ri") && C.dims()[0].dim == 2
            && C.dims()[1].label == TensorDimLabel("ao") && C.dims()[1].dim == 3
            && C.dims()[2].label == TensorDimLabel("x") && C.dims()[2].dim == 2))
        throw std::runtime_error("symmetry tests: gemm(X,Y) wrong output labels/sizes");
      if (!same_symgroup(C.dims()[1].symmetry, g))
        throw std::runtime_error("symmetry tests: gemm lost the surviving twin's SymGroup");

      double maxerr = 0.0;
      for (size_t x = 0; x < 2; ++x)
        for (size_t j = 0; j < 3; ++j)
          for (size_t i = 0; i < 2; ++i) {
            double ref = 0.0;
            for (size_t k = 0; k < 3; ++k)
              ref += (1.0 + i + 10.0*j + 100.0*k) * (3.0 + 5.0*k + 7.0*x);
            maxerr = std::max(maxerr, std::abs(C(i, j, x) - ref));
          }
      if (maxerr > 1e-12)
        throw std::runtime_error("symmetry tests: gemm on symmetric tensor numerically wrong (maxerr "
            + std::to_string(maxerr) + ")");

      // Wrong gemm order / nonexistent label still throw on symmetric tensors.
      if (!throws_ia([&]{ gemm(Y, X, C, "ao", "ao"); }))      // "ao" is not Y's last dim
        throw std::runtime_error("symmetry tests: gemm(Y,X) should throw (label not last)");
      if (!throws_ia([&]{ gemm(X, Y, C, "zz", "ao"); }))      // label not on X's last dim
        throw std::runtime_error("symmetry tests: gemm(X,Y,'zz','ao') should throw");
    }

    // Regression: the original hand-written mixed-type gemm values must land in
    // the SAME place through the new seam (Story 2 test, re-asserted here)
    //   C[0][0]=22+22i C[0][1]=28+28i  C[1][0]=49+49i C[1][1]=64+64i
    {
      Tensor<double, Executor::Host>    A({{"a", 2}, {"k", 3}});
      Tensor<cplx,  Executor::Host>    B({{"k", 3}, {"b", 2}});
      for (size_t k = 0; k < 3; ++k)
        for (size_t i = 0; i < 2; ++i)
          A(i, k) = 3.0*i + k + 1.0;
      for (size_t j = 0; j < 2; ++j)
        for (size_t k = 0; k < 3; ++k) {
          const double v = 2.0*k + j + 1.0;
          B.linear(k + j*3) = cplx(v, v);
        }
      Tensor<cplx, Executor::Host> C;
      gemm(A, B, C, "k", "k");
      const cplx Cexp[2][2] = {{ cplx(22,22), cplx(28,28) }, { cplx(49,49), cplx(64,64) }};
      for (size_t j = 0; j < 2; ++j)
        for (size_t i = 0; i < 2; ++i)
          if (std::abs(C(i, j) - Cexp[i][j]) > 1e-12)
            throw std::runtime_error("symmetry tests: regression gemm value mismatch");
    }
  }

  // ------------------------------------------------------------------
  //  4. TensorBackend seam: host unary ops (fill/zero/scale/conjugate) + gemm
  // ------------------------------------------------------------------
  {
    // Real scalar type.
    {
      using B = TensorBackend<double, Executor::Host>;
      TensorBuffer<double, Executor::Host> buf(3);
      B::fill(buf, 2.0);
      for (int i = 0; i < 3; ++i) if (buf.data()[i] != 2.0)
        throw std::runtime_error("symmetry tests: backend fill (real) wrong");
      B::zero(buf);
      for (int i = 0; i < 3; ++i) if (buf.data()[i] != 0.0)
        throw std::runtime_error("symmetry tests: backend zero (real) wrong");
      B::fill(buf, 3.0);
      B::scale(buf, -2.0);
      for (int i = 0; i < 3; ++i) if (buf.data()[i] != -6.0)
        throw std::runtime_error("symmetry tests: backend scale (real) wrong");
      B::conjugate(buf);                                    // no-op on real
      for (int i = 0; i < 3; ++i) if (buf.data()[i] != -6.0)
        throw std::runtime_error("symmetry tests: backend conjugate (real) not a no-op");
    }
    // Complex scalar type.
    {
      using B = TensorBackend<cplx, Executor::Host>;
      TensorBuffer<cplx, Executor::Host> buf(2);
      B::fill(buf, cplx(1.0, -2.0));
      B::conjugate(buf);
      if (buf.data()[0] != cplx(1.0, 2.0) || buf.data()[1] != cplx(1.0, 2.0))
        throw std::runtime_error("symmetry tests: backend conjugate (complex) wrong");
      B::scale(buf, cplx(0.0, 1.0));                         // multiply by i
      if (buf.data()[0] != cplx(-2.0, 1.0))
        throw std::runtime_error("symmetry tests: backend scale (complex) wrong");
      B::zero(buf);
      if (buf.data()[0] != cplx(0.0, 0.0))
        throw std::runtime_error("symmetry tests: backend zero (complex) wrong");
    }
    // Host gemm matches the hand-written reference exactly (new home, same values).
    {
      using B = TensorBackend<double, Executor::Host>;
      const size_t M = 2, Kp = 3, N = 2;
      TensorBuffer<double, Executor::Host> A(M * Kp), Bbuf(Kp * N), C(M * N);
      // A(i,k) = i + 2k + 1       B(k,j) = 3k + j
      for (size_t k = 0; k < Kp; ++k)
        for (size_t i = 0; i < M; ++i)
          A.data()[i + k * M] = 1.0 + i + 2.0*k;
      for (size_t j = 0; j < N; ++j)
        for (size_t k = 0; k < Kp; ++k)
          Bbuf.data()[k + j * Kp] = 3.0*k + j;
      B::gemm(A, M, Kp, Bbuf, N, C);
      for (size_t j = 0; j < N; ++j)
        for (size_t i = 0; i < M; ++i) {
          double ref = 0.0;
          for (size_t k = 0; k < Kp; ++k)
            ref += (1.0 + i + 2.0*k) * (3.0*k + j);
          if (std::abs(C.data()[i + j * M] - ref) > 1e-12)
            throw std::runtime_error("symmetry tests: backend gemm value mismatch");
        }
    }
  }

  // ------------------------------------------------------------------
  //  5. GF2 linkage (Stories 05/07): the physics-level construction with the
  //     new API:  hcore{ao,ao}=Hermitian, mo_coeff{ao,mo}=plain,
  //     eri3{ri,ao,ao}=Symmetric. No numerics — shape/label/symmetry only.
  // ------------------------------------------------------------------
  {
    using T2 = Tensor<double, Executor::Host>;
    auto hcore_g    = Hermitian();
    auto eri3_g     = Symmetric();
    T2 hcore   ({TensorDim{"ao", 30, hcore_g}, TensorDim{"ao", 30, hcore_g}});
    T2 mo_coeff({TensorDim{"ao", 30},              TensorDim{"mo", 12}});
    T2 eri3    ({TensorDim{"ri", 120}, TensorDim{"ao", 30, eri3_g}, TensorDim{"ao", 30, eri3_g}});

    std::cout << "\n--- Story 06: GF2 inputs (new symmetry API) ---" << std::endl;
    // These tensors are constructed for their shape/label/symmetry, not their
    // numerics (Story 07 reads in the data). Print metadata only - the full data
    // block is all-zero and just noise.
    hcore   .print(std::cout, /*with_data=*/false);
    mo_coeff.print(std::cout, /*with_data=*/false);
    eri3    .print(std::cout, /*with_data=*/false);

    if (hcore.rank() != 2 || hcore.dims()[0].dim != 30 || hcore.dims()[1].dim != 30
        || !same_symgroup(hcore.dims()[0].symmetry, hcore.dims()[1].symmetry)
        || hcore.dims()[0].symmetry->kind() != SymKind::Hermitian)
      throw std::runtime_error("symmetry tests: GF2 hcore shape/symmetry wrong");
    if (hcore.transpose(0, 1).rank() != 2)                  // real hermitian: no-op
      throw std::runtime_error("symmetry tests: GF2 hcore transpose failed");

    if (mo_coeff.rank() != 2 || !null_symgroup(mo_coeff.dims()[0].symmetry)
        || !null_symgroup(mo_coeff.dims()[1].symmetry))
      throw std::runtime_error("symmetry tests: GF2 mo_coeff shape/symmetry wrong");
    auto mo_transposed = mo_coeff.transpose(0, 1);
    if (mo_transposed.dims()[0].label != TensorDimLabel("mo")
        || mo_transposed.dims()[1].label != TensorDimLabel("ao"))
      throw std::runtime_error("symmetry tests: GF2 mo_coeff transpose metadata wrong");

    if (eri3.rank() != 3 || eri3.label_indices("ao").size() != 2
        || eri3.dims()[1].symmetry->kind() != SymKind::Symmetric)
      throw std::runtime_error("symmetry tests: GF2 eri3 shape/symmetry wrong");
  }

  return 0;
}

int run_tensor_solve_tests() {
  using Host = Executor::Host;
  using T = Tensor<double, Host>;
  const SolveAxis axis{TensorDimLabel("row"), TensorDimLabel("col")};

  // LAPACK path, two RHS columns: A X = B with row-fast/column-major storage.
  {
    T A({{"row", 2}, {"col", 2}});
    T B({{"row", 2}, {"rhs", 2}});
    A(0, 0) = 2.0; A(1, 0) = 1.0;
    A(0, 1) = 1.0; A(1, 1) = 3.0;
    B(0, 0) = 1.0; B(1, 0) = 2.0;
    B(0, 1) = 3.0; B(1, 1) = 4.0;
    const T A0 = A, B0 = B;
    T X;
    solve(A, B, X, {axis});
    for (size_t j = 0; j < 2; ++j)
      for (size_t i = 0; i < 2; ++i) {
        double value = 0.0;
        for (size_t k = 0; k < 2; ++k) value += A0(i, k) * X(k, j);
        if (std::abs(value - B0(i, j)) > 1e-12)
          throw std::runtime_error("solve tests: double LAPACK residual is wrong");
      }
    if (A(0, 0) != A0(0, 0) || B(1, 1) != B0(1, 1))
      throw std::runtime_error("solve tests: public solve modified its inputs");
  }

  // The complex-double path is LAPACK zgesv (not the generic QR fallback).
  {
    using C = std::complex<double>;
    Tensor<C, Host> A({{"row", 2}, {"col", 2}});
    Tensor<C, Host> B({{"row", 2}, {"rhs", 1}}), X;
    A(0, 0) = C(2, 1); A(1, 0) = C(1, -1);
    A(0, 1) = C(0, 1); A(1, 1) = C(3, 0);
    B(0, 0) = C(1, 2); B(1, 0) = C(4, -1);
    const auto A0 = A, B0 = B;
    solve(A, B, X, {axis});
    for (size_t i = 0; i < 2; ++i) {
      C value{};
      for (size_t k = 0; k < 2; ++k) value += A0(i, k) * X(k, 0);
      if (std::abs(value - B0(i, 0)) > 1e-12)
        throw std::runtime_error("solve tests: complex LAPACK residual is wrong");
    }
  }

  // Multi-axis solve space: verifies fastest-to-slowest flattening order.
  {
    T A({{"row_spin", 2}, {"row_ao", 2}, {"col_spin", 2}, {"col_ao", 2}});
    T B({{"row_spin", 2}, {"row_ao", 2}, {"rhs", 2}});
    for (size_t rs = 0; rs < 2; ++rs)
      for (size_t ra = 0; ra < 2; ++ra) {
        const size_t flat = rs + 2 * ra;
        for (size_t cs = 0; cs < 2; ++cs)
          for (size_t ca = 0; ca < 2; ++ca)
            A(rs, ra, cs, ca) = (flat == cs + 2 * ca) ? 10.0 + flat : 0.0;
        B(rs, ra, 0) = 100.0 + flat;
        B(rs, ra, 1) = 200.0 + flat;
      }
    T X;
    solve(A, B, X, {SolveAxis{"row_spin", "col_spin"}, SolveAxis{"row_ao", "col_ao"}});
    if (X.rank() != 3 || X.dims()[0].label != TensorDimLabel("col_spin")
        || X.dims()[1].label != TensorDimLabel("col_ao")
        || X.dims()[2].label != TensorDimLabel("rhs"))
      throw std::runtime_error("solve tests: multi-axis output shape is wrong");
    for (size_t cs = 0; cs < 2; ++cs)
      for (size_t ca = 0; ca < 2; ++ca)
        for (size_t r = 0; r < 2; ++r) {
          const size_t flat = cs + 2 * ca;
          const double expected = (r == 0 ? 100.0 + flat : 200.0 + flat) / (10.0 + flat);
          if (std::abs(X(cs, ca, r) - expected) > 1e-12)
            throw std::runtime_error("solve tests: multi-axis flattening is wrong");
        }
  }

  // Coefficient free dimensions broadcast across all RHS free dimensions.
  {
    T A({{"row", 2}, {"col", 2}, {"a", 2}});
    T B({{"row", 2}, {"b0", 2}, {"b1", 3}});
    for (size_t a = 0; a < 2; ++a) {
      A(0, 0, a) = 2.0 + a; A(1, 0, a) = 0.0;
      A(0, 1, a) = 0.0;     A(1, 1, a) = 4.0 + a;
    }
    for (size_t b1 = 0; b1 < 3; ++b1)
      for (size_t b0 = 0; b0 < 2; ++b0) {
        B(0, b0, b1) = 2.0 + b0 + 10.0 * b1;
        B(1, b0, b1) = 8.0 + b0 + 10.0 * b1;
      }
    T X;
    solve(A, B, X, {axis});
    if (X.rank() != 4 || X.dims()[0].label != TensorDimLabel("col")
        || X.dims()[1].label != TensorDimLabel("a")
        || X.dims()[2].label != TensorDimLabel("b0")
        || X.dims()[3].label != TensorDimLabel("b1"))
      throw std::runtime_error("solve tests: broadcast output shape is wrong");
    for (size_t b1 = 0; b1 < 3; ++b1)
      for (size_t b0 = 0; b0 < 2; ++b0)
        for (size_t a = 0; a < 2; ++a) {
          if (std::abs(X(0, a, b0, b1) - B(0, b0, b1) / (2.0 + a)) > 1e-12
              || std::abs(X(1, a, b0, b1) - B(1, b0, b1) / (4.0 + a)) > 1e-12)
            throw std::runtime_error("solve tests: free-dimension broadcast is wrong");
        }
  }

  // Non-LAPACK scalars exercise the generic column-pivoted QR fallback.
  {
    using MP = numerics::float50;
    Tensor<MP, Host> A({{"row", 2}, {"col", 2}});
    Tensor<MP, Host> B({{"row", 2}, {"rhs", 1}});
    A(0, 0) = MP(2); A(1, 0) = MP(1);
    A(0, 1) = MP(1); A(1, 1) = MP(3);
    B(0, 0) = MP(1); B(1, 0) = MP(2);
    Tensor<MP, Host> X;
    solve(A, B, X, {axis});
    const MP r0 = MP(2) * X(0, 0) + X(1, 0) - MP(1);
    const MP r1 = X(0, 0) + MP(3) * X(1, 0) - MP(2);
    if (abs(r0) > MP("1e-40") || abs(r1) > MP("1e-40"))
      throw std::runtime_error("solve tests: multiprecision QR residual is wrong");
  }
  {
    Tensor<float, Host> A({{"row", 2}, {"col", 2}}), B({{"row", 2}, {"rhs", 1}}), X;
    A(0, 0) = 2.0f; A(1, 0) = 1.0f; A(0, 1) = 1.0f; A(1, 1) = 3.0f;
    B(0, 0) = 1.0f; B(1, 0) = 2.0f;
    solve(A, B, X, {axis});
    if (std::abs(2.0f * X(0, 0) + X(1, 0) - 1.0f) > 1e-5f
        || std::abs(X(0, 0) + 3.0f * X(1, 0) - 2.0f) > 1e-5f)
      throw std::runtime_error("solve tests: float QR residual is wrong");
  }
  {
    using C = std::complex<long double>;
    Tensor<C, Host> A({{"row", 2}, {"col", 2}}), B({{"row", 2}, {"rhs", 1}}), X;
    A(0, 0) = C(1, 2); A(1, 0) = C(3, -1);
    A(0, 1) = C(2, -3); A(1, 1) = C(4, 1);
    B(0, 0) = C(5, 1); B(1, 0) = C(-2, 3);
    const auto A0 = A, B0 = B;
    solve(A, B, X, {axis});
    for (size_t i = 0; i < 2; ++i) {
      C value{};
      for (size_t k = 0; k < 2; ++k) value += A0(i, k) * X(k, 0);
      if (std::abs(value - B0(i, 0)) > 1e-15L)
        throw std::runtime_error("solve tests: complex long double QR residual is wrong");
    }
  }

  // Singular systems warn at verbosity level 1 and fail rather than returning
  // a fabricated solution.  Restore the process-wide warning setting afterward.
  {
    T A({{"row", 2}, {"col", 2}}), B({{"row", 2}, {"rhs", 1}}), X;
    A(0, 0) = 1.0; A(1, 0) = 2.0;
    A(0, 1) = 2.0; A(1, 1) = 4.0;
    B(0, 0) = 1.0; B(1, 0) = 2.0;
    const int old_verbosity = verbosity();
    set_verbosity(1);
    const bool threw = throws_as<std::runtime_error>([&] { solve(A, B, X, {axis}); });
    set_verbosity(old_verbosity);
    if (!threw)
      throw std::runtime_error("solve tests: singular LAPACK solve should throw");
  }

  // Boost complex multiprecision is also classified as FloatingPoint and uses
  // the ADL-based conjugate/magnitude QR path.
  {
    using R = boost::multiprecision::cpp_bin_float_50;
    using C = boost::multiprecision::cpp_complex_50;
    const auto real = [](int x) { return C(R(x), R(0)); };
    Tensor<C, Host> A({{"row", 2}, {"col", 2}});
    Tensor<C, Host> B({{"row", 2}, {"rhs", 1}}), X;
    A(0, 0) = real(2); A(1, 0) = real(1);
    A(0, 1) = real(1); A(1, 1) = real(3);
    B(0, 0) = real(1); B(1, 0) = real(2);
    solve(A, B, X, {axis});
    using boost::multiprecision::abs;
    if (abs(real(2) * X(0, 0) + X(1, 0) - real(1)) > R("1e-40")
        || abs(X(0, 0) + real(3) * X(1, 0) - real(2)) > R("1e-40"))
      throw std::runtime_error("solve tests: boost complex QR residual is wrong");
  }

  // QR-path singularity also warns and throws.
  {
    Tensor<float, Host> A({{"row", 2}, {"col", 2}}), B({{"row", 2}, {"rhs", 1}}), X;
    A(0, 0) = 1.0f; A(1, 0) = 2.0f;
    A(0, 1) = 2.0f; A(1, 1) = 4.0f;
    B(0, 0) = 1.0f; B(1, 0) = 2.0f;
    const int old_verbosity = verbosity();
    set_verbosity(1);
    const bool threw = throws_as<std::runtime_error>([&] { solve(A, B, X, {axis}); });
    set_verbosity(old_verbosity);
    if (!threw)
      throw std::runtime_error("solve tests: singular QR solve should throw");
  }

  // Boost complex backend conjugation uses the same ADL-compatible abstraction.
  {
    using C = boost::multiprecision::cpp_complex_50;
    Tensor<C, Host> Z({{"z", 1}});
    Z(0) = C(1, 2);
    TensorBackend<C, Host>::conjugate(Z.buffer());
  }

  // Axis positions and validation are contractual, like gemm's contracted-axis positions.
  {
    T A({{"col", 2}, {"row", 2}}), B({{"row", 2}, {"rhs", 1}}), X;
    if (!throws_ia([&] { solve(A, B, X, {axis}); }))
      throw std::runtime_error("solve tests: wrong coefficient axis ordering should throw");
    if (!throws_ia([&] { solve(A, B, X, {}); }))
      throw std::runtime_error("solve tests: empty axes should throw");
  }
  {
    T A({{"row", 2}}), B({{"row", 2}}), X;
    if (!throws_ia([&] { solve(A, B, X, {axis}); }))
      throw std::runtime_error("solve tests: insufficient coefficient rank should throw");
  }
  {
    T A({{"row", 2}, {"col", 3}}), B({{"row", 2}}), X;
    if (!throws_ia([&] { solve(A, B, X, {axis}); }))
      throw std::runtime_error("solve tests: unequal row/column extents should throw");
  }
  {
    T A({{"row", 2}, {"col", 2}}), B({{"row", 3}}), X;
    if (!throws_ia([&] { solve(A, B, X, {axis}); }))
      throw std::runtime_error("solve tests: unequal RHS row extent should throw");
  }
  {
    T Awrong({{"col", 2}, {"row", 2}}), B({{"row", 2}}), X;
    Awrong(0, 0) = 2.0; Awrong(1, 0) = 1.0;
    Awrong(0, 1) = 1.0; Awrong(1, 1) = 3.0;
    B(0) = 1.0; B(1) = 2.0;
    if (!throws_ia([&] { solve(Awrong, B, X, {axis}); }))
      throw std::runtime_error("solve tests: valid but unpermuted input should throw");
    auto A = Awrong.permute_axes({1, 0});
    solve(A, B, X, {axis});
    if (std::abs(2.0 * X(0) + X(1) - 1.0) > 1e-12
        || std::abs(X(0) + 3.0 * X(1) - 2.0) > 1e-12)
      throw std::runtime_error("solve tests: explicit permute before solve failed");
  }
  {
    T A({{"row", 2}, {"col", 2}}), B({{"row", 2}}), X({{"wrong", 2}});
    A(0, 0) = 2.0; A(1, 0) = 1.0; A(0, 1) = 1.0; A(1, 1) = 3.0;
    B(0) = 1.0; B(1) = 2.0;
    if (!throws_ia([&] { solve(A, B, X, {axis}); }))
      throw std::runtime_error("solve tests: incompatible preallocated output should throw");
  }
  {
    T A({{"row", 2}, {"dup", 2}}), B({{"row", 2}, {"dup", 1}}), X;
    if (!throws_ia([&] { solve(A, B, X, {SolveAxis{"row", "dup"}}); }))
      throw std::runtime_error("solve tests: duplicate output labels should throw");
  }
  {
    const SymGroup g = Symmetric();
    std::vector<TensorDim> adims{TensorDim{"ao", 2, g}, TensorDim{"ao", 2, g}};
    T A(adims), B({{"ao", 2}}), X;
    A(0, 0) = 2.0; A(1, 0) = 1.0; A(0, 1) = 1.0; A(1, 1) = 3.0;
    B(0) = 1.0; B(1) = 2.0;
    solve(A, B, X, {SolveAxis{"ao", "ao"}});
    if (std::abs(2.0 * X(0) + X(1) - 1.0) > 1e-12
        || std::abs(X(0) + 3.0 * X(1) - 2.0) > 1e-12)
      throw std::runtime_error("solve tests: duplicate-label positional solve failed");
  }

  return 0;
}

// ============================================================================
//  Story 05 -- InputCatalog input framework, parser, and semantic GF2 load
// ============================================================================

int run_input_data_loading_tests() {
  using Host = Executor::Host;

  // Helper: true iff f() throws std::invalid_argument whose message contains
  // every listed needle (and nothing else is checked).
  auto throws_ia_with = [](auto&& f, const std::vector<std::string>& needles) {
    try { f(); }
    catch (const std::invalid_argument& e) {
      const std::string msg = e.what();
      for (const std::string& n : needles)
        if (msg.find(n) == std::string::npos)
          return false;
      return true;
    }
    catch (...) { return false; }
    return false;
  };

  const auto text_with = [](const std::string& extra) {
    return std::string("CALCULATION = GF2\n")
         + "HCORE = paths/hcore\n"
         + "MO_COEFF = paths/mo_coeff\n"
         + "ERI3 = paths/eri3\n"
         + "OVERLAP = paths/overlap\n"
         + "DENSITY_MATRIX = paths/density_matrix\n"
         + "BETA = 10.0\n"
         + "MATSUBARA_HALF_N = 2\n"
         + "MU = 0.0\n"
         + extra;
  };

  // ------------------------------------------------------------------
  //  A. Parser / registry / sanitization (pure, no files)
  // ------------------------------------------------------------------
  // A1. Happy path: required values, comments, blank lines, case-insensitive
  //     keyword lookup, path value preserved byte-for-byte.
  {
    const std::string text =
      "# header comment\n"
      "// another comment\n\n"
      "cAlCuLaTiOn = GF2\n"
      "hcore  =  paths/hcore\n"
      "Mo_Coeff=paths/mo_coeff\n"
      "ERI3 = My/Path/eri3\n"
      "OVERLAP = paths/overlap\n"
      "DENSITY_MATRIX = paths/density_matrix\n"
      "BETA = 10.0\n"
      "MATSUBARA_HALF_N = 2\n"
      "MU = -0.25\n"
      "eta   =  1e-5\n";
    const ResolvedInput r = parse_input(text);
    if (r.calc != Calc::Gf2)
      throw std::runtime_error("input tests: happy-path calculation is not GF2");
    if (r.get_string("HCORE") != "paths/hcore")
      throw std::runtime_error("input tests: HCORE path value is wrong");
    if (r.get_string("MO_COEFF") != "paths/mo_coeff")
      throw std::runtime_error("input tests: MO_COEFF path value is wrong");
    if (r.get_string("ERI3") != "My/Path/eri3")
      throw std::runtime_error("input tests: ERI3 path case was not preserved exactly");
    if (r.get_double("ETA") != 1e-5)
      throw std::runtime_error("input tests: ETA value is wrong");
    if (!r.supplied("ETA"))
      throw std::runtime_error("input tests: user-supplied ETA is not reported as supplied");
    if (r.supplied("HCORE") != true)
      throw std::runtime_error("input tests: HCORE is not reported as supplied");
  }

  // A2. Keywords are case-insensitive; values are case-sensitive.
  {
    const std::string text =
      "CALCULATION = gf2\n"
      "HCORE = hcore\nMO_COEFF = mo\n"
      "ErI3 = My/Path/eri3\n"
      "OVERLAP = My/Path/Overlap\n"
      "DENSITY_MATRIX = My/Path/Density\n"
      "BETA = 10\n"
      "MATSUBARA_HALF_N = 2\n"
      "MU = 0\n";
    const ResolvedInput r = parse_input(text);
    if (r.get_string("ERI3") != "My/Path/eri3")
      throw std::runtime_error("input tests: keyword case-folding broke the ERI3 value");
  }
  {
    const std::string text =
      "calculation = Gf2\n"
      "HCORE = hcore\nMO_COEFF = mo\nERI3 = eri3\n"
      "OVERLAP = overlap\nDENSITY_MATRIX = density_matrix\nBETA = 10\nMATSUBARA_HALF_N = 2\nMU = 0\n";
    const ResolvedInput r = parse_input(text);
    if (r.calc != Calc::Gf2)
      throw std::runtime_error("input tests: calc case-folding is wrong");
  }

  // A3. Double forms: ordinary and Fortran-D exponents, full token only.
  {
    for (const std::string tok : {"1e-5", "1E-5", "1D-5", "1d-5", "0.00001", "1.0E-5", "+0.00001", "0.01e-3"}) {
      const ResolvedInput r = parse_input(
          "CALCULATION = GF2\nHCORE = h\nMO_COEFF = m\nERI3 = e\nOVERLAP = s\nDENSITY_MATRIX = p\nBETA = 10\nMATSUBARA_HALF_N = 2\nMU = 0\neta = " + tok + "\n");
      if (r.get_double("ETA") != 1e-5) {
        std::ostringstream oss;
        oss << "input tests: double token '" << tok << "' did not parse to 1e-5";
        throw std::runtime_error(oss.str());
      }
    }
  }

  // A4. ETA default + provenance.
  {
    const ResolvedInput defaulted = parse_input(text_with(""));
    if (defaulted.get_double("ETA") != 1e-5)
      throw std::runtime_error("input tests: ETA default is not 1e-5");
    if (defaulted.supplied("ETA"))
      throw std::runtime_error("input tests: defaulted ETA is reported as supplied");

    const ResolvedInput supplied = parse_input(text_with("eta = 2.5e-4\n"));
    if (supplied.get_double("ETA") != 2.5e-4 || !supplied.supplied("ETA"))
      throw std::runtime_error("input tests: user ETA value/provenance is wrong");
  }

  // A5. Unknown keywords are a hard error, naming the keyword.
  {
    if (!throws_ia_with([&] { parse_input(text_with("HERI3 = x\n")); }, {"HERI3"}))
      throw std::runtime_error("input tests: unknown keyword HERI3 should be rejected");
    if (!throws_ia_with(
        [&] { parse_input("CALCULATION = GF2\nHCORE = h\nMO_COEFF = m\nERI3 = e\nOVERLAP = s\nDENSITY_MATRIX = p\nBETA = 10\nMATSUBARA_HALF_N = 2\nMU = 0\nBOGUS = 1\n"); },
        {"BOGUS"}))
      throw std::runtime_error("input tests: unknown keyword BOGUS should be rejected");
  }

  // A6. CALCULATION selection rules.
  {
    // Missing CALCULATION -> hard error.
    if (!throws_ia_with(
        [&] { parse_input("HCORE = h\nMO_COEFF = m\nERI3 = e\nOVERLAP = s\nDENSITY_MATRIX = p\nBETA = 10\nMATSUBARA_HALF_N = 2\nMU = 0\n"); },
        {"CALCULATION"}))
      throw std::runtime_error("input tests: missing CALCULATION should be rejected");
    // Unknown calculation name -> hard error, naming the value.
    if (!throws_ia_with(
        [&] { parse_input("CALCULATION = GWR\n"); },
        {"GWR"}))
      throw std::runtime_error("input tests: unknown calculation name should be rejected");
  }

  // A7. Missing REQUIRED GF2 keywords are reported together.
  {
    if (!throws_ia_with(
        [&] { parse_input("CALCULATION = GF2\nMO_COEFF = mo\n"); },
        {"HCORE", "ERI3"}))
      throw std::runtime_error("input tests: missing required HCORE/ERI3 must be reported in one error");
  }

  // A8. Malformed lines.
  {
    if (!throws_ia_with(
        [&] { parse_input("CALCULATION = GF2\nHCORE = h\nMO_COEFF = m\nERI3 = e\nOVERLAP = s\nDENSITY_MATRIX = p\nBETA = 10\nMATSUBARA_HALF_N = 2\nMU = 0\njust some words\n"); },
        {}))
      throw std::runtime_error("input tests: a line without '=' should be rejected");
    if (!throws_ia_with(
        [&] { parse_input("CALCULATION = GF2\nHCORE = h\nMO_COEFF = m\nERI3 = e\nOVERLAP = s\nDENSITY_MATRIX = p\nBETA = 10\nMATSUBARA_HALF_N = 2\nMU = 0\n= value\n"); },
        {}))
      throw std::runtime_error("input tests: an empty keyword should be rejected");
  }

  // A9. Bad double values: partial tokens, non-numerics, non-finite.
  {
    for (const std::string bad : {"foo", "1e-5x", "5!", "inf", "infinity", "nan"}) {
      if (!throws_ia_with(
          [&] { parse_input("CALCULATION = GF2\nHCORE = h\nMO_COEFF = m\nERI3 = e\nOVERLAP = s\nDENSITY_MATRIX = p\nBETA = 10\nMATSUBARA_HALF_N = 2\nMU = 0\neta = " + bad + "\n"); },
          {"ETA"})) {
        std::ostringstream oss;
        oss << "input tests: bad double token '" << bad << "' should be rejected";
        throw std::runtime_error(oss.str());
      }
    }
  }

  // A10. Sanitization: ETA must be strictly positive.
  {
    if (!throws_ia_with(
        [&] { parse_input(text_with("eta = -1e-5\n")); },
        {"ETA"}))
      throw std::runtime_error("input tests: negative ETA should be rejected");
    if (!throws_ia_with(
        [&] { parse_input(text_with("eta = 0\n")); },
        {"ETA"}))
      throw std::runtime_error("input tests: zero ETA should be rejected");
  }

  // A10b. Story 07.1 scalar parsing/ranges: strict positive integer half-grid,
  // positive finite beta, and finite (but not necessarily positive) mu.
  {
    const auto scalar_text = [](const std::string& half, const std::string& beta, const std::string& mu) {
      return std::string("CALCULATION = GF2\nHCORE = h\nMO_COEFF = m\nERI3 = e\n")
           + "OVERLAP = s\nDENSITY_MATRIX = p\nBETA = " + beta
           + "\nMATSUBARA_HALF_N = " + half + "\nMU = " + mu + "\n";
    };
    const ResolvedInput r = parse_input(scalar_text("2", "10.0", "-1.25"));
    if (r.get_double("BETA") != 10.0 || r.get_int("MATSUBARA_HALF_N") != 2 || r.get_double("MU") != -1.25)
      throw std::runtime_error("input tests: BETA/MATSUBARA_HALF_N/MU typed accessors are wrong");
    for (const std::string bad : {"4.0", "1e3", "x", "4x", "-4", "+4"}) {
      if (!throws_ia_with([&] { parse_input(scalar_text(bad, "10", "0")); }, {"MATSUBARA_HALF_N"}))
        throw std::runtime_error("input tests: malformed MATSUBARA_HALF_N token should be rejected");
    }
    if (!throws_ia_with([&] { parse_input(scalar_text("0", "10", "0")); }, {"MATSUBARA_HALF_N"}))
      throw std::runtime_error("input tests: zero MATSUBARA_HALF_N should be rejected");
    if (!throws_ia_with([&] { parse_input(scalar_text("2", "0", "0")); }, {"BETA"}))
      throw std::runtime_error("input tests: zero BETA should be rejected");
    if (!throws_ia_with([&] { parse_input(scalar_text("2", "-1", "0")); }, {"BETA"}))
      throw std::runtime_error("input tests: negative BETA should be rejected");
    if (!throws_ia_with([&] { parse_input(scalar_text("2", "10", "nan")); }, {"MU"}))
      throw std::runtime_error("input tests: nonfinite MU should be rejected");
  }

  // A11. Duplicate keyword -> hard error naming both lines' keyword.
  {
    if (!throws_ia_with(
        [&] { parse_input(text_with("eta = 1e-5\neta = 2e-5\n")); },
        {"ETA"}))
      throw std::runtime_error("input tests: duplicate ETA (last-wins) should be rejected");
  }
  {
    // Duplicated dataset paths are equally ambiguous.
    if (!throws_ia_with(
        [&] { parse_input(text_with("HCORE = h2\n")); },
        {"HCORE"}))
      throw std::runtime_error("input tests: duplicate HCORE (last-wins) should be rejected");
  }

  // ------------------------------------------------------------------
  //  B. GF2 semantic data loading (self-contained HDF5 via HighFive)
  // ------------------------------------------------------------------
  {
    // A small, deterministic, nontrivial fixture.
    const std::string h5 = (std::filesystem::temp_directory_path() / "cppgw_story05_test.h5").string();
    std::remove(h5.c_str());

    const size_t N_AO = 24, N_MO = 12, N_RI = 30;
    {
      HighFive::File f(h5, HighFive::File::Truncate);
      auto fill = [](std::vector<double>& v, double base) {
        for (size_t i = 0; i < v.size(); ++i) v[i] = base + (double)i;
      };
      std::vector<double> hcore(N_AO * N_AO);    fill(hcore, 1.0);
      std::vector<double> mo    (N_AO * N_MO);   fill(mo,    100.0);
      std::vector<double> eri3  (N_RI * N_AO * N_AO); fill(eri3, 1000.0);
      std::vector<double> overlap(N_AO * N_AO);  fill(overlap, 3000.0);
      std::vector<double> density(N_AO * N_AO);  fill(density, 4000.0);
      std::vector<double> wide  (N_AO * 2);      fill(wide,  10000.0);   // deliberately non-square
      std::vector<double> aomis (2 * 2);         fill(aomis, 20000.0);   // AO extent != N_AO
      f.createDataSet<double>("hcore",    HighFive::DataSpace(std::vector<size_t>{N_AO, N_AO})).write_raw(hcore.data());
      f.createDataSet<double>("mo_coeff", HighFive::DataSpace(std::vector<size_t>{N_AO, N_MO})).write_raw(mo.data());
      f.createDataSet<double>("eri3",     HighFive::DataSpace(std::vector<size_t>{N_RI, N_AO, N_AO})).write_raw(eri3.data());
      f.createDataSet<double>("overlap",  HighFive::DataSpace(std::vector<size_t>{N_AO, N_AO})).write_raw(overlap.data());
      f.createDataSet<double>("density_matrix", HighFive::DataSpace(std::vector<size_t>{N_AO, N_AO})).write_raw(density.data());
      f.createDataSet<double>("wide",     HighFive::DataSpace(std::vector<size_t>{N_AO, 2})).write_raw(wide.data());
      f.createDataSet<double>("ao_mis",   HighFive::DataSpace(std::vector<size_t>{2, 2})).write_raw(aomis.data());
    }

    // B1. Happy load: dims, labels, declared symmetries, and an exact copy.
    {
      const InputCatalog cat = InputCatalog::from_text(
          "CALCULATION = GF2\nHCORE = hcore\nMO_COEFF = mo_coeff\nERI3 = eri3\nOVERLAP = overlap\nDENSITY_MATRIX = density_matrix\nBETA = 10\nMATSUBARA_HALF_N = 2\nMU = 0\neta = 1e-5\n",
          h5);
      Gf2Input g = cat.require_gf2();

      // hcore: 2x2, both axes 'ao', ONE shared Hermitian SymGroup.
      if (g.hcore.rank() != 2)
        throw std::runtime_error("input tests: hcore has wrong rank");
      if (g.hcore.dims()[0].label != TensorDimLabel("ao")
          || g.hcore.dims()[1].label != TensorDimLabel("ao"))
        throw std::runtime_error("input tests: hcore axes are not labelled 'ao'");
      if (g.hcore.dims()[0].dim != N_AO || g.hcore.dims()[1].dim != N_AO)
        throw std::runtime_error("input tests: hcore has wrong dimensions");
      if (!g.hcore.dims()[0].symmetry || g.hcore.dims()[0].symmetry != g.hcore.dims()[1].symmetry)
        throw std::runtime_error("input tests: hcore AO axes are not bound to ONE shared SymGroup");
      if (g.hcore.dims()[0].symmetry->kind() != SymKind::Hermitian)
        throw std::runtime_error("input tests: hcore AO pair should be a Hermitian family");

      // mo_coeff: 2 dims { mo (fastest), ao }, plain axes, no SymGroups
      // (H5 shape (ao, mo): mo is the inner/fast axis).
      if (g.mo_coeff.rank() != 2)
        throw std::runtime_error("input tests: mo_coeff has wrong rank");
      if (g.mo_coeff.dims()[0].label != TensorDimLabel("mo")
          || g.mo_coeff.dims()[1].label != TensorDimLabel("ao"))
        throw std::runtime_error("input tests: mo_coeff axes are not labelled mo(fast)/ao");
      if (g.mo_coeff.dims()[0].dim != N_MO || g.mo_coeff.dims()[1].dim != N_AO)
        throw std::runtime_error("input tests: mo_coeff has wrong dimensions");
      if (g.mo_coeff.dims()[0].symmetry || g.mo_coeff.dims()[1].symmetry)
        throw std::runtime_error("input tests: mo_coeff must be plain (no SymGroups)");

      // eri3: 3x3, both AO axes 'ao' under ONE shared Symmetric SymGroup; 'ri' plain.
      if (g.eri3.rank() != 3)
        throw std::runtime_error("input tests: eri3 has wrong rank");
      if (g.eri3.dims()[0].label != TensorDimLabel("ao")
          || g.eri3.dims()[1].label != TensorDimLabel("ao")
          || g.eri3.dims()[2].label != TensorDimLabel("ri"))
        throw std::runtime_error("input tests: eri3 axes are not labelled ao/ao/ri");
      if (g.eri3.dims()[0].dim != N_AO || g.eri3.dims()[1].dim != N_AO || g.eri3.dims()[2].dim != N_RI)
        throw std::runtime_error("input tests: eri3 has wrong dimensions");
      if (!g.eri3.dims()[0].symmetry || g.eri3.dims()[0].symmetry != g.eri3.dims()[1].symmetry)
        throw std::runtime_error("input tests: eri3 AO axes are not bound to ONE shared SymGroup");
      if (g.eri3.dims()[0].symmetry->kind() != SymKind::Symmetric)
        throw std::runtime_error("input tests: eri3 AO pair should be a Symmetric family");
      if (g.eri3.dims()[2].symmetry)
        throw std::runtime_error("input tests: eri3 ri axis must be plain");

      // Symmetry is advisory: transposing a bound pair on a real scalar is a
      // no-op (Story 06), so the SAME coordinate of the result must match.
      {
        if (g.hcore.transpose(0, 1)(2, 1) != g.hcore(2, 1))
          throw std::runtime_error("input tests: Hermitian (real) transpose must be a no-op");
        if (g.eri3.transpose(0, 1)(2, 1, 3) != g.eri3(2, 1, 3))
          throw std::runtime_error("input tests: Symmetric (real) transpose must be a no-op");
      }

      // Data was copied exactly from the file (no reinterpretation).
      {
        const double* h = g.hcore.data();
        const double* m = g.mo_coeff.data();
        const double* e = g.eri3.data();
        for (size_t j = 0; j < N_AO; ++j)
          for (size_t i = 0; i < N_AO; ++i)
            if (h[i + j * N_AO] != 1.0 + i + j * N_AO)
              throw std::runtime_error("input tests: hcore element mismatch");
        for (size_t a = 0; a < N_AO; ++a)
          for (size_t mm = 0; mm < N_MO; ++mm)
            if (m[mm + a * N_MO] != 100.0 + a * N_MO + mm)
              throw std::runtime_error("input tests: mo_coeff element mismatch");
        for (size_t k = 0; k < N_RI; ++k)
          for (size_t j = 0; j < N_AO; ++j)
            for (size_t i = 0; i < N_AO; ++i)
              if (e[i + j * N_AO + k * N_AO * N_AO] != 1000.0 + i + j * N_AO + k * N_AO * N_AO)
                throw std::runtime_error("input tests: eri3 element mismatch");
      }

      // eta round-tripped and marked supplied.
      if (g.eta != 1e-5 || !g.eta_was_supplied)
        throw std::runtime_error("input tests: supplied eta did not round-trip exactly");
    }

    // B2. Missing dataset -> clear, keyword-bearing error (not success).
    // HDF5's C-level error handler prints a HDF5-DIAG block to stderr itself
    // (before HighFive raises the C++ exception we assert on). That output
    // looks like an unintended failure, so silence the C handler for exactly
    // this deliberate negative test; the RAII guard restores the default
    // handler even if the check below throws.
    struct Hdf5ErrorQuiet {
      H5E_auto2_t saved_api_func = nullptr;   void* saved_api_data = nullptr;
      Hdf5ErrorQuiet() {
        H5Eget_auto2(H5E_DEFAULT, &saved_api_func, &saved_api_data);
        // Silence the default API error callback (the one that prints the
        // HDF5-DIAG stack to stderr) for the duration of the guard.
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
      }
      ~Hdf5ErrorQuiet() {
        H5Eset_auto2(H5E_DEFAULT, saved_api_func, saved_api_data);
      }
      Hdf5ErrorQuiet(const Hdf5ErrorQuiet&) = delete;
      Hdf5ErrorQuiet& operator=(const Hdf5ErrorQuiet&) = delete;
    } quiet;
    if (!throws_ia_with(
        [&] {
          const InputCatalog cat = InputCatalog::from_text(
              "CALCULATION = GF2\nHCORE = hcore\nMO_COEFF = mo_coeff\nERI3 = nope_here\nOVERLAP = overlap\nDENSITY_MATRIX = density_matrix\nBETA = 10\nMATSUBARA_HALF_N = 2\nMU = 0\n", h5);
          (void)cat.require_gf2();
        }, {"ERI3", "nope_here"}))
      throw std::runtime_error("input tests: a missing ERI3 dataset should fail with its name");

    // B3. Rank mismatch on a required dataset is a hard error.
    {
      if (!throws_ia_with(
          [&] {
            const InputCatalog cat = InputCatalog::from_text(
                "CALCULATION = GF2\nHCORE = eri3\nMO_COEFF = mo_coeff\nERI3 = eri3\nOVERLAP = overlap\nDENSITY_MATRIX = density_matrix\nBETA = 10\nMATSUBARA_HALF_N = 2\nMU = 0\n", h5);
            (void)cat.require_gf2();
          }, {"HCORE", "rank 2"}))
        throw std::runtime_error("input tests: a rank-3 dataset used as HCORE should be rejected for rank");
    }

    // B3b. HCORE must be a SQUARE AO x AO matrix.
    {
      if (!throws_ia_with(
          [&] {
            const InputCatalog cat = InputCatalog::from_text(
                "CALCULATION = GF2\nHCORE = wide\nMO_COEFF = mo_coeff\nERI3 = eri3\nOVERLAP = overlap\nDENSITY_MATRIX = density_matrix\nBETA = 10\nMATSUBARA_HALF_N = 2\nMU = 0\n", h5);
            (void)cat.require_gf2();
          }, {"HCORE", "square"}))
        throw std::runtime_error("input tests: a non-square dataset used as HCORE should be rejected");
    }

    // B3c. MO_COEFF's AO extent must match HCORE's.
    {
      if (!throws_ia_with(
          [&] {
            const InputCatalog cat = InputCatalog::from_text(
                "CALCULATION = GF2\nHCORE = hcore\nMO_COEFF = ao_mis\nERI3 = eri3\nOVERLAP = overlap\nDENSITY_MATRIX = density_matrix\nBETA = 10\nMATSUBARA_HALF_N = 2\nMU = 0\n", h5);
            (void)cat.require_gf2();
          }, {"MO_COEFF", "AO"}))
        throw std::runtime_error("input tests: a MO_COEFF AO extent mismatch should be rejected");
    }

    // B4. Element type must be float64; an integer dataset is rejected.
    {
      {
        HighFive::File f(h5, HighFive::File::AccessMode::ReadWrite | HighFive::File::AccessMode::Create);
        std::vector<int> bad(N_AO * N_AO);
        for (size_t i = 0; i < bad.size(); ++i) bad[i] = (int)i;
        f.createDataSet<int>("hcore_int", HighFive::DataSpace(std::vector<size_t>{N_AO, N_AO})).write_raw(bad.data());
      }
      if (!throws_ia_with(
          [&] {
            const InputCatalog cat = InputCatalog::from_text(
                "CALCULATION = GF2\nHCORE = hcore_int\nMO_COEFF = mo_coeff\nERI3 = eri3\nOVERLAP = overlap\nDENSITY_MATRIX = density_matrix\nBETA = 10\nMATSUBARA_HALF_N = 2\nMU = 0\n", h5);
            (void)cat.require_gf2();
          }, {"HCORE", "float64"}))
        throw std::runtime_error("input tests: an integer HCORE dataset must be rejected as non-float64");
    }

    // The happy-path fixture (B1) is left in place for a manual CLI smoke
    // check (`./cppgw -i gf2.in -d <fixture>`); do not fail the suite over cleanup.
    (void)h5;
  }

  // ------------------------------------------------------------------
  //  C. Story 07.1 numerical GF2 initial-guess checks
  // ------------------------------------------------------------------
  {
    auto make_manual_input = [](size_t n_ao, size_t n_ri) {
      Gf2Input in;
      const SymGroup hm = Hermitian();
      const SymGroup sm = Symmetric();
      in.hcore = Tensor<double, Host>({TensorDim{"ao", n_ao, hm}, TensorDim{"ao", n_ao, hm}});
      const SymGroup og = Hermitian();
      const SymGroup pg = Hermitian();
      in.overlap = Tensor<double, Host>({TensorDim{"ao", n_ao, og}, TensorDim{"ao", n_ao, og}});
      in.density_matrix = Tensor<double, Host>({TensorDim{"ao", n_ao, pg}, TensorDim{"ao", n_ao, pg}});
      in.mo_coeff = Tensor<double, Host>({TensorDim{"mo", n_ao}, TensorDim{"ao", n_ao}});
      in.eri3 = Tensor<double, Host>({TensorDim{"ao", n_ao, sm}, TensorDim{"ao", n_ao, sm}, TensorDim{"ri", n_ri}});
      in.beta = 5.0;
      in.matsubara_half_n = 2;
      in.mu = -0.2;
      return in;
    };

    // One-AO analytic Green's-function check over both negative and positive Matsubara points.
    {
      Gf2Input in = make_manual_input(1, 1);
      in.hcore(0, 0) = 0.7;
      in.overlap(0, 0) = 1.3;
      in.density_matrix(0, 0) = 0.4;
      in.eri3(0, 0, 0) = 0.9;
      Gf2InitialGuess r = make_gf2_initial_guess(in);
      const double sigma_ref = 0.9 * (0.9 * 0.4) - 0.5 * (0.9 * 0.4 * 0.9);
      if (std::abs(r.sigma_hf(0, 0) - sigma_ref) > 1e-13)
        throw std::runtime_error("GF2 initial-guess tests: one-AO sigma_hf is wrong");
      if (std::abs(r.electron_count - 0.4 * 1.3) > 1e-13)
        throw std::runtime_error("GF2 initial-guess tests: one-AO electron count is wrong");
      if (r.green.size() != 2 * in.matsubara_half_n || r.green.data().dims()[0].label != TensorDimLabel(GridExpansionMatsubara<cplx, Fermionic>::dim_label))
        throw std::runtime_error("GF2 initial-guess tests: Matsubara representation metadata is wrong");
      for (size_t n = 0; n < r.green.size(); ++n) {
        const cplx denom = cplx(in.mu, r.green.grid()(n).value) * in.overlap(0, 0) - (in.hcore(0, 0) + sigma_ref);
        const cplx ref = cplx(1.0, 0.0) / denom;
        if (std::abs(r.green.data()(n, 0, 0) - ref) > 1e-12)
          throw std::runtime_error("GF2 initial-guess tests: one-AO Green's function is wrong");
      }
    }

    // Two-AO static self-energy, electron count, residual A_n G_n = I, and ETA non-use.
    {
      Gf2Input in = make_manual_input(2, 2);
      in.beta = 7.0;
      in.matsubara_half_n = 2;
      in.mu = 0.35;
      in.eta = 1e-5;
      in.hcore(0,0)=0.6; in.hcore(1,0)=0.1; in.hcore(0,1)=0.1; in.hcore(1,1)=0.9;
      in.overlap(0,0)=1.1; in.overlap(1,0)=0.2; in.overlap(0,1)=0.2; in.overlap(1,1)=1.4;
      in.density_matrix(0,0)=0.8; in.density_matrix(1,0)=0.15; in.density_matrix(0,1)=0.12; in.density_matrix(1,1)=0.5;
      in.eri3(0,0,0)=0.7; in.eri3(1,0,0)=0.2; in.eri3(0,1,0)=0.3; in.eri3(1,1,0)=0.5;
      in.eri3(0,0,1)=0.4; in.eri3(1,0,1)=0.6; in.eri3(0,1,1)=0.1; in.eri3(1,1,1)=0.8;

      Gf2InitialGuess r = make_gf2_initial_guess(in);
      double rho[2]{};
      for (size_t Q = 0; Q < 2; ++Q)
        for (size_t k = 0; k < 2; ++k)
          for (size_t l = 0; l < 2; ++l)
            rho[Q] += in.eri3(k,l,Q) * in.density_matrix(k,l);
      for (size_t u = 0; u < 2; ++u)
        for (size_t v = 0; v < 2; ++v) {
          double J = 0.0, K = 0.0;
          for (size_t Q = 0; Q < 2; ++Q) {
            J += in.eri3(u,v,Q) * rho[Q];
            for (size_t k = 0; k < 2; ++k)
              for (size_t l = 0; l < 2; ++l)
                K += in.eri3(u,l,Q) * in.density_matrix(k,l) * in.eri3(k,v,Q);
          }
          const double sigma_ref = J - 0.5 * K;
          if (std::abs(r.sigma_hf(u,v) - sigma_ref) > 1e-13)
            throw std::runtime_error("GF2 initial-guess tests: two-AO sigma_hf is wrong");
          if (std::abs(r.fock(u,v) - (in.hcore(u,v) + sigma_ref)) > 1e-13)
            throw std::runtime_error("GF2 initial-guess tests: fock != hcore + sigma_hf");
        }
      double ne = 0.0;
      for (size_t u = 0; u < 2; ++u)
        for (size_t v = 0; v < 2; ++v)
          ne += in.density_matrix(u,v) * in.overlap(v,u);
      if (std::abs(r.electron_count - ne) > 1e-13)
        throw std::runtime_error("GF2 initial-guess tests: two-AO electron count is wrong");

      for (size_t n = 0; n < r.green.size(); ++n) {
        for (size_t u = 0; u < 2; ++u) {
          for (size_t v = 0; v < 2; ++v) {
            cplx lhs{};
            for (size_t k = 0; k < 2; ++k) {
              const cplx A = cplx(in.mu, r.green.grid()(n).value) * in.overlap(u,k) - r.fock(u,k);
              lhs += A * r.green.data()(n,k,v);
            }
            const cplx ref = (u == v) ? cplx(1.0, 0.0) : cplx(0.0, 0.0);
            if (std::abs(lhs - ref) > 1e-11)
              throw std::runtime_error("GF2 initial-guess tests: multi-AO solve residual is too large");
          }
        }
      }

      Gf2Input eta_changed = in;
      eta_changed.eta = 9.9;
      Gf2InitialGuess r_eta = make_gf2_initial_guess(eta_changed);
      for (size_t i = 0; i < r.green.data().total_elements(); ++i)
        if (std::abs(r.green.data().linear(i) - r_eta.green.data().linear(i)) > 0.0)
          throw std::runtime_error("GF2 initial-guess tests: eta changed the Matsubara Green's function");
    }

    // Singular coefficient matrices must identify the Matsubara point rather than returning partial data.
    {
      Gf2Input in = make_manual_input(1, 1);
      in.hcore(0,0) = 0.0;
      in.overlap(0,0) = 0.0;
      in.density_matrix(0,0) = 0.0;
      in.eri3(0,0,0) = 0.0;
      bool saw_frequency_context = false;
      try {
        (void)make_gf2_initial_guess(in);
      } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        saw_frequency_context = msg.find("Matsubara index") != std::string::npos;
      }
      if (!saw_frequency_context)
        throw std::runtime_error("GF2 initial-guess tests: singular solve did not report Matsubara context");
    }
  }

  // ------------------------------------------------------------------
  //  D. CLI smoke check (manual acceptance, Story-2 style); the in-process
  //     suite already covers the pipeline via A/B/C above.
  // ------------------------------------------------------------------

  return 0;
}

} // namespace cppgw
