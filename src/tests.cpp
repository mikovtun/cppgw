#include "main.hpp"
#include "tests.hpp"

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

} // namespace cppgw
