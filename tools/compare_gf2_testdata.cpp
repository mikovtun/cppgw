// ============================================================================
//  Story 07.1 reference-data comparison (one-shot tool, not part of the app)
// ----------------------------------------------------------------------------
//  Compares the cppgw output file (g0, sigma_hf, attributes) against the
//  reference file produced by tools/gen_gf2_testdata.py (g0, sigma_static,
//  omega_n, attrs beta/mu).
//
//  Usage:
//    compare_gf2_testdata REFERENCE_FILE OUTPUT_FILE [TOL]
//
//  Conventions compared:
//   * sigma_hf  == vj - 0.5*vk            (static restricted HF self-energy)
//   * g0[n,u,v] == inverse((i*omega_n + mu) S - F)
//     - reference omega_n are the first n_ref POSITIVE fermionic frequencies
//     - the cppgw grid is the full ascending (negative ... positive) grid, so
//       the positive half (indices halfN .. end) must match the reference
//   * output attributes eta/beta/matsubara_half_n/mu are reported
// ============================================================================

#include <highfive/highfive.hpp>

#include <complex>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using C = std::complex<double>;

namespace {

template <class T>
T read_scalar(const HighFive::File& f, const std::string& name) {
  if (!f.hasAttribute(name))
    throw std::runtime_error("attribute '" + name + "' is missing");
  return f.getAttribute(name).template read<T>();
}

void report_max_error(const char* what, double max_abs, double max_rel, double tol_rel) {
  const bool ok = (max_rel <= tol_rel);
  std::cout << "  " << what << ": max|err|=" << max_abs
            << "  max|err|/max(1,|ref|)=" << max_rel
            << (ok ? "  OK" : "  ** MISMATCH **") << "\n";
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " REFERENCE_FILE OUTPUT_FILE [TOL_REL]\n";
    return 2;
  }
  const double tol = (argc > 3) ? std::stod(argv[3]) : 1e-8;

  HighFive::File ref_file(argv[1], HighFive::File::ReadOnly);
  HighFive::File out_file(argv[2], HighFive::File::ReadOnly);

  // ------------------------------------------------------------------
  //  Scalar parameters (output attributes)
  // ------------------------------------------------------------------
  std::cout << "=== input parameters (output file attributes) ===\n";
  const double eta = read_scalar<double>(out_file, "eta");
  const int eta_supplied = read_scalar<int>(out_file, "eta_was_supplied");
  const double beta = read_scalar<double>(out_file, "beta");
  const long long half_n = read_scalar<long long>(out_file, "matsubara_half_n");
  const long long n_matsu = read_scalar<long long>(out_file, "n_matsubara");
  const double mu = read_scalar<double>(out_file, "mu");
  std::cout << "  eta=" << eta << (eta_supplied ? " (supplied)" : " (default)")
            << "  beta=" << beta << "  matsubara_half_n=" << half_n
            << "  n_matsubara=" << n_matsu << "  mu=" << mu << "\n";
  if (n_matsu != 2 * half_n)
    throw std::runtime_error("n_matsubara != 2 * matsubara_half_n");

  const double ref_beta = read_scalar<double>(ref_file, "beta");
  const double ref_mu = read_scalar<double>(ref_file, "mu");
  if (beta != ref_beta)
    throw std::runtime_error("beta (" + std::to_string(beta)
        + ") != reference beta (" + std::to_string(ref_beta) + ")");
  if (mu != ref_mu)
    throw std::runtime_error("mu (" + std::to_string(mu)
        + ") != reference mu (" + std::to_string(ref_mu) + ")");
  std::cout << "  beta/mu agree with the reference file\n";

  // ------------------------------------------------------------------
  //  Shared shapes
  // ------------------------------------------------------------------
  const std::vector<size_t> sshape =
      ref_file.getDataSet("sigma_static").getDimensions();
  if (sshape.size() != 2 || sshape[0] != sshape[1])
    throw std::runtime_error("reference sigma_static is not square");
  const size_t nao = sshape[0];

  const std::vector<size_t> gshape_ref = ref_file.getDataSet("g0").getDimensions();
  if (gshape_ref.size() != 3 || gshape_ref[1] != nao || gshape_ref[2] != nao)
    throw std::runtime_error("reference g0 has the wrong shape "
        + std::to_string(gshape_ref[0]) + "x" + std::to_string(gshape_ref[1]) + "x" + std::to_string(gshape_ref[2]));
  const size_t n_ref = gshape_ref[0];

  const std::vector<size_t> sshape_out = out_file.getDataSet("sigma_hf").getDimensions();
  if (sshape_out.size() != 2 || sshape_out[0] != nao || sshape_out[1] != nao)
    throw std::runtime_error("output sigma_hf has the wrong shape / does not match reference");
  const std::vector<size_t> gshape_out = out_file.getDataSet("g0").getDimensions();
  if (gshape_out.size() != 3 || gshape_out[1] != nao || gshape_out[2] != nao)
    throw std::runtime_error("output g0 has the wrong shape");
  const size_t n_out = gshape_out[0];
  if (gshape_out[0] != (size_t)n_matsu || gshape_out[0] != (size_t)half_n * 2)
    throw std::runtime_error("output g0 length does not match its attributes");

  // ------------------------------------------------------------------
  //  Static self-energy: sigma_hf (nao,nao) vs sigma_static (nao,nao)
  // ------------------------------------------------------------------
  {
    std::cout << "=== sigma_hf: " << nao << "x" << nao
              << " vs reference sigma_static ===\n";
    std::vector<double> sref(nao * nao), sout(nao * nao);
    ref_file.getDataSet("sigma_static").read_raw(sref.data());
    out_file.getDataSet("sigma_hf").read_raw(sout.data());
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t v = 0; v < nao; ++v)
      for (size_t u = 0; u < nao; ++u) {
        const double r = sref[u * nao + v], s = sout[u * nao + v];
        max_abs = std::max(max_abs, std::abs(r - s));
        max_rel = std::max(max_rel, std::abs(r - s) / std::max(1.0, std::abs(r)));
      }
    report_max_error("sigma_hf", max_abs, max_rel, tol);
    if (max_rel > tol)
      return 1;
  }

  // ------------------------------------------------------------------
  //  Matsubara grid check
  // ------------------------------------------------------------------
  // Reference omega_n (fermionic, positive) vs the positive half of the
  // cppgw ascending grid, reconstructed from beta.
  {
    std::cout << "=== Matsubara frequencies ===\n";
    std::vector<double> wref(n_ref);
    ref_file.getDataSet("omega_n").read_raw(wref.data());
    const size_t half = n_out / 2;
    if (n_ref > half)
      throw std::runtime_error("reference has more positive frequencies ("
          + std::to_string(n_ref) + ") than the cppgw grid (" + std::to_string(half)
          + "); cannot compare");
    const double fac = M_PI / beta;
    double max_err = 0.0;
    for (size_t k = 0; k < n_ref; ++k) {
      // cppgw frequency at index half+k (ascending, i.e. k-th positive point)
      const double wc = fac * (2.0 * (double)k + 1.0);
      max_err = std::max(max_err, std::abs(wref[k] - wc));
    }
    std::cout << "  n_ref=" << n_ref << " positive reference points; cppgw grid has "
              << n_out << " points; max |omega| mismatch = " << max_err
              << (max_err < 1e-12 ? "  OK" : "  ** MISMATCH **") << "\n";
    if (max_err >= 1e-12)
      return 1;
  }

  // ------------------------------------------------------------------
  //  Green's function: g0 positive half (cppgw) vs reference g0
  // ------------------------------------------------------------------
  {
    std::cout << "=== g0: cppgw indices " << (n_out / 2) << ".." << (n_out - 1)
              << " vs reference indices 0.." << (n_ref - 1) << " ===\n";
    std::vector<C> gref(n_ref * nao * nao), gout(n_out * nao * nao);
    ref_file.getDataSet("g0").read_raw(gref.data());
    out_file.getDataSet("g0").read_raw(gout.data());
    auto gout_at = [&](size_t n, size_t u, size_t v) -> C { return gout[n * nao * nao + u * nao + v]; };
    auto gref_at = [&](size_t n, size_t u, size_t v) -> C { return gref[n * nao * nao + u * nao + v]; };
    const size_t half = n_out / 2;
    double max_abs = 0.0, max_rel = 0.0;
    size_t worst_u = 0, worst_v = 0, worst_n = 0;
    for (size_t k = 0; k < n_ref; ++k) {
      for (size_t v = 0; v < nao; ++v)
        for (size_t u = 0; u < nao; ++u) {
          const C r = gref_at(k, u, v), s = gout_at(half + k, u, v);
          const double d = std::abs(r - s);
          max_abs = std::max(max_abs, d);
          max_rel = std::max(max_rel, d / std::max(1.0, std::abs(r)));
          if (d >= max_abs) { worst_u = u; worst_v = v; worst_n = k; }
        }
    }
    std::cout << "  worst element: n_ref=" << worst_n << " u=" << worst_u << " v=" << worst_v << "\n";
    report_max_error("g0 (positive half)", max_abs, max_rel, tol);
    if (max_rel > tol)
      return 1;
  }

  std::cout << "\n=== Story 07.1 comparison: ALL MATCH ===\n";
  return 0;
}
