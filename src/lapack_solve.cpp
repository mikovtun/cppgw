#include "backend.hpp"

#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cppgw::detail {

namespace {

extern "C" {
  void dgesv_(const int* n, const int* nrhs, double* a, const int* lda,
              int* ipiv, double* b, const int* ldb, int* info);
  void zgesv_(const int* n, const int* nrhs, std::complex<double>* a, const int* lda,
              int* ipiv, std::complex<double>* b, const int* ldb, int* info);
}

int checked_lapack_int(size_t value, const char* what) {
  if (value > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument(std::string("TensorBackend::solve_in_place: ")
        + what + " exceeds the LAPACK integer range");
  return static_cast<int>(value);
}

void handle_gesv_info(int info) {
  if (info < 0)
    throw std::invalid_argument("TensorBackend::solve_in_place: LAPACK gesv illegal argument "
        + std::to_string(-info));
  if (info > 0) {
    const std::string msg = "TensorBackend::solve_in_place: LAPACK gesv singular pivot "
        + std::to_string(info);
    warn_if(1, msg);
    throw std::runtime_error(msg);
  }
}

} // namespace

void lapack_gesv_in_place(TensorBuffer<double, Executor::Host>& coefficient,
                          size_t n,
                          TensorBuffer<double, Executor::Host>& rhs,
                          size_t nrhs) {
  const int ni = checked_lapack_int(n, "matrix dimension");
  const int ri = checked_lapack_int(nrhs, "RHS count");
  std::vector<int> pivots(n);
  int info = 0;
  dgesv_(&ni, &ri, coefficient.data(), &ni, pivots.data(), rhs.data(), &ni, &info);
  handle_gesv_info(info);
}

void lapack_gesv_in_place(TensorBuffer<std::complex<double>, Executor::Host>& coefficient,
                          size_t n,
                          TensorBuffer<std::complex<double>, Executor::Host>& rhs,
                          size_t nrhs) {
  const int ni = checked_lapack_int(n, "matrix dimension");
  const int ri = checked_lapack_int(nrhs, "RHS count");
  std::vector<int> pivots(n);
  int info = 0;
  zgesv_(&ni, &ri, coefficient.data(), &ni, pivots.data(), rhs.data(), &ni, &info);
  handle_gesv_info(info);
}

} // namespace cppgw::detail
