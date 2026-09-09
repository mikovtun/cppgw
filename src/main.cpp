#include "main.hpp"

int main(int argc, char** argv) {
  using namespace cppgw;
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
  Tensor<double, Executor::Host>::gemm(matrix1, matrix2, matrix3,
                                        "y", "x");
  matrix3.print();

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

  return 0;
}
