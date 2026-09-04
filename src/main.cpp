#include "types.hpp"
#include "grid.hpp"
#include "tensor.hpp"
#include <iostream>
#include <cassert>
#include <complex>

int main(int argc, char** argv) {
  using namespace cppgw;
  static_assert(Grid<MatsubaraGrid<Fermionic>>);
  auto beta = InverseTemperature(10.0);
  MatsubaraGrid<Fermionic> fgrid(4, beta);
  MatsubaraGrid<Bosonic> bgrid(4, beta);
  std::cout << "Fermionic grid" << std::endl;
  for(auto& p : fgrid.full_points()) 
    std::cout << p.value << " ";
  std::cout << std::endl << "Bosonic grid" << std::endl;
  for(auto& p : bgrid.full_points()) 
    std::cout << p.value << " ";
  std::cout << std::endl;
  std::cout << bgrid.N() << std::endl;

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
  return 0;
}
