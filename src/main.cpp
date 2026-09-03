#include "types.hpp"
#include "grid.hpp"
#include <iostream>
#include <cassert>

namespace cppgw {


}
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
  return 0;
}
