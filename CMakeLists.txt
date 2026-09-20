cmake_minimum_required(VERSION 3.20)
project(cppgw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

include(FetchContent)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Boost REQUIRED)

FetchContent_Declare(
  HighFive
  GIT_REPOSITORY https://github.com/highfive-devs/highfive
  GIT_TAG v3.3.0
)

FetchContent_MakeAvailable(HighFive)


add_executable(cppgw
  src/main.cpp
  src/input.cpp
  src/gf2.cpp
  src/tests.cpp
  src/fourier_tests.cpp
  src/inverse_fourier_tests.cpp
  src/lapack_solve.cpp)

target_link_libraries(cppgw
  PRIVATE
    BLAS::BLAS
    LAPACK::LAPACK
    HighFive
)
target_include_directories(cppgw PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${Boost_INCLUDE_DIRS}
)
