#pragma once

#include <concepts>
#include "boost/multiprecision/cpp_bin_float.hpp"
#include "boost/multiprecision/number.hpp"

namespace cppgw {
  namespace numerics {
    // --- Constants ---
    template <typename T>
    T pi() { return T("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651e+00"); }

    // --- Convenient types ---
    using float50 = boost::multiprecision::cpp_bin_float_50;

    // --- Concept definitions ---

    template <typename T>
    concept NativeRealFloatingPoint = std::floating_point<T>;
    
    // A boost::multiprecision::number<...> type whose backend is classified
    // as a real (non-complex) floating point.
    // number_category<T>::value is an enum of number_category_type (number_kind_*).
    template <typename T>
    concept BoostRealFloatingPoint =
      requires { typename T::backend_type; } &&      // must be a boost multiprecision type
      boost::multiprecision::number_category<T>::value ==
        boost::multiprecision::number_kind_floating_point;

      template <typename T>
        concept RealFloatingPoint = NativeRealFloatingPoint<T> || BoostRealFloatingPoint<T>;

      // Standard std::complex<T> where T is a real float
      template <typename T>
        concept NativeComplexFloatingPoint = requires {
          typename T::value_type;
        } && NativeRealFloatingPoint<typename T::value_type>
      && std::same_as<T, std::complex<typename T::value_type>>;

      // Boost complex multiprecision type
      template <typename T>
        concept BoostComplexFloatingPoint =
          requires { typename T::backend_type; } &&
          boost::multiprecision::number_category<T>::value ==
            boost::multiprecision::number_kind_complex;

      template <typename T>
        concept ComplexFloatingPoint = NativeComplexFloatingPoint<T> || BoostComplexFloatingPoint<T>;

      template <typename T>
        concept FloatingPoint = RealFloatingPoint<T> || ComplexFloatingPoint<T>;

    // --- Tests ---
    static_assert( NativeRealFloatingPoint<float>);
    static_assert( NativeRealFloatingPoint<double>);
    static_assert( NativeRealFloatingPoint<long double>);
    static_assert(!NativeRealFloatingPoint<int>);

    static_assert(!BoostRealFloatingPoint<float>);
    static_assert(!BoostRealFloatingPoint<double>);
    static_assert( BoostRealFloatingPoint<float50>);

    static_assert( RealFloatingPoint<double>);
    static_assert( RealFloatingPoint<float50>);
    static_assert( NativeComplexFloatingPoint<std::complex<double>>);
    static_assert(!NativeComplexFloatingPoint<double>);
    static_assert( ComplexFloatingPoint<std::complex<double>>);

    // TODO: flesh out tests
  }
}
