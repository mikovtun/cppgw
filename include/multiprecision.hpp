#pragma once

#include <concepts>
#include "boost/multiprecision/cpp_dec_float.hpp"
#include "boost/multiprecision/number.hpp"

namespace cppgw {
  namespace numerics {
    // --- Constants ---
    template <typename T>
    T pi() { return T("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651e+00"); }

    // --- Concept definitions ---

    template <typename T>
    concept NativeRealFloatingPoint = std::floating_point<T>;
    
    template <typename T>
    concept BoostRealFloatingPoint = requires {
      typename T::backend_type;
    } && std::is_floating_point_v<typename boost::multiprecision::number_category<T>::type>
      && std::derived_from<
        T,
        boost::multiprecision::number<
          typename T::backend_type,
        T::thread_safe ? boost::multiprecision::et_on : boost::multiprecision::et_off?
          >
          >;

      template <typename T>
        concept RealFloatingPoint = NativeRealFloatingPoint<T> || BoostRealFloatingPoint<T>;

      // Standard std::complex<T> where T is a real float
      template <typename T>
        concept NativeComplexFloatingPoint = requires {
          typename T::value_type;
        } && NativeRealFloatingPoint<typename T::value_type>
      && std::same_as<T, std::complex<typename T::value_type>>;

      // Boost cpp_complex<Precision> or multiprecision complex types
      template <typename T>
        concept BoostComplexFloatingPoint = requires {
          typename T::backend_type;
        } && std::derived_from<
      T, 
        boost::multiprecision::number<
          typename T::backend_type, 
        T::thread_safe ? boost::multiprecision::et_on : boost::multiprecision::et_off
          >
          >
          // Boost classifies complex numbers under complex_number_type
          && std::is_same_v<
          typename boost::multiprecision::number_category<T>::type, 
        boost::multiprecision::complex_number_type
          >;

      template <typename T>
        concept AnyComplexFloatingPoint = NativeComplexFloatingPoint<T> || BoostComplexFloatingPoint<T>;

    // --- Tests ---
    static_assert( NativeRealFloatingPoint<float>);
    static_assert( NativeRealFloatingPoint<double>);
    static_assert( NativeRealFloatingPoint<long double>);
    static_assert(!NativeRealFloatingPoint<int>);

    static_assert(!BoostRealFloatingPoint<float>);
    static_assert(!BoostRealFloatingPoint<double>);
    static_assert( BoostRealFloatingPoint<cpp_bin_float_50>);

    // TODO: flesh out tests
  }
}
