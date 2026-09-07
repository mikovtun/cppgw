#pragma once
#include <type_traits>
#include <iterator>

namespace cppgw {
  
  struct Executor {
    struct Host {};
    struct Device {};
  };

  // Element type of a contiguous container/range (std::vector, std::span,
  // std::array, C-array). Yields an error in a SFINAE context for scalars,
  // so it can be used to detect "range of scalars" vs "scalar".
  template <class C>
  using Elem = std::decay_t<decltype(*std::begin(std::declval<const C&>()))>;

}

