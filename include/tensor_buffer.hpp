#pragma once
#include "enums.hpp"
#include <vector>
#include <memory>

namespace cppgw {

// TensorBuffer<T, E> is the storage primitive: it owns 'count' elements of type T living on Executor E.
// It intentionally exposes a minimal interface (data(), size()).
// All math lives in LinAlgBackend, not here

// Forward declare generic
template <typename T, typename Executor>
class TensorBuffer;

// Host storage
template <typename T>
class TensorBuffer<T, Executor::Host> {
private:
    size_t size_;
    T*     data_;
public:
    using E = Executor::Host;
    explicit TensorBuffer(size_t count)
        : size_(count), data_(count ? new T[count]() : nullptr) {}

    ~TensorBuffer() { delete[] data_; }

    TensorBuffer(const TensorBuffer&)            = delete;
    TensorBuffer& operator=(const TensorBuffer&) = delete;
    TensorBuffer(TensorBuffer&&)                 = delete;
    TensorBuffer& operator=(TensorBuffer&&)      = delete;

    T*       data()       { return data_; }
    const T* data() const { return data_; }
    size_t   size() const { return size_; }
};




}
