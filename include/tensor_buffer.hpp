#pragma once
#include "common.hpp"
#include <vector>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <algorithm>

namespace cppgw {

// TensorBuffer<T, E> is the storage primitive: it owns 'count' elements of type T living on Executor E.
// Pointer management, lifetime, resizes, and copies are handled by this class
// All math lives in TensorBackend, not here

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
    // Grow the buffer by a new axis of size K built on top of the existing data.
    // The old element i lands at new offset:
    //   slow axis (slowest): k*N + i
    //   fast  axis (fastest): i*K + k
    // if fill = true, every slice k copies element i (broadcast/tile);
    // otherwise only slice k = 0 contains data, the rest is zero.
    // Single allocation: the new store is filled in one pass and swapped in.
    void resize_new_dim(size_t K, bool new_dim_slow, bool fill) {
        if (K == 0) {
            delete[] data_;
            data_ = nullptr;
            size_ = 0;
            return;
        }
        if (K == 1) return;   // layout unchanged

        const size_t N = size_;
        T* nb = new T[N * K]();

        if (new_dim_slow) {
            // K consecutive N-sized blocks, all equal
            const size_t blocks = fill ? K : size_t(1);
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memcpy(nb, data_, N * sizeof(T));
                for (size_t k = 1; k < blocks; ++k)
                    std::memcpy(nb + k * N, nb, N * sizeof(T));
            } else {
                std::copy(data_, data_ + N, nb);
                for (size_t k = 1; k < blocks; ++k)
                    std::copy(nb, nb + N, nb + k * N);
            }
        } else {
            // tile element i across the fastest axis
            const size_t depth = fill ? K : size_t(1);
            for (size_t i = 0; i < N; ++i)
                for (size_t k = 0; k < depth; ++k)
                    nb[i * K + k] = data_[i];
        }

        delete[] data_;
        data_ = nb;
        size_ = N * K;
    }

};




}
