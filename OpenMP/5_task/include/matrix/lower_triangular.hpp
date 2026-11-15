#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hpc {

template <typename T>
class LowerTriangularMatrix {
   public:
    using value_type = T;

    explicit LowerTriangularMatrix(std::size_t n)
        : n_(n), data_(n * (n + 1) / 2) {}

    LowerTriangularMatrix(std::size_t n, std::vector<T> data)
        : n_(n), data_(std::move(data)) {
        if (data_.size() != n * (n + 1) / 2) {
            throw std::invalid_argument(
                "Invalid vector size for Lower Triangular Matrix");
        }
    }

    std::size_t nrows() const noexcept { return n_; }
    std::size_t ncols() const noexcept { return n_; }

    const T* row_data(std::size_t i) const noexcept {
        return data_.data() + i * (i + 1) / 2;
    }
    std::size_t row_size(std::size_t i) const noexcept { return i + 1; }

    T& operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * (i + 1) / 2 + j];
    }
    const T& operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * (i + 1) / 2 + j];
    }

   private:
    std::size_t n_;
    std::vector<T> data_;
};

}  // namespace hpc
