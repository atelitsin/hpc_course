#pragma once

#include <cstddef>
#include <vector>

namespace hpc {

template <typename T>
class DenseMatrix {
   public:
    using value_type = T;

    explicit DenseMatrix(std::size_t n) : n_(n), data_(n * n) {}

    DenseMatrix(std::size_t n, std::vector<T> data) : n_(n), data_(std::move(data)) {}

    std::size_t nrows() const noexcept { return n_; }
    std::size_t ncols() const noexcept { return n_; }

    const T* row_data(std::size_t i) const noexcept { return data_.data() + i * n_; }
    std::size_t row_size(std::size_t /*i*/) const noexcept { return n_; }

    T& operator()(std::size_t i, std::size_t j) noexcept { return data_[i * n_ + j]; }
    const T& operator()(std::size_t i, std::size_t j) const noexcept { return data_[i * n_ + j]; }

   private:
    std::size_t n_;
    std::vector<T> data_;
};

}  // namespace hpc