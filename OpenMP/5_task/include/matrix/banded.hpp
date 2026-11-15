#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace hpc {

template <typename T>
class BandedMatrix {
   public:
    using value_type = T;

    BandedMatrix(std::size_t n, std::size_t kl, std::size_t ku)
        : n_(n), kl_(kl), ku_(ku), stride_(kl + ku + 1), data_(n * stride_) {}

    std::size_t nrows() const noexcept { return n_; }
    std::size_t ncols() const noexcept { return n_; }
    std::size_t kl() const noexcept { return kl_; }
    std::size_t ku() const noexcept { return ku_; }

    std::size_t nnz() const noexcept {
        std::size_t total = 0;
        for (std::size_t i = 0; i < n_; ++i) total += row_size(i);
        return total;
    }

    const T* row_data(std::size_t i) const noexcept {
        std::size_t left_pad = (i < kl_) ? (kl_ - i) : 0;
        return data_.data() + i * stride_ + left_pad;
    }

    std::size_t row_size(std::size_t i) const noexcept {
        std::size_t j_start = (i < kl_) ? 0 : (i - kl_);
        std::size_t j_end = std::min(n_ - 1, i + ku_);
        return j_end - j_start + 1;
    }

    T& operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * stride_ + kl_ + (j - i)];
    }
    const T& operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * stride_ + kl_ + (j - i)];
    }

   private:
    std::size_t n_, kl_, ku_, stride_;
    std::vector<T> data_;
};

}  // namespace hpc