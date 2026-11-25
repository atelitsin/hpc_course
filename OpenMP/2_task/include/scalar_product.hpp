#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

inline void check_same_size(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Длины векторов не совпадают. Скалярное произведение не может быть определено.");
    }
}

inline long long scalar_product_single_thread(const std::vector<int>& a, const std::vector<int>& b) {
    check_same_size(a, b);
    std::size_t length = a.size();
    long long result = 0;
    for (std::size_t i = 0; i < length; i++) {
        result += static_cast<long long>(a[i]) * b[i];
    }
    return result;
}

inline long long scalar_product_parallel(const std::vector<int>& a, const std::vector<int>& b) {
    check_same_size(a, b);
    std::size_t length = a.size();
    long long result = 0;
#pragma omp parallel for reduction(+ : result)
    for (std::size_t i = 0; i < length; i++) {
        result += static_cast<long long>(a[i]) * b[i];
    }
    return result;
}
