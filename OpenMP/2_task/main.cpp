#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <vector>

void check_same_size(const std::vector<int> &a, const std::vector<int> &b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Длины векторов не совпадают. Скалярное произведение не может быть определено.");
    }
}

long long scalar_product_single_thread(const std::vector<int> &a, const std::vector<int> &b) {
    check_same_size(a, b);
    size_t length = a.size();
    long long result = 0;
    for (size_t i = 0; i < length; i++) {
        result += static_cast<long long>(a[i]) * b[i];
    }
    return result;
}

long long scalar_product_parallel(const std::vector<int> &a, const std::vector<int> &b) {
    check_same_size(a, b);
    size_t length = a.size();
    long long result = 0;
#pragma omp parallel for reduction(+ : result)
    for (size_t i = 0; i < length; i++) {
        result += static_cast<long long>(a[i]) * b[i];
    }
    return result;
}

int main() {
    std::vector<int> a = {1, 1, 3, 5, 10};
    std::vector<int> b = {2, 2, 1, 4};
    try {
        long long result_single = scalar_product_single_thread(a, b);
        long long result_parallel = scalar_product_parallel(a, b);
        std::cout << result_single << std::endl;
        std::cout << result_parallel << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}