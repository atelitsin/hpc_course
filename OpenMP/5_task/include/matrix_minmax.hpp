#pragma once

#include <stdexcept>
#include <vector>

inline int find_max_of_mins_lower_triangular_single_thread(const std::vector<int> &matrix,
                                                           size_t dimension) {
    if (dimension == 0) {
        throw std::invalid_argument("Размерность матрицы не может быть нулевой");
    }
    if (matrix.size() != (dimension * (dimension + 1) / 2)) {
        throw std::invalid_argument("Количество элементов матрицы не соответствует ее размерности");
    }

    
}

inline int find_max_of_mins_upper_triangular_single_thread();