#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

inline void check_matrix_correctness(const int* matrix, std::size_t rows,
                                     std::size_t columns) {
    if (matrix == nullptr) {
        throw std::invalid_argument(
            "Указатель на матрицу не может быть nullptr");
    }
    if (rows == 0 || columns == 0) {
        throw std::invalid_argument(
            "Число строк и столбцов должно быть положительным");
    }
}

inline int find_max_of_mins_single_thread(const int* matrix, std::size_t rows,
                                          std::size_t columns) {
    check_matrix_correctness(matrix, rows, columns);

    int result = std::numeric_limits<int>::lowest();
    for (std::size_t i = 0; i < rows; i++) {
        const int* row = matrix + i * columns;
        int row_min = row[0];
        for (std::size_t j = 1; j < columns; ++j) {
            row_min = std::min(row_min, row[j]);
        }
        result = std::max(result, row_min);
    }
    return result;
}

inline int find_max_of_mins_multi_thread(const int* matrix, std::size_t rows,
                                         std::size_t columns) {
    check_matrix_correctness(matrix, rows, columns);
    int result = std::numeric_limits<int>::lowest();
#pragma omp parallel for reduction(max : result)
    for (std::size_t i = 0; i < rows; i++) {
        const int* row = matrix + i * columns;
        int row_min = row[0];
        for (std::size_t j = 1; j < columns; ++j) {
            row_min = std::min(row_min, row[j]);
        }
        result = std::max(result, row_min);
    }
    return result;
}
