#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

inline void check_matrix_correctness(const std::vector<std::vector<int>> &matrix, std::size_t rows,
                                     std::size_t columns) {
    if (rows == 0 || columns == 0) {
        throw std::invalid_argument("Число строк и столбцов должно быть положительным");
    }
    if (matrix.size() != rows) {
        throw std::invalid_argument("Число строк матрицы не совпадает с заданным");
    }
    for (std::size_t i = 0; i < rows; i++) {
        if (matrix[i].size() != columns) {
            throw std::invalid_argument("Длина строки матрицы не совпадает с числом столбцов");
        }
    }
}

inline int find_max_of_mins_single_thread(const std::vector<std::vector<int>> &matrix,
                                          std::size_t rows, std::size_t columns) {
    check_matrix_correctness(matrix, rows, columns);

    int result = std::numeric_limits<int>::lowest();
    for (std::size_t i = 0; i < rows; i++) {
        int row_min = matrix[i][0];
        for (std::size_t j = 1; j < columns; j++) {
            row_min = std::min(row_min, matrix[i][j]);
        }
        result = std::max(row_min, result);
    }
    return result;
}

inline int find_max_of_mins_multi_thread(const std::vector<std::vector<int>> &matrix,
                                         std::size_t rows, std::size_t columns) {
    check_matrix_correctness(matrix, rows, columns);

    int result = std::numeric_limits<int>::lowest();
#pragma omp parallel for reduction(max : result)
    for (std::size_t i = 0; i < rows; i++) {
        int row_min = matrix[i][0];
        for (std::size_t j = 1; j < columns; j++) {
            row_min = std::min(row_min, matrix[i][j]);
        }
        result = std::max(row_min, result);
    }
    return result;
}