#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

class LowerDiagonalMatrix {
  public:
    LowerDiagonalMatrix(std::size_t dim, std::vector<int> data)
        : dim_(dim), data_(std::move(data)) {
        if (dim_ == 0) {
            throw std::invalid_argument("Размерность матрицы не может быть нулевой");
        }
        if (data_.size() != dim / 2 * (dim + 1)) {
            throw std::invalid_argument(
                "Количество элементов матрицы не соответствует ее размерности");
        }
    }

    std::size_t size() const {
        return dim_;
    }


  private:
    size_t dim_;
    std::vector<int> data_;
}