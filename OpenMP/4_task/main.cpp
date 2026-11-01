#include <iostream>
#include <vector>

#include "matrix_minmax.hpp"


int main() {
    std::vector<std::vector<int>> a = {{1, 2},{3, 4}};
    std::cout << find_max_of_mins_single_thread(a, 2, 2);
}