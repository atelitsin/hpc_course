#include <iostream>
#include <omp.h>
#include <random>
#include <vector>


template <typename F>
double integrate_single_thread(F func, double a, double b, size_t n) {
    double sum = 0.0;
    const double h = (b - a) / static_cast<double>(n);
    for (size_t i = 0; i < n; i++) {
        const double x = a + (static_cast<double>(i) + 0.5) * h;
        sum += func(x);
    }
    return sum * h;
}



void print_vector(const std::vector<int> &arr) {
    size_t length = arr.size();
    for (size_t i = 0; i < length; ++i) {
        std::cout << arr[i] << " ";
    }
}


double func(double x) {
    return 2 * x * x;
}


int main() {
    std::cout << integrate_single_thread(func, 0, 5, 10000);
}