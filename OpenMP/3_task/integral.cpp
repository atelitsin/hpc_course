#include <iostream>
#include <omp.h>


template <typename F>
double integrate_single_thread(F func, double a, double b, std::size_t n) {
    double sum = 0.0;
    const double h = (b - a) / static_cast<double>(n);
    for (std::size_t i = 0; i < n; i++) {
        const double x = a + (static_cast<double>(i) + 0.5) * h;
        sum += func(x);
    }
    return sum * h;
}


template <typename F>
double integrate_multi_thread(F func, double a, double b, std::size_t n) {
    double sum = 0.0;
    const double h = (b - a) / static_cast<double>(n);
#pragma omp parallel for reduction(+ : sum)
    for (std::size_t i = 0; i < n; i++) {
        const double x = a + (static_cast<double>(i) + 0.5) * h;
        sum += func(x);
    }
    return sum * h;
}


double f(double x) {
    return 2 * x * x;
}


int main() {
    std::cout << integrate_single_thread(f, 0, 5, 10000) << std::endl;

    std::cout << integrate_multi_thread(f, 0, 5, 10000);
}