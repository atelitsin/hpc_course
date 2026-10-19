#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

void print_vector(const std::vector<int> &arr) {
    int length = arr.size();
    for (int i = 0; i < length; ++i) {
        std::cout << arr[i] << " ";
    }
}

std::vector<int> generate_vector(const int length, const int min_val, const int max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(min_val, max_val);

    std::vector<int> result(length);
    for (int i = 0; i < length; i++) {
        result[i] = distrib(gen);
    }
    return result;
}

int vector_max_parallel(const std::vector<int> &arr) {
    // std::vector<int> owner(arr.size());
    int length = arr.size();
    int arr_max = std::numeric_limits<int>::lowest();
#pragma omp parallel
    {
        int local_max = std::numeric_limits<int>::lowest();

#pragma omp for
        for (int i = 0; i < length; i++) {
            local_max = std::max(arr[i], local_max);
            // owner[i] = omp_get_thread_num();
        }

#pragma omp critical
        {
            arr_max = std::max(arr_max, local_max);
        }
    }
    // print_vector(owner);
    return arr_max;
}

int vector_max_single(const std::vector<int> &arr) {
    int arr_max = std::numeric_limits<int>::lowest();
    int length = arr.size();
    for (int i = 0; i < length; i++) {
        arr_max = std::max(arr_max, arr[i]);
    }
    return arr_max;
}

int main() {
    std::vector<int> a = generate_vector(1'000'000'000, -1000, 1000);
    // print_vector(a);
    double start = omp_get_wtime();
    vector_max_parallel(a);
    double end = omp_get_wtime();
    double parallel_time = end - start;
    std::cout << "Параллельный алгоритм:" << parallel_time << std::endl;
    start = omp_get_wtime();
    vector_max_single(a);
    end = omp_get_wtime();
    double single_time = end - start;
    std::cout << "Однопоточный алгоритм:" << single_time << std::endl;
    std::cout << "Параллельный алгоритм ускорил работу в " << single_time / parallel_time << "раз"
              << std::endl;
    std::cout << omp_get_max_threads();
}