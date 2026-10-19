// Бенчмарк поиска максимума: последовательная и две параллельные версии.
// В stdout уходит только CSV, всё остальное (диагностика, ошибки) идёт в stderr.
//
// Сборка:  g++-16 -fopenmp -O2 -Wall -Wextra bench.cpp -o bench
// Запуск:  ./bench [n1,n2,...] [reps] [max_threads] > bench.csv
//   n1,n2,...    размеры вектора через запятую (по умолчанию 1e3 ... 1e8, степени десятки)
//   reps         число замеров на точку        (по умолчанию 10)
//   max_threads  максимум потоков              (по умолчанию omp_get_max_threads())

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>
#include <vector>

using MaxFn = int (*)(const std::vector<int> &);

std::vector<int> generate_vector(size_t length, int min_val, int max_val) {
    std::mt19937 gen(42);  // фиксированный сид: данные одинаковы при каждом запуске
    std::uniform_int_distribution<int> distrib(min_val, max_val);

    std::vector<int> result(length);
    for (size_t i = 0; i < length; i++) {
        result[i] = distrib(gen);
    }
    return result;
}

// Последовательная версия (база для сравнения)
int vector_max_single(const std::vector<int> &arr) {
    int arr_max = std::numeric_limits<int>::lowest();
    long long length = arr.size();
    for (long long i = 0; i < length; i++) {
        arr_max = std::max(arr_max, arr[i]);
    }
    return arr_max;
}

// Параллельная версия: локальные максимумы + слияние в critical
int vector_max_critical(const std::vector<int> &arr) {
    long long length = arr.size();
    int arr_max = std::numeric_limits<int>::lowest();
    #pragma omp parallel
    {
        int local_max = std::numeric_limits<int>::lowest();

        #pragma omp for
        for (long long i = 0; i < length; i++) {
            local_max = std::max(arr[i], local_max);
        }

        #pragma omp critical
        {
            arr_max = std::max(arr_max, local_max);
        }
    }
    return arr_max;
}

// Параллельная версия через reduction
int vector_max_reduction(const std::vector<int> &arr) {
    long long length = arr.size();
    int arr_max = std::numeric_limits<int>::lowest();
    #pragma omp parallel for reduction(max : arr_max)
    for (long long i = 0; i < length; i++) {
        arr_max = std::max(arr_max, arr[i]);
    }
    return arr_max;
}

// Делает reps замеров и печатает строки CSV. Возвращает false, если результат неверный.
bool bench(const char *variant, MaxFn fn, const std::vector<int> &a,
           int threads, int reps, int ref) {
    const size_t n = a.size();
    omp_set_num_threads(threads);

    fn(a);  // прогрев: создание пула потоков, подгрузка страниц памяти (в CSV не пишем)

    for (int rep = 1; rep <= reps; rep++) {
        double start = omp_get_wtime();
        int result = fn(a);
        double end = omp_get_wtime();  // печать только после остановки таймера

        if (result != ref) {
            std::cerr << "ОШИБКА: " << variant << ", потоков " << threads
                      << ": получили " << result << ", ожидали " << ref << std::endl;
            return false;
        }
        std::cout << variant << "," << n << "," << threads << "," << rep << ","
                  << (end - start) << "\n";
    }
    return true;
}

// Разбирает список размеров вида "1000,10000,1000000"
std::vector<size_t> parse_sizes(const char *s) {
    std::vector<size_t> sizes;
    char *end = nullptr;
    while (*s) {
        sizes.push_back(std::strtoull(s, &end, 10));
        if (end == s) break;  // не число: прекращаем разбор
        s = (*end == ',') ? end + 1 : end;
    }
    return sizes;
}

int main(int argc, char *argv[]) {
    std::vector<size_t> sizes = (argc > 1)
        ? parse_sizes(argv[1])
        : std::vector<size_t>{1'000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000};
    int reps = (argc > 2) ? std::atoi(argv[2]) : 10;
    int max_threads = (argc > 3) ? std::atoi(argv[3]) : omp_get_max_threads();

    if (sizes.empty() || reps < 1 || max_threads < 1) {
        std::cerr << "Использование: ./bench [n1,n2,...] [reps] [max_threads]" << std::endl;
        return 2;
    }

    std::cerr << "reps=" << reps << ", max_threads=" << max_threads << ", размеров: "
              << sizes.size() << std::endl;

    std::cout.precision(9);
    std::cout << std::fixed;
    std::cout << "variant,n,threads,rep,seconds\n";

    for (size_t n : sizes) {
        std::vector<int> a = generate_vector(n, -1000, 1000);  // один раз на размер
        int ref = vector_max_single(a);  // эталон для проверки параллельных версий

        // база: последовательная версия (threads = 1)
        if (!bench("seq", vector_max_single, a, 1, reps, ref)) return 1;

        // параллельные версии для 1..max_threads потоков
        for (int t = 1; t <= max_threads; t++) {
            if (!bench("critical", vector_max_critical, a, t, reps, ref)) return 1;
            if (!bench("reduction", vector_max_reduction, a, t, reps, ref)) return 1;
        }
        std::cerr << "n=" << n << " готово" << std::endl;
    }
    return 0;
}