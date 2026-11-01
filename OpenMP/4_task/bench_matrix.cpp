// Бенчмарк: максимум среди минимумов строк матрицы N x N.
// Подключает matrix_minmax.hpp обычным #include (файл лежит рядом).
// В stdout уходит только CSV, диагностика и ошибки идут в stderr.
//
// Сборка:  g++-16 -fopenmp -O2 -Wall -Wextra bench_matrix.cpp -o bench_matrix
// Запуск:  ./bench_matrix [N1,N2,...] [reps] [max_threads] > bench_matrix.csv
//   N1,N2,...    сторона квадратной матрицы    (по умолчанию 100,300,1000,3000,10000)
//   reps         число замеров на точку        (по умолчанию 10)
//   max_threads  максимум потоков              (по умолчанию omp_get_max_threads())
//
// В CSV колонка n это сторона матрицы N (число элементов N * N).
// Память: матрица N x N из int, при N = 10000 около 400 МБ.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>
#include <vector>

#include "matrix_minmax.hpp"

#ifndef _OPENMP
#error "Соберите с флагом -fopenmp, иначе параллельная версия будет работать последовательно"
#endif

using Matrix = std::vector<std::vector<int>>;

// Секунды по монотонным часам, разрешение в наносекунды
double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

Matrix generate_matrix(std::size_t rows, std::size_t columns, int min_val, int max_val,
                       unsigned seed) {
    std::mt19937 gen(seed);  // фиксированный сид: данные одинаковы при каждом запуске
    std::uniform_int_distribution<int> distrib(min_val, max_val);

    Matrix matrix(rows, std::vector<int>(columns));
    for (auto &row : matrix) {
        for (int &x : row) {
            x = distrib(gen);
        }
    }
    return matrix;
}

// Независимый эталон (через std::min_element) для проверки правильности обеих версий
int reference_max_of_mins(const Matrix &matrix) {
    int best = std::numeric_limits<int>::lowest();
    for (const auto &row : matrix) {
        best = std::max(best, *std::min_element(row.begin(), row.end()));
    }
    return best;
}

// Делает reps замеров и печатает строки CSV. Возвращает false, если результат неверный.
// Run: вызываемый объект без аргументов, возвращающий int.
template <typename Run>
bool bench(const char *variant, Run run, std::size_t n, int threads, int reps, int ref) {
    omp_set_num_threads(threads);

    run();  // прогрев: создание пула потоков, подгрузка страниц памяти (в CSV не пишем)

    for (int rep = 1; rep <= reps; rep++) {
        double start = now_seconds();
        int result = run();
        double end = now_seconds();  // печать только после остановки таймера

        // значения целые, поэтому сравнение точное
        if (result != ref) {
            std::cerr << "ОШИБКА: " << variant << ", N=" << n << ", потоков " << threads
                      << ": получили " << result << ", ожидали " << ref << '\n';
            return false;
        }
        std::cout << variant << "," << n << "," << threads << "," << rep << "," << (end - start)
                  << "\n";
    }
    return true;
}

// Разбирает список размеров вида "100,1000,10000"
std::vector<std::size_t> parse_sizes(const char *s) {
    std::vector<std::size_t> sizes;
    char *end = nullptr;
    while (*s) {
        sizes.push_back(std::strtoull(s, &end, 10));
        if (end == s) break;  // не число: прекращаем разбор
        s = (*end == ',') ? end + 1 : end;
    }
    return sizes;
}

int main(int argc, char *argv[]) {
    std::vector<std::size_t> sizes = (argc > 1)
                                         ? parse_sizes(argv[1])
                                         : std::vector<std::size_t>{100, 300, 1000, 3000, 10000};
    const int reps = (argc > 2) ? std::atoi(argv[2]) : 10;
    const int max_threads = (argc > 3) ? std::atoi(argv[3]) : omp_get_max_threads();

    if (sizes.empty() || reps < 1 || max_threads < 1) {
        std::cerr << "Использование: ./bench_matrix [N1,N2,...] [reps] [max_threads]\n";
        return 2;
    }
    if (std::find(sizes.begin(), sizes.end(), std::size_t{0}) != sizes.end()) {
        std::cerr << "Размер матрицы N должен быть положительным.\n";
        return 2;
    }

    std::cerr << "reps=" << reps << ", max_threads=" << max_threads
              << ", размеров: " << sizes.size() << std::endl;

    std::cout.precision(9);
    std::cout << std::fixed;
    std::cout << "variant,n,threads,rep,seconds\n";

    try {
        for (std::size_t n : sizes) {
            const Matrix matrix = generate_matrix(n, n, -1000, 1000, 1);  // один раз на размер
            const int ref = reference_max_of_mins(matrix);                // эталон

            auto run_seq = [&]() { return find_max_of_mins_single_thread(matrix, n, n); };
            auto run_par = [&]() { return find_max_of_mins_multi_thread(matrix, n, n); };

            // база: последовательная версия (threads = 1)
            if (!bench("seq", run_seq, n, 1, reps, ref)) return 1;

            // параллельная версия для 1..max_threads потоков
            for (int t = 1; t <= max_threads; t++) {
                if (!bench("reduction", run_par, n, t, reps, ref)) return 1;
            }
            std::cerr << "N=" << n << " готово" << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "Исключение: " << e.what() << '\n';
        return 1;
    }
    return 0;
}