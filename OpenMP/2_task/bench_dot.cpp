// Бенчмарк скалярного произведения: последовательная и параллельная (reduction) версии.
// В stdout уходит только CSV, диагностика и ошибки идут в stderr.
//
// Сборка:  g++-16 -fopenmp -O2 -Wall -Wextra bench_dot.cpp -o bench_dot
// Запуск:  ./bench_dot [n1,n2,...] [reps] [max_threads] > bench_dot.csv
//   n1,n2,...    размеры векторов через запятую (по умолчанию 1e3 ... 1e8, степени десятки)
//   reps         число замеров на точку        (по умолчанию 10)
//   max_threads  максимум потоков              (по умолчанию omp_get_max_threads())
//
// Память: два вектора по n элементов int, то есть при n = 1e8 около 800 МБ.

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <vector>

// ---------- функции пользователя (без изменений) ----------

void check_same_size(const std::vector<int> &a, const std::vector<int> &b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Длины векторов не совпадают. Скалярное произведение не может быть определено.");
    }
}

long long scalar_product_single_thread(const std::vector<int> &a, const std::vector<int> &b) {
    check_same_size(a, b);
    size_t length = a.size();
    long long result = 0;
    for (size_t i = 0; i < length; i++) {
        result += static_cast<long long>(a[i]) * b[i];
    }
    return result;
}

long long scalar_product_parallel(const std::vector<int> &a, const std::vector<int> &b) {
    check_same_size(a, b);
    size_t length = a.size();
    long long result = 0;
#pragma omp parallel for reduction(+ : result)
    for (size_t i = 0; i < length; i++) {
        result += static_cast<long long>(a[i]) * b[i];
    }
    return result;
}

// ---------- бенчмарк ----------

using DotFn = long long (*)(const std::vector<int> &, const std::vector<int> &);

// Секунды по монотонным часам: разрешение в наносекунды, не зависит от omp_get_wtime
double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::vector<int> generate_vector(std::size_t length, int min_val, int max_val, unsigned seed) {
    std::mt19937 gen(seed);  // фиксированный сид: данные одинаковы при каждом запуске
    std::uniform_int_distribution<int> distrib(min_val, max_val);

    std::vector<int> result(length);
    for (std::size_t i = 0; i < length; i++) {
        result[i] = distrib(gen);
    }
    return result;
}

// Делает reps замеров и печатает строки CSV. Возвращает false, если результат неверный.
bool bench(const char *variant, DotFn fn, const std::vector<int> &a, const std::vector<int> &b,
           int threads, int reps, long long ref) {
    const std::size_t n = a.size();
    omp_set_num_threads(threads);

    fn(a, b);  // прогрев: создание пула потоков, подгрузка страниц памяти (в CSV не пишем)

    for (int rep = 1; rep <= reps; rep++) {
        double start = now_seconds();
        long long result = fn(a, b);
        double end = now_seconds();  // печать только после остановки таймера

        if (result != ref) {
            std::cerr << "ОШИБКА: " << variant << ", потоков " << threads << ": получили "
                      << result << ", ожидали " << ref << std::endl;
            return false;
        }
        std::cout << variant << "," << n << "," << threads << "," << rep << "," << (end - start)
                  << "\n";
    }
    return true;
}

// Разбирает список размеров вида "1000,10000,1000000"
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
    std::vector<std::size_t> sizes =
        (argc > 1) ? parse_sizes(argv[1])
                   : std::vector<std::size_t>{1'000,     10'000,     100'000,
                                              1'000'000, 10'000'000, 100'000'000};
    int reps = (argc > 2) ? std::atoi(argv[2]) : 10;
    int max_threads = (argc > 3) ? std::atoi(argv[3]) : omp_get_max_threads();

    if (sizes.empty() || reps < 1 || max_threads < 1) {
        std::cerr << "Использование: ./bench_dot [n1,n2,...] [reps] [max_threads]" << std::endl;
        return 2;
    }

    std::cerr << "reps=" << reps << ", max_threads=" << max_threads
              << ", размеров: " << sizes.size() << std::endl;

    std::cout.precision(9);
    std::cout << std::fixed;
    std::cout << "variant,n,threads,rep,seconds\n";

    try {
        for (std::size_t n : sizes) {
            std::vector<int> a = generate_vector(n, -1000, 1000, 1);  // два разных сида,
            std::vector<int> b = generate_vector(n, -1000, 1000, 2);  // чтобы a и b различались
            long long ref = scalar_product_single_thread(a, b);        // эталон для проверки

            // база: последовательная версия (threads = 1)
            if (!bench("seq", scalar_product_single_thread, a, b, 1, reps, ref)) return 1;

            // параллельная версия для 1..max_threads потоков
            for (int t = 1; t <= max_threads; t++) {
                if (!bench("reduction", scalar_product_parallel, a, b, t, reps, ref)) return 1;
            }
            std::cerr << "n=" << n << " готово" << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "Исключение: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}