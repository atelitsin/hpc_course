// Бенчмарк численного интегрирования. Подключает файл пользователя и использует
// его шаблоны integrate_single_thread / integrate_multi_thread и функцию f.
// В stdout уходит только CSV, диагностика и ошибки идут в stderr.
//
// Сборка (файл с интегралом по умолчанию называется integral.cpp и лежит рядом):
//   g++-16 -fopenmp -O2 -Wall -Wextra bench_integral.cpp -o bench_integral
// Если файл называется иначе, укажи имя при сборке:
//   g++-16 -fopenmp -O2 -DINTEGRAL_SRC='"my_file.cpp"' bench_integral.cpp -o bench_integral
//
// Запуск: ./bench_integral <cheap|heavy> [n1,n2,...] [reps] [max_threads] > out.csv
//   cheap        интеграл f(x) = 2x^2 (функция f из файла пользователя) на [0, 5]
//   heavy        интеграл sin(x) * exp(-x) на [0, pi]: тяжёлая функция, много вычислений на точку
//   n1,n2,...    число отрезков разбиения (по умолчанию 1e3 ... 1e8, степени десятки)
//   reps         число замеров на точку   (по умолчанию 10)
//   max_threads  максимум потоков         (по умолчанию omp_get_max_threads())
//
// Файл пользователя содержит main(). Чтобы не было двух main, на время включения
// он переименован макросом. Более чистый вариант: вынести шаблоны в отдельный
// заголовок (.hpp) и подключать его и в рабочую программу, и сюда.

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

#ifndef INTEGRAL_SRC
#define INTEGRAL_SRC "integral.cpp"
#endif

// После переименования у main пропадает неявный "return 0", и компилятор предупреждает
// об отсутствии return. Эта функция нигде не вызывается, поэтому предупреждение глушим.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#define main integral_user_main
#include INTEGRAL_SRC
#undef main
#pragma GCC diagnostic pop

// ---------- подынтегральные функции и задачи ----------

constexpr double kPi = 3.14159265358979323846;

double heavy(double x) {
    return std::sin(x) * std::exp(-x);
}

struct Problem {
    double a;
    double b;
    double exact;  // точное значение интеграла для проверки
};

// ∫_0^5 2x^2 dx = 2 * 5^3 / 3
const Problem kCheap{0.0, 5.0, 2.0 * 5.0 * 5.0 * 5.0 / 3.0};
// ∫_0^pi sin(x) e^{-x} dx = (1 + e^{-pi}) / 2
const Problem kHeavy{0.0, kPi, (1.0 + std::exp(-kPi)) / 2.0};

// ---------- бенчмарк ----------

// Секунды по монотонным часам, разрешение в наносекунды
double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// |x - y| <= rel_tol * |y|
bool close(double x, double y, double rel_tol) {
    return std::abs(x - y) <= rel_tol * std::abs(y);
}

// Делает reps замеров и печатает строки CSV. Возвращает false, если результат неверный.
// Run: вызываемый объект без аргументов, возвращающий double.
template <typename Run>
bool bench(const char *variant, Run run, std::size_t n, int threads, int reps, double ref) {
    omp_set_num_threads(threads);

    run();  // прогрев: создание пула потоков (в CSV не пишем)

    for (int rep = 1; rep <= reps; rep++) {
        double start = now_seconds();
        double result = run();
        double end = now_seconds();  // печать только после остановки таймера

        // порядок суммирования в reduction другой, поэтому сравниваем с допуском, а не через ==
        if (!close(result, ref, 1e-8)) {
            std::cerr << "ОШИБКА: " << variant << ", n=" << n << ", потоков " << threads
                      << ": получили " << result << ", ожидали " << ref << '\n';
            return false;
        }
        std::cout << variant << "," << n << "," << threads << "," << rep << "," << (end - start)
                  << "\n";
    }
    return true;
}

// func передаётся шаблонным параметром: у каждой лямбды свой тип, поэтому компилятор
// встраивает вызов в цикл интегрирования (как и задумано при передаче функции шаблоном)
template <typename F>
int run_benchmark(F func, const Problem &p, const std::vector<std::size_t> &sizes, int reps,
                  int max_threads) {
    for (std::size_t n : sizes) {
        // эталон для проверки параллельной версии: последовательная версия на этом же n
        const double ref = integrate_single_thread(func, p.a, p.b, n);
        // проверка самой численной схемы: близость к точному значению интеграла
        if (!close(ref, p.exact, 1e-5)) {
            std::cerr << "ОШИБКА: при n=" << n << " результат " << ref
                      << " далёк от точного значения " << p.exact << '\n';
            return 1;
        }

        auto run_seq = [&]() { return integrate_single_thread(func, p.a, p.b, n); };
        auto run_par = [&]() { return integrate_multi_thread(func, p.a, p.b, n); };

        // база: последовательная версия (threads = 1)
        if (!bench("seq", run_seq, n, 1, reps, ref)) return 1;

        // параллельная версия для 1..max_threads потоков
        for (int t = 1; t <= max_threads; t++) {
            if (!bench("reduction", run_par, n, t, reps, ref)) return 1;
        }
        std::cerr << "n=" << n << " готово" << std::endl;
    }
    return 0;
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
    if (argc < 2) {
        std::cerr << "Использование: ./bench_integral <cheap|heavy> [n1,n2,...] [reps] "
                     "[max_threads]\n";
        return 2;
    }
    const std::string which = argv[1];
    std::vector<std::size_t> sizes =
        (argc > 2) ? parse_sizes(argv[2])
                   : std::vector<std::size_t>{1'000,     10'000,     100'000,
                                              1'000'000, 10'000'000, 100'000'000};
    const int reps = (argc > 3) ? std::atoi(argv[3]) : 10;
    const int max_threads = (argc > 4) ? std::atoi(argv[4]) : omp_get_max_threads();

    if ((which != "cheap" && which != "heavy") || sizes.empty() || reps < 1 || max_threads < 1) {
        std::cerr << "Использование: ./bench_integral <cheap|heavy> [n1,n2,...] [reps] "
                     "[max_threads]\n";
        return 2;
    }
    for (std::size_t n : sizes) {
        if (n == 0) {
            std::cerr << "Размер n должен быть положительным.\n";
            return 2;
        }
    }

    std::cerr << which << ": reps=" << reps << ", max_threads=" << max_threads
              << ", размеров: " << sizes.size() << std::endl;

    std::cout.precision(9);
    std::cout << std::fixed;
    std::cout << "variant,n,threads,rep,seconds\n";

    if (which == "cheap") {
        return run_benchmark([](double x) { return f(x); }, kCheap, sizes, reps, max_threads);
    }
    return run_benchmark([](double x) { return heavy(x); }, kHeavy, sizes, reps, max_threads);
}