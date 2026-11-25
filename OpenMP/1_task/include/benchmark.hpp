#pragma once

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <omp.h>

using MaxFn = int (*)(const std::vector<int> &);

inline std::vector<int> generate_vector(std::size_t length, int min_val, int max_val) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> distrib(min_val, max_val);

    std::vector<int> result(length);
    for (std::size_t i = 0; i < length; i++) {
        result[i] = distrib(gen);
    }
    return result;
}

inline bool bench(const char *variant, MaxFn fn, const std::vector<int> &a,
           int threads, int reps, int ref) {
    const std::size_t n = a.size();
    omp_set_num_threads(threads);

    using Clock = std::chrono::steady_clock;
    constexpr double minimum_seconds = 0.002;
    // Volatile сохраняет отдельный вызов даже для одинаковых входных данных.
    MaxFn volatile run = fn;
    run(a);

    std::size_t batch = 1;
    for (int rep = 1; rep <= reps; ++rep) {
        double elapsed;
        for (;;) {
            const auto start = Clock::now();
            for (std::size_t call = 0; call < batch; ++call) {
                const int result = run(a);
                if (result != ref) {
                    std::cerr << "Ошибка: " << variant << ", n=" << n
                              << ", потоков " << threads << ": получили " << result
                              << ", ожидали " << ref << '\n';
                    return false;
                }
            }
            elapsed = std::chrono::duration<double>(Clock::now() - start).count();
            if (elapsed >= minimum_seconds) break;
            batch *= 2;
        }
        std::cout << variant << "," << n << "," << threads << "," << rep << ","
                  << elapsed / static_cast<double>(batch) << "," << batch << "\n";
    }
    return true;
}

inline std::vector<std::size_t> parse_sizes(const char* input) {
    std::vector<std::size_t> sizes;
    const char* current = input;
    while (*current >= '0' && *current <= '9') {
        char* end = nullptr;
        const auto size = std::strtoull(current, &end, 10);
        if (size == 0 || size > std::numeric_limits<std::size_t>::max()) return {};
        sizes.push_back(static_cast<std::size_t>(size));
        if (*end == '\0') return sizes;
        if (*end != ',') return {};
        current = end + 1;
    }
    return {};
}
