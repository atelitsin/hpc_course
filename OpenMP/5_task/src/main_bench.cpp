#include "matrix/banded.hpp"
#include "matrix/dense.hpp"
#include "matrix/row_max_of_mins.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace hpc;

struct Options {
    std::string output = "results/benchmark.csv";
    std::vector<std::size_t> sizes = {1000, 2000, 4000, 8000};
    std::vector<int> threads = {1, 2, 4, 6, 8};
    std::vector<int> chunks = {1, 16, 64};
    int repetitions = 7;
};

struct Schedule {
    const char* name;
    omp_sched_t kind;
};

const std::vector<Schedule> kSchedules = {
    {"static", omp_sched_static},
    {"dynamic", omp_sched_dynamic},
    {"guided", omp_sched_guided},
};

std::vector<std::size_t> parse_sizes(const char* text) {
    std::vector<std::size_t> result;
    const char* current = text;
    while (*current != '\0') {
        char* end = nullptr;
        const auto value = std::strtoull(current, &end, 10);
        if (end == current || value == 0) return {};
        result.push_back(static_cast<std::size_t>(value));
        if (*end == '\0') break;
        if (*end != ',') return {};
        current = end + 1;
    }
    return result;
}

std::vector<int> parse_ints(const char* text) {
    std::vector<int> result;
    const char* current = text;
    while (*current != '\0') {
        char* end = nullptr;
        const long value = std::strtol(current, &end, 10);
        if (end == current || value < 1 || value > 1000000) return {};
        result.push_back(static_cast<int>(value));
        if (*end == '\0') break;
        if (*end != ',') return {};
        current = end + 1;
    }
    return result;
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value_after = [&](const char* name) -> const char* {
            if (argument == name && i + 1 < argc) return argv[++i];
            return nullptr;
        };

        if (const char* value = value_after("--out")) {
            options.output = value;
        } else if (const char* value = value_after("--sizes")) {
            options.sizes = parse_sizes(value);
        } else if (const char* value = value_after("--threads")) {
            options.threads = parse_ints(value);
        } else if (const char* value = value_after("--chunks")) {
            options.chunks = parse_ints(value);
        } else if (const char* value = value_after("--reps")) {
            const auto parsed = parse_ints(value);
            if (parsed.size() != 1) return false;
            options.repetitions = parsed.front();
        } else {
            return false;
        }
    }
    return !options.output.empty() && !options.sizes.empty() && !options.threads.empty() &&
           !options.chunks.empty() && options.repetitions > 0;
}

template <typename F>
double measure(F&& function) {
    const auto start = std::chrono::steady_clock::now();
    const auto result = function();
    const auto finish = std::chrono::steady_clock::now();
    asm volatile("" : : "g"(result) : "memory");
    return std::chrono::duration<double>(finish - start).count();
}

template <typename M, typename Filler>
void fill_matrix(M& matrix, Filler filler) {
    filler(matrix);
}

template <typename M>
void write_measurements(std::FILE* output, const char* format, M& matrix,
                        const Options& options, const char* schedule_name, int chunk,
                        int threads, bool parallel) {
    omp_set_num_threads(threads);
    if (parallel) {
        const auto schedule = std::find_if(
            kSchedules.begin(), kSchedules.end(),
            [schedule_name](const Schedule& item) { return std::strcmp(item.name, schedule_name) == 0; });
        omp_set_schedule(schedule->kind, chunk);
    }

    const auto run = [&] {
        return parallel ? row_max_of_mins_omp(matrix) : row_max_of_mins_seq(matrix);
    };
    const auto expected = row_max_of_mins_seq(matrix);
    if (run() != expected) {
        std::fprintf(stderr, "incorrect result: format=%s n=%zu schedule=%s chunk=%d threads=%d\n",
                     format, matrix.nrows(), schedule_name, chunk, threads);
        std::exit(1);
    }

    run();
    for (int repetition = 1; repetition <= options.repetitions; ++repetition) {
        const double seconds = measure(run);
        if (run() != expected) {
            std::fprintf(stderr, "non-deterministic result: format=%s n=%zu\n", format,
                         matrix.nrows());
            std::exit(1);
        }
        std::fprintf(output, "%s,%zu,%d,%s,%d,%d,%.9f\n", format, matrix.nrows(), threads,
                     schedule_name, chunk, repetition, seconds);
    }
}

template <typename M, typename Filler>
void benchmark_matrix(std::FILE* output, const char* format, M& matrix, Filler filler,
                      const Options& options) {
    fill_matrix(matrix, filler);
    write_measurements(output, format, matrix, options, "sequential", 0, 1, false);

    for (const Schedule& schedule : kSchedules) {
        for (int chunk : options.chunks) {
            for (int threads : options.threads) {
                write_measurements(output, format, matrix, options, schedule.name, chunk,
                                   threads, true);
            }
        }
    }
}

void fill_matrix(DenseMatrix<double>& matrix, std::mt19937_64& generator) {
    std::uniform_real_distribution<double> distribution(-500.0, 500.0);
    for (std::size_t row = 0; row < matrix.nrows(); ++row) {
        for (std::size_t column = 0; column < matrix.ncols(); ++column) {
            matrix(row, column) = distribution(generator);
        }
    }
}

void fill_matrix(BandedMatrix<double>& matrix, std::mt19937_64& generator) {
    std::uniform_real_distribution<double> distribution(-500.0, 500.0);
    for (std::size_t row = 0; row < matrix.nrows(); ++row) {
        const std::size_t row_start = row < matrix.kl() ? 0 : row - matrix.kl();
        for (std::size_t column = 0; column < matrix.row_size(row); ++column) {
            matrix(row, row_start + column) =
                distribution(generator);
        }
    }
}

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        std::fprintf(stderr,
                     "usage: %s [--out FILE] [--sizes N1,N2] [--threads P1,P2] "
                     "[--chunks C1,C2] [--reps COUNT]\n",
                     argv[0]);
        return 2;
    }

    const auto parent = std::filesystem::path(options.output).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::FILE* output = std::fopen(options.output.c_str(), "w");
    if (!output) {
        std::perror("fopen");
        return 1;
    }

    std::fprintf(output, "format,n,threads,schedule,chunk,rep,seconds\n");
    for (std::size_t n : options.sizes) {
        std::mt19937_64 generator(42);
        DenseMatrix<double> dense(n);
        benchmark_matrix(output, "dense", dense,
                         [&generator](auto& matrix) { fill_matrix(matrix, generator); },
                         options);

        generator.seed(42);
        BandedMatrix<double> banded(n, 20, 20);
        benchmark_matrix(output, "banded_kl20", banded,
                         [&generator](auto& matrix) { fill_matrix(matrix, generator); },
                         options);
    }

    std::fclose(output);
    return 0;
}
