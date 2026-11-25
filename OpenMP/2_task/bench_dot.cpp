#include "benchmark.hpp"
#include "scalar_product.hpp"

int main(int argc, char *argv[]) {
    std::vector<std::size_t> sizes =
        (argc > 1) ? parse_sizes(argv[1])
                   : std::vector<std::size_t>{1'000,     10'000,     100'000,
                                              1'000'000, 10'000'000, 100'000'000};
    int reps = (argc > 2) ? std::atoi(argv[2]) : 10;
    int max_threads = (argc > 3) ? std::atoi(argv[3]) : omp_get_max_threads();

    if (sizes.empty() || reps < 1 || max_threads < 1) {
        std::cerr << "Использование: ./task2_bench [n1,n2,...] [reps] [max_threads]" << std::endl;
        return 2;
    }

    std::cerr << "reps=" << reps << ", max_threads=" << max_threads
              << ", размеров: " << sizes.size() << std::endl;

    omp_set_dynamic(0);
    std::cout.precision(12);
    std::cout << std::fixed;
    std::cout << "variant,n,threads,rep,seconds,calls\n";

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
