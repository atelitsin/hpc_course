#pragma once

#include <cstddef>
#include <limits>
#include <vector>

inline int vector_max_parallel_reduction(const std::vector<int>& vec) {
    int found_max = std::numeric_limits<int>::lowest();
#pragma omp parallel for reduction(max : found_max)
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (vec[i] > found_max) {
            found_max = vec[i];
        }
    }
    return found_max;
}

inline int vector_max_parallel(const std::vector<int>& vec) {
    int found_max = std::numeric_limits<int>::lowest();

#pragma omp parallel
    {
        int local_max = std::numeric_limits<int>::lowest();

#pragma omp for
        for (std::size_t i = 0; i < vec.size(); ++i) {
            if (vec[i] > local_max) {
                local_max = vec[i];
            }
        }

#pragma omp critical
        {
            if (local_max > found_max) {
                found_max = local_max;
            }
        }
    }

    return found_max;
}

inline int vector_max_sequential(const std::vector<int>& vec) {
    int found_max = std::numeric_limits<int>::lowest();
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (vec[i] > found_max) {
            found_max = vec[i];
        }
    }
    return found_max;
}
