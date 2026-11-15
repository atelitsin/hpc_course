#pragma once

#include <omp.h>

#include <algorithm>
#include <limits>

namespace hpc {

template <typename M>
typename M::value_type row_max_of_mins_seq(const M& m) {
    using T = typename M::value_type;
    T result = std::numeric_limits<T>::lowest();

    for (std::size_t i = 0; i < m.nrows(); ++i) {
        T rmin = std::numeric_limits<T>::max();
        const T* row = m.row_data(i);
        const std::size_t sz = m.row_size(i);
        for (std::size_t k = 0; k < sz; ++k) {
            rmin = std::min(rmin, row[k]);
        }
        result = std::max(result, rmin);
    }
    return result;
}

template <typename M>
typename M::value_type row_max_of_mins_omp(const M& m) {
    using T = typename M::value_type;
    T result = std::numeric_limits<T>::lowest();

#pragma omp parallel for schedule(runtime) reduction(max : result)
    for (std::size_t i = 0; i < m.nrows(); ++i) {
        T rmin = std::numeric_limits<T>::max();
        const T* row = m.row_data(i);
        const std::size_t sz = m.row_size(i);
        for (std::size_t k = 0; k < sz; ++k) {
            rmin = std::min(rmin, row[k]);
        }
        result = std::max(result, rmin);
    }
    return result;
}

}  // namespace hpc