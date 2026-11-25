add_executable(task2 main.cpp)
target_link_libraries(task2 PRIVATE OpenMP::OpenMP_CXX)

add_executable(task2_bench bench_dot.cpp)
target_include_directories(task2_bench PRIVATE include)
target_link_libraries(task2_bench PRIVATE OpenMP::OpenMP_CXX)
