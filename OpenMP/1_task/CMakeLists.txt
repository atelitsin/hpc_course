add_executable(task1_bench main.cpp)
target_include_directories(task1_bench PRIVATE include)
target_link_libraries(task1_bench PRIVATE OpenMP::OpenMP_CXX)
