if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS /opt/homebrew/bin/g++-16)
    set(CMAKE_CXX_COMPILER /opt/homebrew/bin/g++-16)
endif()

cmake_minimum_required(VERSION 3.20)
project(hpc_task5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

add_executable(bench src/main_bench.cpp)
target_include_directories(bench PRIVATE include src)
target_link_libraries(bench PRIVATE OpenMP::OpenMP_CXX)