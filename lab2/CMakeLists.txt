cmake_minimum_required(VERSION 3.24)
project(lab C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "")

add_executable(lab main2.c)

find_package(OpenMP REQUIRED)

target_compile_options(lab PRIVATE -Wall ${OpenMP_C_FLAGS})
target_link_libraries(lab PRIVATE ${OpenMP_C_FLAGS} m)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic -O2")
