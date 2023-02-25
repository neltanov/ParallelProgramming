cmake_minimum_required(VERSION 3.5)
project(lab1 C)

set(CMAKE_C_STANDARD 99)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(lab1 main.c)
target_link_libraries(lab1 ${MPI_LIBRARIES} m)
