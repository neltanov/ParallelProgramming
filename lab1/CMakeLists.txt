cmake_minimum_required(VERSION 3.24)
project(lab1 C)

set(CMAKE_C_STANDARD 11)

include_directories(/usr/include/x86_64-linux-gnu/mpich)
set(CMAKE_C_COMPILER "/usr/bin/mpicc")
set(CMAKE_CXX_COMPILER "/usr/bin/mpic++")


add_executable(lab1 main.c)
