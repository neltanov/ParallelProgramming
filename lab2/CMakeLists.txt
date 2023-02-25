cmake_minimum_required(VERSION 3.24)
project(lab2 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "")

add_executable(lab2 main.c)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic -O2")

target_link_libraries(lab2 m)