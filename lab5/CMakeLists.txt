cmake_minimum_required(VERSION 3.16)
project(lab5)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(SOURCES main.cpp)
#set(SOURCES main_no_balance.cpp)

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

add_executable(lab5 ${SOURCES})