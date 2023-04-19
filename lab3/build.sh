#!/bin/bash
set -x

mpicc -std=gnu99 -O2 -Wpedantic -Wall -Werror -o matrix_multiplying matrix_multiplying.c
