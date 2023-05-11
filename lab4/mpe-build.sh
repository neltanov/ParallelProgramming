#!/bin/bash

mpecc -mpilog -O3 -Wpedantic -Wall -Werror -o jacobi_method jacobi_method.c -lm
