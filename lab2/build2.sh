#!/bin/bash
set -x

gcc -std=gnu99 -O2 -Wpedantic -Wall -Werror -fopenmp -o main2 main2.c -lm
