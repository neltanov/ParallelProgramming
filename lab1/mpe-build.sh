#!/bin/bash
set -x

mpecc -mpilog -std=gnu99 -O2 -Wpedantic -Wall -Werror -o mpe_main main.c -lm
