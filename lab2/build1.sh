#!/bin/bash
set -x

gcc -std=gnu99 -O2 -Wpedantic -Wall -Werror -fopenmp -o main1 main1.c -lm
