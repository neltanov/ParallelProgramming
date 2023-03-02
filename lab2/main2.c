#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 18000
#define EPS 0.0000001
#define TAU 0.00001

void initMatrix(double *matrix, int size) {
    int line = 0;
    for (int i = 0; i < size * size; i++) {
        if (i == line * size + line) {
            matrix[i] = 2;
            line++;
        } else {
            matrix[i] = 1;
        }
    }
}

void initRightPart(double *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (double) size + 1;
    }
}

void printMatrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%0.3lf ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

void printArray(double *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%0.3lf ", array[i]);
    }
    printf("\n");
}

double euclideanNorm(const double *vec, int size) {
    double res = 0;
    for (int i = 0; i < size; i++)
        res += vec[i] * vec[i];
    return sqrt(res);
}

void mulMatVec(const double *matrix, const double *vec, int size, double *res) {
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        res[i] = 0;
        for (int j = 0; j < size; j++) {
            res[i] += matrix[i * size + j] * vec[j];
        }
    }
}

void subVectors(double *a, const double *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] = a[i] - b[i];
    }
}

void mulNumVec(double num, double *vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = num * vec[i];
    }
}

void singleIterate(double *A, double *x, double *b, int size) {
    double *res = (double *) malloc(size * sizeof(double));
    double criteria = 1;
    while (criteria >= EPS) {
        mulMatVec(A, x, size, res);
        subVectors(res, b, size);
        criteria = euclideanNorm(res, size) / euclideanNorm(b, size);
        printf("norm: %0.15lf\n", criteria);
        mulNumVec(TAU, res, size);
        subVectors(x, res, size);
    }

    free(res);
}

int main(void) {
    double time_begin = omp_get_wtime();
    int m_size = N;

    double *A = (double *) malloc(m_size * m_size * sizeof(double));
    initMatrix(A, m_size);
    double *b = (double *) malloc(m_size * sizeof(double));
    initRightPart(b, m_size);
    double *x = (double *) calloc(m_size, sizeof(double));

    singleIterate(A, x, b, m_size);

    printArray(x, m_size);

    free(A);
    free(b);
    free(x);

    double time_end = omp_get_wtime();
    printf("Time taken: %lf\n", time_end - time_begin);

    return 0;
}
