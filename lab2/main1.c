#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 20000
#define EPS 0.0000001
#define TAU 0.00001

void init_matrix(double *matrix, int size) {
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

void init_right_part(double *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (double) size + 1;
    }
}

void print_matrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%0.3lf ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

void print_array(double *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%0.3lf ", array[i]);
    }
    printf("\n");
}

double euclidean_norm(const double *vec, int size) {
    double res = 0;
    #pragma omp parallel for reduction(+:res)
    for (int i = 0; i < size; i++)
        res += vec[i] * vec[i];
    return sqrt(res);
}

void mul_mat_vec(const double *matrix, const double *vec, int size, double *res) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        res[i] = 0;
        for (int j = 0; j < size; j++) {
            res[i] += matrix[i * size + j] * vec[j];
        }
    }
}

void sub_vectors(double *a, const double *b, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        a[i] = a[i] - b[i];
    }
}

void mul_num_vec(double num, double *vec, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        vec[i] = num * vec[i];
    }
}

void single_iterate(double *A, double *x, double *b, int size) {
    double *res = (double *) malloc(size * sizeof(double));
    double criteria = 1;
    double euclidean_norm_b = euclidean_norm(b, size);

    while (criteria >= EPS) {
        mul_mat_vec(A, x, size, res);
        sub_vectors(res, b, size);
        criteria = euclidean_norm(res, size) / euclidean_norm_b;
        printf("norm: %0.15lf\n", criteria);
        mul_num_vec(TAU, res, size);
        sub_vectors(x, res, size);
    }

    free(res);
}

int main(void) {
    double time_begin = omp_get_wtime();
    int m_size = N;
    double *A = (double *) malloc(m_size * m_size * sizeof(double));
    init_matrix(A, m_size);
    double *b = (double *) malloc(m_size * sizeof(double));
    init_right_part(b, m_size);
    double *x = (double *) calloc(m_size, sizeof(double));

    single_iterate(A, x, b, m_size);
    print_array(x, m_size);

    free(A);
    free(b);
    free(x);

    double time_end = omp_get_wtime();
    printf("Time taken: %lf\n", time_end - time_begin);

    return 0;
}
