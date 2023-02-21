#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define EPS 0.000001
#define TAU 0.01
#define ROOT 0

void init_matrix(double *matrix, int size) {
    int line = 1;
    for (int i = 0; i < size * size; i++) {
        if (i == 0) {
            matrix[i] = 2;
        } else if (i % (line * size + line) == 0) {
            matrix[i] = 2;
            line++;
        } else {
            matrix[i] = 1;
        }
    }
}

void init_right_part(double *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = size + 1;
    }
}

void print_matrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%0.3f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

void print_solution(double *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%0.3f ", array[i]);
    }
    printf("\n");
}

double euclidean_norm(double *vec, int size) {
    double res = 0;
    for (int i = 0; i < size; i++)
        res += vec[i] * vec[i];
    return sqrt(res);
}

void mul_mat_vec(double *matrix, double *vector, int rows, int cols, double *result_vector) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_vector[i] += matrix[i * rows + j] * vector[j];
        }
    }
}

void sub_vectors(double *a, double *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] = a[i] - b[i];
    }
}

void mul_num_vec(double num, double *vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = num * vec[i];
    }
}

void single_iterate(double *part_of_matrix, double *solution, double *b, int vec_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &size);

    double *part_of_result_vector = (double *) calloc(vec_size / size, sizeof(double));
    double *result_vector;

    if (rank == ROOT) {
        result_vector = (double *) malloc(vec_size * sizeof(double));
    }

    mul_mat_vec(part_of_matrix, solution, vec_size,
                vec_size / size, part_of_result_vector);
    MPI_Gather(part_of_result_vector, vec_size * vec_size / size, MPI_DOUBLE,
               result_vector, vec_size * vec_size / size, MPI_DOUBLE,
               ROOT, MPI_COMM_WORLD);
    if (rank == ROOT) {
        sub_vectors(result_vector, b, vec_size);
    }

    double criteria;
    while (criteria >= EPS) {
        mul_mat_vec(part_of_matrix, solution, vec_size,
                    vec_size / size, part_of_result_vector);
        MPI_Gather(part_of_result_vector, vec_size * vec_size / size, MPI_DOUBLE,
                   result_vector, vec_size * vec_size / size, MPI_DOUBLE,
                   ROOT, MPI_COMM_WORLD);
        if (rank == ROOT) {
            sub_vectors(result_vector, b, vec_size);
            mul_num_vec(TAU, result_vector, vec_size);
            sub_vectors(solution, result_vector, vec_size);
            criteria = euclidean_norm(result_vector, vec_size) / euclidean_norm(b, vec_size);
        }
        MPI_Bcast(&criteria, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    }
    free(result_vector);
    free(part_of_result_vector);
}

int main(void) {
    int erCode = MPI_Init(NULL, NULL);
    if (erCode) {
        printf("Startup error, execution stopped\n");
        MPI_Abort(MPI_COMM_WORLD, erCode);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;
    int m_size = 10;

    double *part_of_matrix = (double *) malloc(m_size * m_size / size * sizeof(double));
    double *right_part = (double *) malloc(m_size * sizeof(double));
    init_right_part(right_part, m_size);
    double *solution = (double *) calloc(m_size, sizeof(double));
    double *matrix;
    if (rank == ROOT) {
        start_time = MPI_Wtime();
        matrix = (double *) malloc(m_size * m_size * sizeof(double));
        init_matrix(matrix, m_size);
    }

    MPI_Scatter(matrix, m_size * m_size / size, MPI_DOUBLE,
                part_of_matrix, m_size * m_size / size, MPI_DOUBLE,
                ROOT, MPI_COMM_WORLD);

    single_iterate(part_of_matrix, solution, right_part, m_size);

    if (rank == ROOT) {
        print_solution(solution, m_size);
        end_time = MPI_Wtime();
        printf("Time: %0.6lf\n", end_time - start_time);
    }
    MPI_Finalize();
    free(part_of_matrix);
    free(matrix);
    free(right_part);
    free(solution);
    return 0;
}
