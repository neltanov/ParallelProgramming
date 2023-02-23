#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define EPS 0.000001
#define TAU 0.01
#define ROOT 0

void init_matrix(double *matrix, int matrix_size) {
    int line = 0;
    for (int i = 0; i < matrix_size * matrix_size; i++) {
        if (i == line * matrix_size + line) {
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

void print_matrix(const double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%0.3lf ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void print_array(const double *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%0.3f ", array[i]);
    }
    printf("\n");
}

double euclidean_norm(const double *vec, int size) {
    double res = 0;
    for (int i = 0; i < size; i++)
        res += vec[i] * vec[i];
    return sqrt(res);
}

void mul_mat_vec(const double *matrix, const double *vector,
                 int rows, int cols, double *result_vector) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_vector[i] += matrix[i * rows + j] * vector[j];
        }
    }
}

void sub_vectors(double *a, double const *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] = a[i] - b[i];
    }
}

void mul_num_vec(double num, double *vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = num * vec[i];
    }
}

void single_iterate(double *part_of_matrix, double *solution, double *b, int m_size) {
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *part_of_result_vector = (double *) malloc((m_size / size) * sizeof(double));
    double *result_vector;
    if (rank == ROOT) {
        result_vector = (double *) malloc(m_size * sizeof(double));
    }

    double criteria = 1;
    double euclidean_norm_b = euclidean_norm(b, m_size);
    int i = 0; // for debugging
    while (criteria >= EPS) {
        printf("norm in process #%d: %.16lf\n", rank, criteria);
        mul_mat_vec(part_of_matrix, solution, m_size,
                    m_size / size, part_of_result_vector);
//        print_array(part_of_result_vector, m_size / size);

        MPI_Gather(part_of_result_vector, m_size / size, MPI_DOUBLE,
                   result_vector, m_size / size, MPI_DOUBLE,
                   ROOT, MPI_COMM_WORLD);

        if (rank == ROOT) {
            sub_vectors(result_vector, b, m_size);
            criteria = euclidean_norm(result_vector, m_size) / euclidean_norm_b;
//            printf("norm %lf\n", euclidean_norm(b, m_size));
            mul_num_vec(TAU, result_vector, m_size);
            sub_vectors(solution, result_vector, m_size);
        }
        MPI_Bcast(&criteria, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        MPI_Bcast(solution, m_size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        MPI_Scatter(result_vector, m_size / size, MPI_DOUBLE,
                    part_of_result_vector, m_size / size, MPI_DOUBLE,
                    ROOT, MPI_COMM_WORLD);

        // debug
//        if (i == 1) { break; }
//        i++;
    }
    if (rank == ROOT) {
        free(result_vector);
    }
    free(part_of_result_vector);
}

int main(void) {
    int erCode = MPI_Init(NULL, NULL);
    if (erCode) {
        printf("Startup error, execution stopped\n");
        MPI_Abort(MPI_COMM_WORLD, erCode);
    }

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start_time, end_time;
    int m_size = 8;
    double *part_of_matrix = (double *) calloc(m_size * (m_size / size), sizeof(double));
    double *right_part = (double *) malloc(m_size * sizeof(double));
    init_right_part(right_part, m_size);
    double *solution = (double *) calloc(m_size, sizeof(double));
    double *matrix;
    if (rank == ROOT) {
        start_time = MPI_Wtime();
        matrix = (double *) malloc(m_size * m_size * sizeof(double));
        init_matrix(matrix, m_size);
    }

    MPI_Scatter(matrix, m_size * (m_size / size), MPI_DOUBLE,
                part_of_matrix, m_size * (m_size / size), MPI_DOUBLE,
                ROOT, MPI_COMM_WORLD); // works!

    single_iterate(part_of_matrix, solution, right_part, m_size);

    if (rank == ROOT) {
//        print_array(solution, m_size);
        end_time = MPI_Wtime();
//        printf("Time: %0.6lf\n", end_time - start_time);
        free(matrix);
    }
    free(part_of_matrix);
    free(right_part);
    free(solution);
    MPI_Finalize();
    return 0;
}
