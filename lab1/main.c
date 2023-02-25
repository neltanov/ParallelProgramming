#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define N 18000 // Number of equations in a system of linear equations.
#define EPSILON 0.0000000001 // Сonvergence criterion.
#define TAU 0.0001 // Parameter of the simple iteration method.
#define RANK_ROOT 0 // Number of root process.

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

void print_array(const double *array, int arr_size) {
    for (int i = 0; i < arr_size; i++) {
        if (i != arr_size - 1) {
            printf("%0.3f, ", array[i]);
        } else {
            printf("%0.3f", array[i]);
        }
    }
}

void print_solution(const double *solution_vec, int vec_size) {
    printf("Solution vector: (");
    print_array(solution_vec, vec_size);
    printf(")\n");
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
        result_vector[i] = 0;
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

void single_iterate_method(double *part_of_matrix, double *solution, double *b, int matrix_size) {
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *part_of_result_vector;
    if (matrix_size % size == 0) {
        part_of_result_vector = (double *) malloc(matrix_size / size * sizeof(double));
    } else {
        if (rank == size - 1) {
            part_of_result_vector = (double *) malloc((matrix_size / size + matrix_size % size) * sizeof(double));
        } else {
            part_of_result_vector = (double *) malloc(matrix_size / size * sizeof(double));
        }
    }

    double *result_vector = NULL;
    if (rank == RANK_ROOT) {
        result_vector = (double *) malloc(matrix_size * sizeof(double));
    }

    double criteria = 1; // Will be checked with epsilon.
    double euclidean_norm_b = euclidean_norm(b, matrix_size); // ||b||
    while (criteria >= EPSILON) {
        // Multiplying matrix with solution vector and sending result from all processes to root process.
        if (matrix_size % size == 0) {
            mul_mat_vec(part_of_matrix, solution, matrix_size / size,
                        matrix_size, part_of_result_vector);
            MPI_Gather(part_of_result_vector, matrix_size / size, MPI_DOUBLE,
                       result_vector, matrix_size / size, MPI_DOUBLE,
                       RANK_ROOT, MPI_COMM_WORLD);
        } else {
            MPI_Status status;
            if (rank != size - 1) {
                mul_mat_vec(part_of_matrix, solution, matrix_size / size,
                            matrix_size, part_of_result_vector);
            } else {
                mul_mat_vec(part_of_matrix, solution, matrix_size / size + matrix_size % size,
                            matrix_size, part_of_result_vector);
            }
            if (rank != RANK_ROOT && rank != size - 1) {
                MPI_Send(part_of_result_vector, matrix_size / size, MPI_DOUBLE, RANK_ROOT, 2, MPI_COMM_WORLD);
            } else if (rank == size - 1) {
                MPI_Send(part_of_result_vector, matrix_size / size + matrix_size % size, MPI_DOUBLE, RANK_ROOT, 3,
                         MPI_COMM_WORLD);
            }

            if (rank == RANK_ROOT) {
                for (int i = 0; i < matrix_size / size; i++) {
                    result_vector[i] = part_of_result_vector[i];
                }
                for (int i = 1; i < size - 1; i++) {
                    MPI_Recv(result_vector + i * (matrix_size / size), matrix_size / size, MPI_DOUBLE,
                             i, 2, MPI_COMM_WORLD, &status);
                }
                MPI_Recv(result_vector + (size - 1) * (matrix_size / size), matrix_size / size + matrix_size % size,
                         MPI_DOUBLE,
                         size - 1, 3, MPI_COMM_WORLD, &status);
            }
        }

        if (rank == RANK_ROOT) {
            sub_vectors(result_vector, b, matrix_size); // Ax - b
            criteria = euclidean_norm(result_vector, matrix_size) / euclidean_norm_b; // ||Ax - b|| / ||b||
            printf("Convergence criterion: %.16lf\n", criteria);
            mul_num_vec(TAU, result_vector, matrix_size); // t(Ax - b)
            sub_vectors(solution, result_vector, matrix_size); // x - t(Ax - b)
        }
        MPI_Bcast(&criteria, 1, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
        MPI_Bcast(solution, matrix_size, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);

        // Send result vector to all processes.
        if (matrix_size % size == 0) {
            MPI_Scatter(result_vector, matrix_size / size, MPI_DOUBLE,
                        part_of_result_vector, matrix_size / size, MPI_DOUBLE,
                        RANK_ROOT, MPI_COMM_WORLD);
        } else {
            MPI_Status status;
            if (rank == RANK_ROOT) {
                for (int i = 0; i < matrix_size / size; i++) {
                    part_of_result_vector[i] = result_vector[i];
                }
                for (int process = 1; process < size - 1; process++) {
                    MPI_Send(result_vector + process * (matrix_size / size), matrix_size / size, MPI_DOUBLE,
                             process, 4, MPI_COMM_WORLD);
                }
                MPI_Send(result_vector + (size - 1) * (matrix_size / size), matrix_size / size + matrix_size % size,
                         MPI_DOUBLE,
                         size - 1, 5, MPI_COMM_WORLD);
            }
            if (rank != RANK_ROOT && rank != size - 1) {
                MPI_Recv(part_of_result_vector, matrix_size / size, MPI_DOUBLE,
                         RANK_ROOT, 4, MPI_COMM_WORLD, &status);
            } else if (rank == size - 1) {
                MPI_Recv(part_of_result_vector, matrix_size / size + matrix_size % size, MPI_DOUBLE,
                         RANK_ROOT, 5, MPI_COMM_WORLD, &status);
            }
        }
    }
    free(part_of_result_vector);
    if (rank == RANK_ROOT) {
        free(result_vector);
    }
}

int main(void) {
    int error_code = MPI_Init(NULL, NULL);
    if (error_code) {
        printf("Startup error, execution stopped\n");
        MPI_Abort(MPI_COMM_WORLD, error_code);
    }

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start_time_s = 0, end_time_s;

    int matrix_size = N;

    double *right_part = (double *) malloc(matrix_size * sizeof(double)); // b
    init_right_part(right_part, matrix_size); // b_i = matrix_size + 1
    double *solution_vec = (double *) calloc(matrix_size, sizeof(double)); // x
    double *matrix = NULL; // A
    if (rank == RANK_ROOT) {
        start_time_s = MPI_Wtime();
        matrix = (double *) malloc(matrix_size * matrix_size * sizeof(double));
        init_matrix(matrix, matrix_size);
    }

    double *part_of_matrix; // A_i
    if (matrix_size % size == 0) {
        part_of_matrix = (double *) calloc(matrix_size * (matrix_size / size), sizeof(double));
    } else {
        if (rank == size - 1) {
            part_of_matrix = (double *) calloc(matrix_size * (matrix_size / size + matrix_size % size), sizeof(double));
        } else {
            part_of_matrix = (double *) calloc(matrix_size * (matrix_size / size), sizeof(double));
        }
    }
    // Dividing matrix to size parts.
    if (matrix_size % size == 0) {
        MPI_Scatter(matrix, matrix_size * (matrix_size / size), MPI_DOUBLE,
                    part_of_matrix, matrix_size * (matrix_size / size), MPI_DOUBLE,
                    RANK_ROOT, MPI_COMM_WORLD);
    } else {
        MPI_Status status;
        if (rank == RANK_ROOT) {
            for (int i = 0; i < matrix_size / size * matrix_size; i++) {
                part_of_matrix[i] = matrix[i];
            }
            for (int process = 1; process < size - 1; process++) {
                MPI_Send(matrix + process * (matrix_size / size) * matrix_size, matrix_size / size * matrix_size,
                         MPI_DOUBLE,
                         process, 0, MPI_COMM_WORLD);
            }
            MPI_Send(matrix + (size - 1) * (matrix_size / size) * matrix_size,
                     (matrix_size / size + matrix_size % size) * matrix_size,
                     MPI_DOUBLE,
                     size - 1, 1, MPI_COMM_WORLD);
        }
        if (rank != RANK_ROOT && rank != size - 1) {
            MPI_Recv(part_of_matrix, matrix_size / size * matrix_size, MPI_DOUBLE, RANK_ROOT, 0, MPI_COMM_WORLD,
                     &status);
        } else if (rank == size - 1) {
            MPI_Recv(part_of_matrix, (matrix_size / size + matrix_size % size) * matrix_size, MPI_DOUBLE, RANK_ROOT, 1,
                     MPI_COMM_WORLD,
                     &status);
        }
    }

    single_iterate_method(part_of_matrix, solution_vec, right_part, matrix_size);

    if (rank == RANK_ROOT) {
        print_solution(solution_vec, matrix_size);
        end_time_s = MPI_Wtime();
        printf("Time taken: %0.6lf s\n", end_time_s - start_time_s);
        free(matrix);
    }
    free(part_of_matrix);
    free(solution_vec);
    free(right_part);
    MPI_Finalize();
    return 0;
}
