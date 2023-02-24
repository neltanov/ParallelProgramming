#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define EPS 0.0000001
#define TAU 0.0001
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

void single_iterate(double *part_of_matrix, double *solution, double *b, int m_size) {
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *part_of_result_vector;
    if (m_size % size == 0) {
        part_of_result_vector = (double *) malloc(m_size / size * sizeof(double));
    }
    else {
        if (rank == size - 1) {
            part_of_result_vector = (double *) malloc((m_size / size + m_size % size) * sizeof(double));
        }
        else {
            part_of_result_vector = (double *) malloc(m_size / size * sizeof(double));
        }
    }

    double *result_vector = NULL;
    if (rank == ROOT) {
        result_vector = (double *) malloc(m_size * sizeof(double));
    }

    double criteria = 1; // Will be checked with epsilon.
    double euclidean_norm_b = euclidean_norm(b, m_size); // ||b||
    while (criteria >= EPS) {
        // Multiplying matrix with solution vector and sending result from all processes to root process.
        if (m_size % size == 0) {
            mul_mat_vec(part_of_matrix, solution, m_size / size,
                        m_size, part_of_result_vector);
            MPI_Gather(part_of_result_vector, m_size / size, MPI_DOUBLE,
                       result_vector, m_size / size, MPI_DOUBLE,
                       ROOT, MPI_COMM_WORLD);
        }
        else {
            MPI_Status status;
            mul_mat_vec(part_of_matrix, solution, m_size / size + m_size % size,
                        m_size, part_of_result_vector);

            if (rank != ROOT && rank != size - 1) {
                MPI_Send(part_of_result_vector, m_size / size, MPI_DOUBLE, ROOT, 2, MPI_COMM_WORLD);
            }
            else if (rank == size - 1){
                MPI_Send(part_of_result_vector, m_size / size + m_size % size, MPI_DOUBLE, ROOT, 3, MPI_COMM_WORLD);
            }

            if (rank == ROOT) {
                for (int i = 0; i < m_size / size; i++) {
                    result_vector[i] = part_of_result_vector[i];
                }
                for (int i = 1; i < size - 1; i++) {
                    MPI_Recv(result_vector + i * (m_size / size), m_size / size, MPI_DOUBLE,
                             i, 2, MPI_COMM_WORLD, &status);
                }
                MPI_Recv(result_vector + (size - 1) * (m_size / size), m_size / size + m_size % size, MPI_DOUBLE,
                         size - 1, 3, MPI_COMM_WORLD, &status);
            }
        }

        if (rank == ROOT) {
            sub_vectors(result_vector, b, m_size); // Ax - b
            criteria = euclidean_norm(result_vector, m_size) / euclidean_norm_b; // ||Ax - b|| / ||b||
            printf("Criteria: %.16lf\n", criteria);
            mul_num_vec(TAU, result_vector, m_size); // t(Ax - b)
            sub_vectors(solution, result_vector, m_size); // x - t(Ax - b)
        }
        MPI_Bcast(&criteria, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        MPI_Bcast(solution, m_size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

        // Send result vector to all processes.
        if (m_size % size == 0) {
            MPI_Scatter(result_vector, m_size / size, MPI_DOUBLE,
                        part_of_result_vector, m_size / size, MPI_DOUBLE,
                        ROOT, MPI_COMM_WORLD);
        }
        else {
            MPI_Status status;
            if (rank == ROOT) {
                for (int i = 0; i < m_size / size; i++) {
                    part_of_result_vector[i] = result_vector[i];
                }
                for (int process = 1; process < size - 1; process++) {
                    MPI_Send(result_vector + process * (m_size / size), m_size / size, MPI_DOUBLE,
                             process, 4, MPI_COMM_WORLD);
                }
                MPI_Send(result_vector + (size - 1) * (m_size / size), m_size / size + m_size % size, MPI_DOUBLE,
                         size - 1, 5, MPI_COMM_WORLD);
            }
            if (rank != ROOT && rank != size - 1) {
                MPI_Recv(part_of_result_vector, m_size / size, MPI_DOUBLE, ROOT, 4, MPI_COMM_WORLD, &status);
            }
            else if (rank == size - 1) {
                MPI_Recv(part_of_result_vector, m_size / size + m_size % size, MPI_DOUBLE, ROOT, 5, MPI_COMM_WORLD, &status);
            }
        }
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

    double start_time = 0, end_time;
    int m_size = 15;

    double *right_part = (double *) malloc(m_size * sizeof(double));
    init_right_part(right_part, m_size);
    double *solution = (double *) calloc(m_size, sizeof(double));
    double *matrix = NULL;
    if (rank == ROOT) {
        start_time = MPI_Wtime();
        matrix = (double *) malloc(m_size * m_size * sizeof(double));
        init_matrix(matrix, m_size);
        print_matrix(matrix, m_size, m_size);
    }

    double *part_of_matrix;
    if (m_size % size == 0) {
        part_of_matrix = (double *) calloc(m_size * (m_size / size), sizeof(double));
    }
    else {
        if (rank == size - 1) {
            part_of_matrix = (double *) calloc(m_size * (m_size / size + m_size % size), sizeof(double));
        }
        else {
            part_of_matrix = (double *) calloc(m_size * (m_size / size), sizeof(double));
        }
    }
    // Dividing matrix to size parts.
    if (m_size % size == 0) {
        MPI_Scatter(matrix, m_size * (m_size / size), MPI_DOUBLE,
                    part_of_matrix, m_size * (m_size / size), MPI_DOUBLE,
                    ROOT, MPI_COMM_WORLD);
    }
    else {
        MPI_Status status;
        if (rank == ROOT) {
            for (int i = 0; i < m_size / size * m_size; i++) {
                part_of_matrix[i] = matrix[i];
            }
            for (int process = 1; process < size - 1; process++) {
                MPI_Send(matrix + process * (m_size / size) * m_size, m_size / size * m_size, MPI_DOUBLE,
                         process, 0, MPI_COMM_WORLD);
            }
            MPI_Send(matrix + (size - 1) * (m_size / size) * m_size, (m_size / size + m_size % size) * m_size, MPI_DOUBLE,
                     size - 1, 1, MPI_COMM_WORLD);
        }
        if (rank != ROOT && rank != size - 1) {
            MPI_Recv(part_of_matrix, m_size / size * m_size, MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD, &status);
        }
        else if (rank == size - 1) {
            MPI_Recv(part_of_matrix, (m_size / size + m_size % size) * m_size, MPI_DOUBLE, ROOT, 1, MPI_COMM_WORLD, &status);
        }
    }

    single_iterate(part_of_matrix, solution, right_part, m_size);

    if (rank == ROOT) {
        print_array(solution, m_size);
        end_time = MPI_Wtime();
        printf("Time: %0.6lf\n", end_time - start_time);
        free(matrix);
    }
    free(part_of_matrix);
    free(right_part);
    free(solution);
    MPI_Finalize();
    return 0;
}
