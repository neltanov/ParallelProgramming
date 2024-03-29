#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

#define RANK_ROOT 0
#define LOWER_BOUND 0

#define N1 500
#define N2 500
#define N3 500

void fill_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = j;
        }
    }
}

void print_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        printf("%0.2lf ", matrix[i]);
        if ((i + 1) % cols == 0) {
            printf("\n");
        }
    }
}

void multiply_matrices(const double *A, const double *B, double *C, int n1, int n2, int n3) {
    for (int i = 0; i < n1; i++) {
        for (int k = 0; k < n2; k++) {
            for (int j = 0; j < n3; j++) {
                C[i * n3 + j] += A[i * n2 + k] * B[k * n3 + j];
            }
        }
    }
}

int run(void) {
    int size, rank, sizey, sizex, ranky, rankx;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n1 = N1, n2 = N2, n3 = N3;

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    if (RANK_ROOT == rank) {
        A = calloc(n1 * n2, sizeof(double));
        B = calloc(n2 * n3, sizeof(double));
        C = calloc(n1 * n3, sizeof(double));

        fill_matrix(A, n1, n2);
        fill_matrix(B, n2, n3);
//        print_matrix(A, n1, n2);
//        printf("\n");
//        print_matrix(B, n2, n3);
//        printf("\n");
    }

    const double start_time_s = MPI_Wtime();

    int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2], reorder = 1;

    MPI_Dims_create(size, 2, dims);

    MPI_Comm comm2d;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d);
    MPI_Cart_get(comm2d, 2, dims, periods, coords);

    sizex = dims[0];
    sizey = dims[1];

    rankx = coords[0];
    ranky = coords[1];

    MPI_Comm commOrdinate, commAbscissa;
    MPI_Comm_split(comm2d, ranky, rankx, &commAbscissa);
    MPI_Comm_split(comm2d, rankx, ranky, &commOrdinate);

    if (n1 % sizey != 0 || n3 % sizex != 0) {
        printf("%d mod %d != 0", n3, sizex);
        return 1;
    }

    const int rows_per_process = n1 / sizey;
    const int columns_per_process = n3 / sizex;

    double *part_A = calloc(rows_per_process * n2, sizeof(double));
    double *part_B = calloc(n2 * columns_per_process, sizeof(double));
    double *part_C = calloc(rows_per_process * columns_per_process, sizeof(double));

    if (RANK_ROOT == rankx) {
        MPI_Scatter(A, rows_per_process * n2, MPI_DOUBLE,
                    part_A, rows_per_process * n2, MPI_DOUBLE, RANK_ROOT, commOrdinate);
    }
    MPI_Bcast(part_A, rows_per_process * n2, MPI_DOUBLE, RANK_ROOT, commAbscissa);

    MPI_Datatype vertical_slice;
    MPI_Type_vector(n2, columns_per_process, n3, MPI_DOUBLE, &vertical_slice);
    MPI_Type_commit(&vertical_slice);

    MPI_Datatype vertical_slice_resized;
    MPI_Type_create_resized(vertical_slice, LOWER_BOUND,
                            (int) (columns_per_process * sizeof(double)), &vertical_slice_resized);
    MPI_Type_commit(&vertical_slice_resized);

    if (RANK_ROOT == ranky) {
        MPI_Scatter(B, 1, vertical_slice_resized,
                    part_B, n2 * columns_per_process, MPI_DOUBLE, RANK_ROOT, commAbscissa);
    }
    MPI_Bcast(part_B, n2 * columns_per_process, MPI_DOUBLE, RANK_ROOT, commOrdinate);

    multiply_matrices(part_A, part_B, part_C, rows_per_process, n2, columns_per_process);

    MPI_Datatype matrix_block;
    MPI_Type_vector(rows_per_process, columns_per_process, n3, MPI_DOUBLE, &matrix_block);
    MPI_Type_commit(&matrix_block);

    MPI_Datatype matrix_block_resized;
    MPI_Type_create_resized(matrix_block, LOWER_BOUND,
                            (int) (columns_per_process * sizeof(double)), &matrix_block_resized);
    MPI_Type_commit(&matrix_block_resized);

    int *sendcounts = NULL;
    int *displs = NULL;

    if (RANK_ROOT == rank) {
        sendcounts = malloc(sizex * sizey * sizeof(int));
        displs = malloc(sizex * sizey * sizeof(int));

        for (int i = 0; i < sizex * sizey; i++) {
            sendcounts[i] = 1;
        }
        for (int i = 0; i < sizey; i++) {
            for (int j = 0; j < sizex; j++) {
                displs[j * sizex + i] = sizex * i * rows_per_process + j;
            }
        }
    }

    MPI_Gatherv(part_C, rows_per_process * columns_per_process, MPI_DOUBLE,
                C, sendcounts, displs, matrix_block_resized, RANK_ROOT, comm2d);

    MPI_Type_free(&matrix_block);
    MPI_Type_free(&matrix_block_resized);

    MPI_Type_free(&vertical_slice_resized);
    MPI_Type_free(&vertical_slice);

    if (RANK_ROOT == rank) {
//        print_matrix(C, n1, n3);
        free(A);
        free(B);
        free(C);
        free(sendcounts);
        free(displs);
    }

    free(part_A);
    free(part_B);
    free(part_C);

    const double end_time_s = MPI_Wtime();
    if (RANK_ROOT == rank) {
        printf("Time taken: %lf sec\n", end_time_s - start_time_s);
    }

    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    const int exit_code = run();

    MPI_Finalize();

    return exit_code;
}
