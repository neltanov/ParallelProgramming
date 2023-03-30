#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

#define RANK_ROOT 0

typedef struct {
    double *data;
    int rows;
    int cols;
} Mat;

void mat_fill(Mat mat, int comm_size) {
    for (int y = 0; y < mat.rows; y += 1) {
        for (int x = 0; x < mat.cols; x += 1) {
            mat.data[y * mat.cols + x] = x;
        }
    }
}

void mat_print(Mat mat) {
    for (int i = 0; i < mat.rows * mat.cols; i += 1) {
        printf("%lf ", mat.data[i]);

        if ((i + 1) % mat.cols == 0) {
            printf("\n");
        }
    }
}

int run(void) {
    int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2], reorder = 1;
    int size, rank, sizey, sizex, ranky, rankx;
    int prevy, prevx, nexty, nextx;

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Dims_create(size, 2, dims);
    sizex = dims[0];
    sizey = dims[1];

    MPI_Comm comm2d;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d);
    MPI_Comm_rank(comm2d, &rank);
    MPI_Cart_get(comm2d, 2, dims, periods, coords);
    rankx = coords[0];
    ranky = coords[1];

    if (RANK_ROOT == rank) {
        printf("Size of 2d cart: (%d, %d)\n", sizex, sizey);
    }

    MPI_Comm commOrdinat, commAbsciss;
    MPI_Comm_split(comm2d, ranky, rankx, &commAbsciss);
    MPI_Comm_split(comm2d, rankx, ranky, &commOrdinat);

    int ord_size, ord_rank, abs_size, abs_rank;
    MPI_Comm_size(commOrdinat, &ord_size);
    MPI_Comm_size(commAbsciss, &abs_size);

    /*
     * (0;1) (1;1) (2;1) (3;1) (4;1)
     * (0;0) (1;0) (2;0) (3;0) (4;0)
     * */

    double *dataA = NULL;
    double *dataB = NULL;
    double *data = NULL;

    if (RANK_ROOT == rank) {
        int n1 = 10, n2 = 12, n3 = 15;

        dataA = (double *) calloc(n1 * n2, sizeof(*dataA));
        Mat A = {dataA, n1, n2};

        dataB = (double *) calloc(n2 * n3, sizeof(*dataB));
        Mat B = {dataB, n2, n3};

        mat_fill(A, size);
        mat_fill(B, size);

        mat_print(A);
        mat_print(B);
    }

//    const int N = 9;
//    const int columns_per_process = N / size;

//    data = (double *) calloc(N * columns_per_process, sizeof(*data));
//
//    MPI_Datatype vertical_int_slice;
//    MPI_Datatype vertical_int_slice_resized;
//    MPI_Type_vector(
//            /* blocks count - number of rows */ N,
//            /* block length  */ columns_per_process,
//            /* stride - block start offset */ N,
//            /* old type - element type */ MPI_DOUBLE,
//            /* new type */ &vertical_int_slice
//    );
//    MPI_Type_commit(&vertical_int_slice);
//
//    MPI_Type_create_resized(
//            vertical_int_slice,
//            /* lower bound */ 0,
//            /* extent - size in bytes */ (int) (columns_per_process * sizeof(*data)),
//            /* new type */ &vertical_int_slice_resized
//    );
//    MPI_Type_commit(&vertical_int_slice_resized);
//
//    MPI_Scatter(
//            /* send buffer */ dataA,
//            /* number of <send data type> elements sent */ 1,
//            /* send data type */ vertical_int_slice_resized,
//            /* recv buffer */ data,
//            /* number of <recv data type> elements received */ N * columns_per_process,
//            /* recv data type */ MPI_DOUBLE,
//                              RANK_ROOT,
//                              MPI_COMM_WORLD
//    );
//
//    sleep(1 + rank);
//    printf("RANK %d:\n", rank);
//    mat_print((Mat) {data, .rows = N, .cols = columns_per_process});
//
//    MPI_Type_free(&vertical_int_slice_resized);
//    MPI_Type_free(&vertical_int_slice);
//    free(data);
    free(dataA);
    free(dataB);
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    const int exit_code = run();

    MPI_Finalize();

    return exit_code;
}
