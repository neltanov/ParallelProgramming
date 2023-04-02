#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

#define RANK_ROOT 0

void fill_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = j;
        }
    }
}

void print_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        printf("%lf ", matrix[i]);

        if ((i + 1) % cols == 0) {
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
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d); // создание 2d решетки
    MPI_Comm_rank(comm2d, &rank); // получение номеров процесса в 2d решетке (старая нумерация)
    MPI_Cart_get(comm2d, 2, dims, periods, coords); // получение номера процесса в виде координат решетки
    rankx = coords[0];
    ranky = coords[1];

    if (RANK_ROOT == rank) {
        printf("Size of 2d cart: %dx%d\n", sizex, sizey);
    }

    MPI_Comm commOrdinate, commAbscissa;
    MPI_Comm_split(comm2d, ranky, rankx, &commAbscissa);
    MPI_Comm_split(comm2d, rankx, ranky, &commOrdinate);

    int ord_size, ord_rank, abs_size, abs_rank;
    MPI_Comm_size(commOrdinate, &ord_size);
    MPI_Comm_size(commAbscissa, &abs_size);
    MPI_Comm_rank(commAbscissa, &abs_rank);
    MPI_Comm_rank(commOrdinate, &ord_rank);

    /* Вот такая решетка получается:
     * (0;1) (1;1) (2;1) (3;1) (4;1)
     * (0;0) (1;0) (2;0) (3;0) (4;0)
     * */

    int n1 = 11, n2 = 12, n3 = 15;
    double *A;
    double *B;

    double *part_A = calloc((n1 / sizey + 1) * n2, sizeof(double));
    double *part_B = calloc(n2 * (n3 / sizex + 1), sizeof(double));

    int *sendcounts = malloc(ord_size * sizeof(int));

    int *displs = malloc(ord_size * sizeof(int));

    if (RANK_ROOT == rank) {
        A = malloc(n1 * n2 * sizeof(double));
        B = malloc(n2 * n3 * sizeof(double));

        fill_matrix(A, n1, n2);
        fill_matrix(B, n2, n3);
    }
    // Подготовка sencounts и displs к подаче в scatterv
    int nmin = n1 / ord_size;
    int nextra = n1 % ord_size;
    int k = 0;
    for (int i = 0; i < ord_size; i++) {
        if (i < nextra) {
            sendcounts[i] = (nmin + 1) * n2;
        } else {
            sendcounts[i] = nmin * n2;
        }
        displs[i] = k;
        k += sendcounts[i];
    }

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, part_A, (n1 / ord_size + 1) * n2, MPI_DOUBLE, RANK_ROOT, commOrdinate);
    if (rankx == 0) {
        sleep(ranky);
        print_matrix(part_A, n1 / ord_size + 1, n2);
        printf("\n");
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

    if (RANK_ROOT == rank) {
        free(A);
        free(B);
    }
    free(part_A);
    free(part_B);

    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    const int exit_code = run();

    MPI_Finalize();

    return exit_code;
}
