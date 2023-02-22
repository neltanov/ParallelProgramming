#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define EPS 0.000000000001
#define TAU 0.00001

#define ROOT 0

void initMatrix(double *matrix, int size) {
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

void initRightPart(double *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = size + 1;
    }
}

void printMatrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%0.3f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

void printArray(double *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%0.3f ", array[i]);
    }
    printf("\n");
}

double euclideanNorm(double *vec, int size) {
    double res = 0;
    for (int i = 0; i < size; i++)
        res += vec[i] * vec[i];
    return sqrt(res);
}

void mulMatVec(double *matrix, double *vec, int mx_size, double *res) {
    int nproc, rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rangeLength = mx_size / nproc;
    // Умножение соответствующих процессу строк на вектор.
    for (int i = rank * rangeLength; i < rank * rangeLength + rangeLength; i++) {
        for (int j = 0; j < mx_size; j++) {
            res[i] += matrix[i * mx_size + j] * vec[j];
        }
    }
    if (rank != ROOT) {
        MPI_Send(res + rank * rangeLength, rangeLength, MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD);
    }

    if (rank == ROOT) {
        for (int process = 1; process < nproc; process++) {
            MPI_Recv(res + process * rangeLength,
                     rangeLength, MPI_DOUBLE, process, 0, MPI_COMM_WORLD, &status);
        }
        // Перемножение оставшейся части матрицы на вектор.
        for (int i = mx_size - mx_size % nproc; i < mx_size; i++) {
            for (int j = 0; j < mx_size; j++) {
                res[i] += matrix[i + mx_size + j] * vec[j];
            }
        }
    }
    MPI_Bcast(res, mx_size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
}

void subVectors(double *a, double *b, int size) {
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
    double *res = (double *) calloc(size, sizeof(double));

    mulMatVec(A, x, size, res);
    subVectors(res, b, size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double criteria = 1;
    while (criteria >= EPS) {
        printf("Euclidean norm in process #%d: %0.16lf\n", rank, euclideanNorm(res, size) / euclideanNorm(b, size));
        mulMatVec(A, x, size, res);
        subVectors(res, b, size);
        criteria = euclideanNorm(res, size) / euclideanNorm(b, size);
        mulNumVec(TAU, res, size);
        subVectors(x,res, size);
    }
    free(res);
}

int main(int argc, char *argv[]) {
    int erCode = MPI_Init(NULL, NULL);
    if (erCode) {
        printf("Startup error, execution stopped\n");
        MPI_Abort(MPI_COMM_WORLD, erCode);
    }
    double start_time, end_time;
    start_time = MPI_Wtime();
//    int N = atoi(argv[1]);
//    double epsilon = strtod(argv[2], &end);

    int N = 26000;

    double *A = (double *) malloc(N * N * sizeof(double));
    initMatrix(A, N);
    double *b = (double *) malloc(N * sizeof(double));
    initRightPart(b, N);
    double *x = (double *) calloc(N, sizeof(double));

    singleIterate(A, x, b, N);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == ROOT) {
        printArray(x, N);
    }
    free(A);
    free(b);
    free(x);

    end_time = MPI_Wtime();
    printf("Time: %0.2lf\n", end_time - start_time);
    MPI_Finalize();
    return 0;
}
