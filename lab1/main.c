#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void initMatrix(double *matrix, size_t size) {
    size_t line = 1;
    for (size_t i = 0; i < size * size; i++) {
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

void initRightPart(double *array, size_t size) {
    for (size_t i = 0; i < size; i++) {
        array[i] = (double) size + 1;
    }
}

void initSolution(double *array, size_t size) {
    for (size_t i = 0; i < size; i++) {
        array[i] = 0;
    }
}

void printMatrix(double *matrix, size_t size) {
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            printf("%0.3f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

void printArray(double *array, size_t size) {
    for (size_t i = 0; i < size; i++) {
        printf("%0.3f ", array[i]);
    }
    printf("\n");
}

double sqrt(double n) {
    const double epsilon = 0.000001;
    double sqrt = 0, root = 0;
    while (sqrt < n) {
        root += epsilon;
        sqrt = root * root;
    }
    return sqrt;
}

double euclideanNorm(double *vec, size_t size) {
    double res = 0;
    for (size_t i = 0; i < size; i++)
        res += vec[i] * vec[i];
    return sqrt(res);
}

void mulMatVec(double *matrix, double *vec, size_t size, double *res) {
    for (size_t i = 0; i < size; i++)
        for (size_t j = 0; j < size; j++)
            res[i] += matrix[i * size + j] * vec[j];
}

void subVectors(double *a, double *b, size_t size) {
    for (size_t i = 0; i < size; i++) {
        a[i] = a[i] - b[i];
    }
}

void mulNumVec(double num, double *vec, size_t size) {
    for (size_t i = 0; i < size; i++) {
        vec[i] = num * vec[i];
    }
}

void singleIterate(double *A, double *x, double *b, size_t size, double epsilon) {
    const double tau = 0.01;
    double *res = (double *) malloc(size * sizeof(double));

    mulMatVec(A, x, size, res);
    subVectors(res, b, size);

    while (euclideanNorm(res, size) / euclideanNorm(b, size) >= epsilon) {
        mulMatVec(A, x, size, res);
        subVectors(res, b, size);
        mulNumVec(tau, res, size);
        subVectors(x, res, size);
    }
    free(res);
}

int main(int argc, char *argv[]) {
//    char *end;
//    size_t N = strtoul(argv[1], &end, 10);
//    double epsilon = strtod(argv[2], &end);

    size_t N = 10;
    double epsilon = 0.0000001;
    double *A = (double *) malloc(N * N * sizeof(double));
    initMatrix(A, N);
    double *b = (double *) malloc(N * sizeof(double));
    initRightPart(b, N);
    double *x = (double *) malloc(N * sizeof(double));
    initSolution(x, N);

    singleIterate(A, x, b, N, epsilon);

    printArray(x, N);

    // Распараллеливание
    int size, rank;
    double starttime, endtime;
    int erCode = MPI_Init(&argc, &argv);
    if (erCode) {
        printf("Startup error, execution stopped\n");
        MPI_Abort(MPI_COMM_WORLD, erCode);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        starttime = MPI_Wtime();
    }

    MPI_Bcast(A, (int) (N*N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    printMatrix(A, N);
    double *newA = (double *) malloc(N / size * N * sizeof(double));
    size_t k = 0;
    for (size_t i = rank; i < N; i += size) {
        for (size_t j = 0; j < N; j++) {
            newA[k * N + j] = A[i * N + j];
            k++;
        }
    }
    for (size_t i = 0; i < N / size; i++) {
        for (size_t j = 0; j < N; j++) {
            printf("%0.3lf ", newA[i * N + j]);
        }
        printf("\n");
    }

    if (rank == 0) {
//        printArray(x, N);
        endtime = MPI_Wtime();
        printf("Time taken: %lf", (endtime - starttime));
    }

    MPI_Finalize();

    free(A);
    free(b);
    free(x);
    return 0;
}