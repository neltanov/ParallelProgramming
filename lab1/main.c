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
    int nproc, rank;
    double starttime, endtime;

    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rangeLength = (int) size / nproc;

    // Отправка длин диапазонов каждому процессу из корневого.
    if (rank == 0) {
//        starttime = MPI_Wtime();
        if (size % nproc == 0) {
            for (int i = 0; i < nproc; i++) {
                MPI_Send(&rangeLength, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            }
        }
        else {
            for (int i = 0; i < nproc - 1; i++) {
                MPI_Send(&rangeLength, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            }
            rangeLength += (int) size - rangeLength * nproc;
            MPI_Send(&rangeLength, 1, MPI_INT, nproc - 1, 1, MPI_COMM_WORLD);
        }
    }
    // Переменная для записи состояния
    MPI_Status status;

    // Получение длины диапозона из корневого процесса.
    MPI_Recv(&rangeLength, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

    // Буфер для вычисленных значений умножения строк на вектор.
    double *buf = (double *) malloc(rangeLength * sizeof(double));
    // Умножение соответствующих процессу стро к на вектор.
    for (size_t i = 0; i < rangeLength; i++) {
        for (size_t j = 0; j < size; j++) {
            buf[i] += matrix[i * size + j] * vec[j];
        }
    }

    MPI_Send(buf, rangeLength, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

    if (rank == 0) {
//        MPI_Aint *extent = (MPI_Aint *) 8;
//        MPI_Aint *lb;
        rangeLength = (int) size / nproc;
        if (size % nproc == 0) {
            for (int i = 0; i < nproc; i++) {
                MPI_Recv(res + i * (int) size / nproc * sizeof(double),
                         rangeLength, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
            }
        }
        else {
            for (int i = 0; i < nproc - 1; i++) {
                MPI_Recv(res + i * (int) size / nproc * sizeof(double),
                         rangeLength, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
            }
            rangeLength = (int) size / nproc + (int) size - rangeLength * nproc;
            MPI_Recv(res + (nproc - 1) * (int) size / nproc * sizeof(double),
                     rangeLength, MPI_DOUBLE, nproc - 1, 2, MPI_COMM_WORLD, &status);
        }
//        endtime = MPI_Wtime();
//        printf("Time taken: %lf", (endtime - starttime));
    }

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

    int erCode = MPI_Init(NULL, NULL);
    if (erCode) {
        printf("Startup error, execution stopped\n");
        MPI_Abort(MPI_COMM_WORLD, erCode);
    }

    mulMatVec(A, x, size, res);
    subVectors(res, b, size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    while (euclideanNorm(res, size) / euclideanNorm(b, size) >= epsilon) {
        printf("norm in process #%d = %lf\n", rank, euclideanNorm(res, size) / euclideanNorm(b, size));
        mulMatVec(A, x, size, res);
        subVectors(res, b, size);
        mulNumVec(tau, res, size);
        subVectors(x,res, size);
    }
    printf("success! process #%d\n", rank);

    free(res);
}

int main(int argc, char *argv[]) {
//    char *end;
//    size_t N = strtoul(argv[1], &end, 10);
//    double epsilon = strtod(argv[2], &end);

    size_t N = 10;
    double epsilon = 0.000001;
    double *A = (double *) malloc(N * N * sizeof(double));
    initMatrix(A, N);
    double *b = (double *) malloc(N * sizeof(double));
    initRightPart(b, N);
    double *x = (double *) malloc(N * sizeof(double));
    initSolution(x, N);

    singleIterate(A, x, b, N, epsilon);

    printArray(x, N);

    free(A);
    free(b);
    free(x);
    MPI_Finalize();
    return 0;
}