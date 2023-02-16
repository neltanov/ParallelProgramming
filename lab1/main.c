#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define EPS 0.0000000001
#define TAU 0.01

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

void initSolution(double *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = 0;
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

double sqrt(double n) {
    const double epsilon = 0.000001;
    double sqrt = 0, root = 0;
    while (sqrt < n) {
        root += epsilon;
        sqrt = root * root;
    }
    return sqrt;
}

double euclideanNorm(double *vec, int size) {
    double res = 0;
    for (int i = 0; i < size; i++)
        res += vec[i] * vec[i];
    return sqrt(res);
}

void mulMatVec(double *matrix, double *vec, int size, double *res) {
    int nproc, rank;
//    double starttime, endtime;

    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rangeLength = size / nproc;
    printf("process #%d\n", rank);
    // Отправка длин диапазонов каждому процессу из корневого.
    if (rank == 0) {
//        starttime = MPI_Wtime();
        if (size % nproc == 0) {
            for (int i = 0; i < nproc; i++) {
                printf("Problem!!! %d", i);
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
        printf("Data has sent to all processes\n");
    }
    // Переменная для записи состояния
    MPI_Status status;

    // Получение длины диапозона из корневого процесса.
    MPI_Recv(&rangeLength, 1, MPI_INT, ROOT, 1, MPI_COMM_WORLD, &status);
    printf("Data has recieved from root process to process #%d. RangeLength = %d\n", rank, rangeLength);
    printMatrix(matrix, size);
    printArray(vec, size);
    // Буфер для вычисленных значений умножения строк на вектор.
    double *buf = (double *) malloc(rangeLength * sizeof(double));
    // Умножение соответствующих процессу строк на вектор.
    for (int i = 0; i < rangeLength; i++) {
        for (int j = 0; j < size; j++) {
            buf[i] += matrix[i * size + j] * vec[j];
        }
    }
    printArray(buf, size);
    MPI_Send(buf, rangeLength, MPI_DOUBLE, ROOT, 2, MPI_COMM_WORLD);
    printf("Data has sent to root process from process #%d\n", rank);

    if (rank == 0) {
        rangeLength = size / nproc;
        if (size % nproc == 0) {
            for (int process = 0; process < nproc; process++) {
                MPI_Recv(res + process * size / nproc,
                         rangeLength, MPI_DOUBLE, process, 2, MPI_COMM_WORLD, &status);
            }
        }
        else {
            for (int process = 0; process < nproc - 1; process++) {
                MPI_Recv(res + process * size / nproc,
                         rangeLength, MPI_DOUBLE, process, 2, MPI_COMM_WORLD, &status);
            }
            rangeLength += (int) size - rangeLength * nproc;
            MPI_Recv(res + (nproc - 1) * size / nproc,
                     rangeLength, MPI_DOUBLE, nproc - 1, 2, MPI_COMM_WORLD, &status);
        }
        printf("Data has recieved from process #%d\n", rank);

//        endtime = MPI_Wtime();
//        printf("Time taken: %lf", (endtime - starttime));
        for (int process = 0; process < nproc; process++) {
            MPI_Send(res, size, MPI_DOUBLE, process, 3, MPI_COMM_WORLD);
        }
        printf("Result vector has sent to all processes\n");
    }
    MPI_Recv(res, size, MPI_DOUBLE, ROOT, 3, MPI_COMM_WORLD, &status);
    printf("Result vector has received from root process to process #%d\n", rank);
}

void subVectors(double *a, double *b, int size) {
    for (size_t i = 0; i < size; i++) {
        a[i] = a[i] - b[i];
    }
}

void mulNumVec(double num, double *vec, int size) {
    for (size_t i = 0; i < size; i++) {
        vec[i] = num * vec[i];
    }
}

void singleIterate(double *A, double *x, double *b, int size) {
    double *res = (double *) malloc(size * sizeof(double));

    int erCode = MPI_Init(NULL, NULL);
    if (erCode) {
        printf("Startup error, execution stopped\n");
        MPI_Abort(MPI_COMM_WORLD, erCode);
    }

    // test
//    double *testMatrix = (double *) malloc(9 * sizeof(double));
//    testMatrix[0] = 0;
//    testMatrix[1] = 2;
//    testMatrix[2] = 2;
//    testMatrix[3] = 2;
//    testMatrix[4] = 2;
//    testMatrix[5] = 2;
//    testMatrix[6] = 2;
//    testMatrix[7] = 2;
//    testMatrix[8] = 2;
//    double *testVec = (double *) malloc(3 * sizeof(double));
//    testVec[0] = 2;
//    testVec[1] = 2;
//    testVec[2] = 2;
//    double *testRes = (double *) malloc (3 * sizeof(double));
//    mulMatVec(testMatrix, testVec, 3, testRes);
//    printArray(testRes, 3);


    mulMatVec(A, x, size, res);
    subVectors(res, b, size);
//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    printf("#%d res after Ax-b: \n", rank);

    while (euclideanNorm(res, size) / euclideanNorm(b, size) >= EPS) {
//        printf("norm in process #%d = %lf\n", rank, euclideanNorm(res, size) / euclideanNorm(b, size));
        mulMatVec(A, x, size, res);
        subVectors(res, b, size);
        mulNumVec(TAU, res, size);
        subVectors(x,res, size);
    }
    MPI_Finalize();
    free(res);
}

int main(int argc, char *argv[]) {
    // Если нужно будет использовать ввод с консоли.
//    char *end;
//    size_t N = strtoul(argv[1], &end, 10);
//    double epsilon = strtod(argv[2], &end);

    int N = 10;

    double *A = (double *) malloc(N * N * sizeof(double));
    initMatrix(A, N);
    double *b = (double *) malloc(N * sizeof(double));
    initRightPart(b, N);
    double *x = (double *) malloc(N * sizeof(double));
    initSolution(x, N);

    singleIterate(A, x, b, N);

    printArray(x, N);

    free(A);
    free(b);
    free(x);
    return 0;
}