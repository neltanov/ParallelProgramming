#include <iostream>
#include <pthread.h>
#include <cmath>
#include <mpi.h>
#include <cstdlib>
#include <fstream>
#include <cstring>

#define L 1000
#define LISTS_COUNT 500
#define TASK_COUNT 2000
#define MIN_TASKS_TO_SHARE 2

#define EXECUTOR_FINISHED_WORK (-1)
#define SENDING_TASKS 656
#define SENDING_TASK_COUNT 787
#define NO_TASKS_TO_SHARE (-565)

pthread_t threads[2];
pthread_mutex_t mutex;
int *tasks;
std::ofstream *log_files;

double summary_disbalance = 0;
bool finished_execution = false;

int process_count;
int process_rank;
int remaining_tasks;
int executed_tasks;
int additional_tasks;
double global_result = 0;

void print_tasks(int *task_set) {
    std::cout << "Process :" << process_rank;
    for (int i = 0; i < TASK_COUNT; i++) {
        std::cout << task_set[i] << " ";
    }
}

void init_task_set(int *task_set, int task_count, int iter_counter) {
    for (int i = 0; i < task_count; i++) {
        task_set[i] = abs(50 - i % 100) * abs(process_rank - (iter_counter % process_count)) * L;
    }
}

void execute_task_set(int *task_set) {
    for (int i = 0; i < remaining_tasks; i++) {
        pthread_mutex_lock(&mutex);
        int weight = task_set[i];
        pthread_mutex_unlock(&mutex);

        for (int j = 0; j < weight; j++) {
            global_result += sqrt(0.001488);
        }

        executed_tasks++;
    }
    remaining_tasks = 0;
}

void *start_executor(void *args) {
    tasks = new int[TASK_COUNT];
    double start_time, finish_time, iteration_duration, shortest_iteration, longest_iteration;

    for (int i = 0; i < LISTS_COUNT; i++) {
        start_time = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "Iteration " << i << ". Initializing tasks. " << std::endl;
        init_task_set(tasks, TASK_COUNT, i);
        executed_tasks = 0;
        remaining_tasks = TASK_COUNT;
        additional_tasks = 0;

        execute_task_set(tasks);
        std::cout << "Process " << process_rank << " executed tasks in " <<
                  MPI_Wtime() - start_time << " Now requesting for some additional. " << std::endl;
        int thread_response;

        for (int proc_idx = 0; proc_idx < process_count; proc_idx++) {

            if (proc_idx != process_rank) {
                std::cout << "Process " << process_rank << " is asking " << proc_idx <<
                          " for some tasks." << std::endl;

                MPI_Send(&process_rank, 1, MPI_INT, proc_idx, 888, MPI_COMM_WORLD);

                std::cout << "waiting for task count" << std::endl;

                MPI_Recv(&thread_response, 1, MPI_INT, proc_idx, SENDING_TASK_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::cout << "Process " << proc_idx << " answered " << thread_response << std::endl;

                if (thread_response != NO_TASKS_TO_SHARE) {
                    additional_tasks = thread_response;
                    memset(tasks, 0, TASK_COUNT);

                    std::cout << "waiting for tasks" << std::endl;

                    MPI_Recv(tasks, additional_tasks, MPI_INT, proc_idx, SENDING_TASKS, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                    pthread_mutex_lock(&mutex);
                    remaining_tasks = additional_tasks;
                    pthread_mutex_unlock(&mutex);
                    execute_task_set(tasks);
                }
            }

        }
        finish_time = MPI_Wtime();
        iteration_duration = finish_time - start_time;

        MPI_Allreduce(&iteration_duration, &longest_iteration, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&iteration_duration, &shortest_iteration, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        std::cout << "Process " << process_rank << " executed " << executed_tasks <<
                  " tasks. " << additional_tasks << " were additional." << std::endl;
        std::cout << "Cos sum is " << global_result << ". Time taken: " << iteration_duration << std::endl;
        summary_disbalance += (longest_iteration - shortest_iteration) / longest_iteration;
        std::cout << "Max time difference: " << longest_iteration - shortest_iteration << std::endl;
        std::cout << "Disbalance rate is " <<
                  ((longest_iteration - shortest_iteration) / longest_iteration) * 100 << "%" << std::endl;
        log_files[process_rank] << iteration_duration << std::endl;
    }

    std::cout << "Proc " << process_rank << " finished iterations sending signal" << std::endl;
    pthread_mutex_lock(&mutex);
    finished_execution = true;
    pthread_mutex_unlock(&mutex);
    int signal = EXECUTOR_FINISHED_WORK;
    MPI_Send(&signal, 1, MPI_INT, process_rank, 888, MPI_COMM_WORLD);
    delete[] tasks;
    pthread_exit(nullptr);
}

void *start_reciever(void *args) {
    int asking_proc_rank, answer, pending_message;
    MPI_Status status;
    MPI_Barrier(MPI_COMM_WORLD);
    while (!finished_execution) {
        MPI_Recv(&pending_message, 1, MPI_INT, MPI_ANY_SOURCE, 888, MPI_COMM_WORLD, &status);

        if (pending_message == EXECUTOR_FINISHED_WORK) {
            std::cout << "Executor finished work on proc " << process_rank << std::endl;
        }
        asking_proc_rank = pending_message;
        pthread_mutex_lock(&mutex);
        std::cout << "Process " << asking_proc_rank << " requested tasks. I have " <<
                  remaining_tasks << " tasks now. " << std::endl;
        if (remaining_tasks >= MIN_TASKS_TO_SHARE) {
            answer = remaining_tasks / (process_count * 2);
            remaining_tasks = remaining_tasks / (process_count * 2);

            std::cout << "Sharing " << answer << " tasks. " << std::endl;

            MPI_Send(&answer, 1, MPI_INT, asking_proc_rank, SENDING_TASK_COUNT, MPI_COMM_WORLD);
            MPI_Send(&tasks[TASK_COUNT - answer], answer, MPI_INT, asking_proc_rank, SENDING_TASKS, MPI_COMM_WORLD);
        } else {
            answer = NO_TASKS_TO_SHARE;
            MPI_Send(&answer, 1, MPI_INT, asking_proc_rank, SENDING_TASK_COUNT, MPI_COMM_WORLD);
        }
        pthread_mutex_unlock(&mutex);
    }

    pthread_exit(nullptr);
}


int main(int argc, char *argv[]) {
    int thread_support;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_support);
    if (thread_support != MPI_THREAD_MULTIPLE) {
        MPI_Finalize();
        return -1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    pthread_mutex_init(&mutex, nullptr);
    pthread_attr_t thread_attributes;

    log_files = new std::ofstream[process_count];
    char *name = new char[12];
    for (int i = 0; i < process_count; i++) {
        sprintf(name, "Log_%d.txt", i);
        log_files[i].open(name);
    }
    double start = MPI_Wtime();
    pthread_attr_init(&thread_attributes);
    pthread_attr_setdetachstate(&thread_attributes, PTHREAD_CREATE_JOINABLE);
    pthread_create(&threads[0], &thread_attributes, start_reciever, NULL);
    pthread_create(&threads[1], &thread_attributes, start_executor, NULL);
    pthread_join(threads[0], nullptr);
    pthread_join(threads[1], nullptr);
    pthread_attr_destroy(&thread_attributes);
    pthread_mutex_destroy(&mutex);

    if (process_rank == 0) {
        std::cout << "Summary disbalance:" << summary_disbalance / (LISTS_COUNT) * 100 << "%" << std::endl;
        std::cout << "time taken: " << MPI_Wtime() - start << std::endl;
    }

    MPI_Finalize();
    return 0;
}