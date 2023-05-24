#include <iostream>
#include <pthread.h>
#include <cmath>
#include <mpi.h>
#include <cstdlib>
#include <cstring>

#define L 1000
#define LISTS_COUNT 500
#define TASK_COUNT 2000
#define MIN_TASKS_TO_SHARE 2

#define EXECUTOR_FINISHED_WORK (-1)
#define NO_TASKS_TO_SHARE (-2)
#define SENDING_TASKS 500
#define SENDING_TASK_COUNT 500

bool finished_execution = false;

pthread_mutex_t mutex;
typedef struct {
    int *tasks;
    int remaining_tasks;
    int executed_tasks;
    int additional_tasks;
} Task_info;

Task_info task_info;

double global_result = 0;
double summary_disbalance = 0;

void print_tasks(int *task_set, int process_rank) {
    std::cout << "Process :" << process_rank;
    for (int i = 0; i < TASK_COUNT; i++) {
        std::cout << task_set[i] << " ";
    }
}

void init_task_set(int *task_set, int task_count, int iter_counter, int process_count, int process_rank) {
    for (int i = 0; i < task_count; i++) {
        task_set[i] = abs(50 - i % 100) * abs(process_rank - (iter_counter % process_count)) * L;
    }
}

void execute_task_set(const int *task_set) {
    int i = 0;
    while (true) {
        pthread_mutex_lock(&mutex);
        if (i == task_info.remaining_tasks) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        int weight = task_set[i];
        i++;
        pthread_mutex_unlock(&mutex);

        for (int j = 0; j < weight; j++) {
            global_result += sqrt(0.00005);
        }
        pthread_mutex_lock(&mutex);
        task_info.executed_tasks++;
        pthread_mutex_unlock(&mutex);
    }
    pthread_mutex_lock(&mutex);
    task_info.remaining_tasks = 0;
    pthread_mutex_unlock(&mutex);
}

void start_executor(int process_count, int process_rank) {
    task_info.tasks = new int[TASK_COUNT];
    double start_time, finish_time, iteration_duration, shortest_iteration, longest_iteration, current_disbalance;

    for (int i = 0; i < LISTS_COUNT; i++) {
        start_time = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "Iteration " << i << ". Initializing tasks. " << std::endl;

        init_task_set(task_info.tasks, TASK_COUNT, i, process_count, process_rank);
        pthread_mutex_lock(&mutex);
        task_info.executed_tasks = 0;
        task_info.remaining_tasks = TASK_COUNT;
        task_info.additional_tasks = 0;
        pthread_mutex_unlock(&mutex);

        execute_task_set(task_info.tasks);
        std::cout << "Process " << process_rank << " executed tasks in " <<
                  MPI_Wtime() - start_time << ". Now requesting for some additional. " << std::endl;
        int thread_response;

        for (int proc_idx = 0; proc_idx < process_count; proc_idx++) {
            if (proc_idx == process_rank) {
                continue;
            }
            std::cout << "Process " << process_rank << " is asking " << proc_idx <<
                      " for some tasks." << std::endl;

            MPI_Send(&process_rank, 1, MPI_INT, proc_idx, 1, MPI_COMM_WORLD);

            std::cout << "Proc" << process_rank << " is waiting for task count from " << proc_idx << std::endl;

            MPI_Recv(&thread_response, 1, MPI_INT, proc_idx, SENDING_TASK_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::cout << "Process " << proc_idx << " answered: 'I have " << thread_response << " tasks'." << std::endl;

            if (thread_response == NO_TASKS_TO_SHARE) {
                continue;
            }

            task_info.additional_tasks = thread_response;
            memset(task_info.tasks, 0, TASK_COUNT);

            std::cout << "Process" << process_rank << " is waiting for tasks." << std::endl;
            // TODO: endless waiting
            MPI_Recv(task_info.tasks, task_info.additional_tasks, MPI_INT, proc_idx, SENDING_TASKS, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

            pthread_mutex_lock(&mutex);
            task_info.remaining_tasks = task_info.additional_tasks;
            pthread_mutex_unlock(&mutex);
            execute_task_set(task_info.tasks);
        }
        finish_time = MPI_Wtime();
        iteration_duration = finish_time - start_time;

        MPI_Allreduce(&iteration_duration, &longest_iteration, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&iteration_duration, &shortest_iteration, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        std::cout << "Process " << process_rank << " executed " << task_info.executed_tasks <<
                  " tasks. " << task_info.additional_tasks << " were additional." << std::endl;
        std::cout << "Sum of square roots is " << global_result << ". Time taken: " << iteration_duration << std::endl;

        current_disbalance = (longest_iteration - shortest_iteration) / longest_iteration;
        summary_disbalance += current_disbalance;
        std::cout << "Max time difference: " << longest_iteration - shortest_iteration << std::endl;
        std::cout << "Disbalance rate is " << current_disbalance * 100 << "%" << std::endl;
    }

    std::cout << "Process " << process_rank << " finished iterations sending signal" << std::endl;

    pthread_mutex_lock(&mutex);
    finished_execution = true;
    delete[] task_info.tasks;
    pthread_mutex_unlock(&mutex);

    int signal = EXECUTOR_FINISHED_WORK;
    MPI_Send(&signal, 1, MPI_INT, process_rank, 1, MPI_COMM_WORLD);
}

void *start_reciever(void *args) {
    int process_count;
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int asking_proc_rank, answer, pending_message;
    MPI_Status status;
    MPI_Barrier(MPI_COMM_WORLD);
    while (true) {
        pthread_mutex_lock(&mutex);
        if (finished_execution) {
            break;
        }
        pthread_mutex_unlock(&mutex);

        MPI_Recv(&pending_message, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);

        if (pending_message == EXECUTOR_FINISHED_WORK) {
            std::cout << "Executor finished work on process " << process_rank << std::endl;
            break;
        }
        asking_proc_rank = pending_message;
        pthread_mutex_lock(&mutex);
        std::cout << "Process " << asking_proc_rank << " requested tasks. I have " <<
                  task_info.remaining_tasks << " tasks now. " << std::endl;
        if (task_info.remaining_tasks >= MIN_TASKS_TO_SHARE) {
            answer = task_info.remaining_tasks / (process_count * 2);
            task_info.remaining_tasks = task_info.remaining_tasks / (process_count * 2);

            std::cout << "Sharing " << answer << " tasks. " << std::endl;

            MPI_Send(&answer, 1, MPI_INT, asking_proc_rank, SENDING_TASK_COUNT, MPI_COMM_WORLD);

            MPI_Send(&task_info.tasks[TASK_COUNT - answer], answer, MPI_INT, asking_proc_rank, SENDING_TASKS,
                     MPI_COMM_WORLD);
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

    int process_count;
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    pthread_t receiver_thread;
    pthread_attr_t thread_attributes;

    double start = MPI_Wtime();
    pthread_mutex_init(&mutex, nullptr);
    pthread_attr_init(&thread_attributes);
    pthread_attr_setdetachstate(&thread_attributes, PTHREAD_CREATE_JOINABLE);
    pthread_create(&receiver_thread, &thread_attributes, start_reciever, nullptr);
    start_executor(process_count, process_rank);
    pthread_join(receiver_thread, nullptr);
    pthread_attr_destroy(&thread_attributes);
    pthread_mutex_destroy(&mutex);

    if (process_rank == 0) {
        std::cout << "Summary disbalance:" << summary_disbalance / (LISTS_COUNT) * 100 << "%" << std::endl;
        std::cout << "time taken: " << MPI_Wtime() - start << std::endl;
    }

    MPI_Finalize();
    return 0;
}