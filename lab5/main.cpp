#include <iostream>
#include <pthread.h>
#include <cmath>
#include <mpi.h>
#include <cstdlib>
#include <cstring>

constexpr int L = 1000;
constexpr int NUM_OF_TASK_LISTS = 10;
constexpr int NUM_OF_TASKS = 20000;
constexpr int MIN_SHARING_TASKS = 2;

constexpr int FINISH_SIGNAL = -1;
constexpr int NO_TASKS = -2;
constexpr int TASKS_TAG = 2;
constexpr int TASK_COUNT_TAG = 3;

pthread_mutex_t mutex;

typedef struct {
    int *tasks;
    int remaining_tasks;
    int executed_tasks;
    int additional_tasks;
} Task_info;
Task_info task_info;

bool finished_execution = false;
double global_result = 0;
double summary_disbalance = 0;

void print_tasks(int *tasks, int process_rank) {
    std::cout << "Tasks in process " << process_rank << ":" << std::endl;
    for (int i = 0; i < NUM_OF_TASKS; i++) {
        std::cout << tasks[i] << " ";
    }
    std::cout << std::endl;
}

void init_task_set(int *tasks, int task_count, int iter_counter, int process_count, int process_rank) {
    for (int i = 0; i < task_count; i++) {
        tasks[i] = abs(50 - i % 100) * abs(process_rank - (iter_counter % process_count)) * L;
    }
}

double execute_tasks(const int *tasks) {
    int i = 0;
    double local_result = 0;
    while (true) {
        pthread_mutex_lock(&mutex);
        if (0 == task_info.remaining_tasks) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        int weight = tasks[i];
        task_info.executed_tasks++;
        task_info.remaining_tasks--;
        pthread_mutex_unlock(&mutex);
        i++;

        for (int j = 0; j < weight; j++) {
            local_result += sqrt(0.00005);
        }
    }
    pthread_mutex_lock(&mutex);
    task_info.remaining_tasks = 0;
    pthread_mutex_unlock(&mutex);

    return local_result;
}

void start_executor(int process_count, int process_rank) {
    pthread_mutex_lock(&mutex);
    task_info.tasks = new int[NUM_OF_TASKS];
    pthread_mutex_unlock(&mutex);
    double start_time, finish_time, iteration_duration, shortest_iteration, longest_iteration, current_disbalance;
    int total_executed_tasks = 0;
    double local_result = 0;
    double global_result_on_cur_iteration = 0;

    for (int i = 0; i < NUM_OF_TASK_LISTS; i++) {
        start_time = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "Iteration " << i << ". Initializing tasks. " << std::endl;

        pthread_mutex_lock(&mutex);
        init_task_set(task_info.tasks, NUM_OF_TASKS, i, process_count, process_rank);
        print_tasks(task_info.tasks, process_rank);
        task_info.remaining_tasks = NUM_OF_TASKS;
        pthread_mutex_unlock(&mutex);

        task_info.executed_tasks = 0;
        task_info.additional_tasks = 0;

        local_result += execute_tasks(task_info.tasks);

        int thread_response;

        for (int proc_idx = 0; proc_idx < process_count; proc_idx++) {
            if (proc_idx == process_rank) {
                continue;
            }
            MPI_Send(&process_rank, 1, MPI_INT, proc_idx, 1, MPI_COMM_WORLD);
            MPI_Recv(&thread_response, 1, MPI_INT, proc_idx, TASK_COUNT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (thread_response == NO_TASKS) {
                continue;
            }
            task_info.additional_tasks += thread_response;
            memset(task_info.tasks, 0, NUM_OF_TASKS * sizeof(int));
            MPI_Recv(task_info.tasks, thread_response, MPI_INT, proc_idx, TASKS_TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            pthread_mutex_lock(&mutex);
            task_info.remaining_tasks = thread_response;
            pthread_mutex_unlock(&mutex);
            print_tasks(task_info.tasks, process_rank);
            local_result += execute_tasks(task_info.tasks);
        }
        finish_time = MPI_Wtime();
        iteration_duration = finish_time - start_time;

        MPI_Allreduce(&iteration_duration, &longest_iteration, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&iteration_duration, &shortest_iteration, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&task_info.executed_tasks, &total_executed_tasks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_result, &global_result_on_cur_iteration, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        std::cout << "[EXECUTOR] Process " << process_rank << " executed " << task_info.executed_tasks <<
                  " tasks. " << task_info.additional_tasks << " were additional." << std::endl;

        pthread_mutex_lock(&mutex);
        global_result += global_result_on_cur_iteration;
        pthread_mutex_unlock(&mutex);

        current_disbalance = (longest_iteration - shortest_iteration) / longest_iteration;
        summary_disbalance += current_disbalance;
        std::cout << "[EXECUTOR] Sum of square roots on process "<< process_rank << " is " << local_result << ". Time taken: " << iteration_duration
                  << std::endl;
        if (process_rank == 0) {
            std::cout << "[EXECUTOR] Total executed tasks: " << total_executed_tasks << std::endl;
            std::cout << "[EXECUTOR] Max time difference: " << longest_iteration - shortest_iteration << std::endl;
            std::cout << "[EXECUTOR] Disbalance rate is " << current_disbalance * 100 << "%" << std::endl;
            std::cout << "Sum of square roots on all processes in iteration " << i << " is " << global_result_on_cur_iteration << std::endl;
        }
        local_result = 0;
    }

    std::cout << "Process " << process_rank << " finished iterations sending signal." << std::endl;

    pthread_mutex_lock(&mutex);
    if (process_rank == 0) {
        std::cout << "Global result: " << global_result << std::endl;
    }
    finished_execution = true;
    delete[] task_info.tasks;
    pthread_mutex_unlock(&mutex);

    int signal = FINISH_SIGNAL;
    MPI_Send(&signal, 1, MPI_INT, process_rank, 1, MPI_COMM_WORLD);
}

void *start_receiver(void *args) {
    int process_count, process_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    int asking_proc_rank, answer, pending_message;
    MPI_Status status;
//    MPI_Barrier(MPI_COMM_WORLD);
    int tasks_to_give = NUM_OF_TASKS;
    while (true) {
        pthread_mutex_lock(&mutex);
        if (finished_execution) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        pthread_mutex_unlock(&mutex);

        MPI_Recv(&pending_message, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);

        if (pending_message == FINISH_SIGNAL) {
            break;
        }
        asking_proc_rank = pending_message;

        pthread_mutex_lock(&mutex);
        if (task_info.remaining_tasks >= MIN_SHARING_TASKS) {
            answer = task_info.remaining_tasks / (process_count);
            std::cout << "Process " << process_rank << " can give " << answer << " to process " << asking_proc_rank << std::endl;
            task_info.remaining_tasks -= answer;
            MPI_Send(&answer, 1, MPI_INT, asking_proc_rank, TASK_COUNT_TAG, MPI_COMM_WORLD);
            MPI_Send(&task_info.tasks[tasks_to_give - answer], answer, MPI_INT, asking_proc_rank, TASKS_TAG,
                     MPI_COMM_WORLD);
            tasks_to_give -= answer;
        } else {
            answer = NO_TASKS;
            MPI_Send(&answer, 1, MPI_INT, asking_proc_rank, TASK_COUNT_TAG, MPI_COMM_WORLD);
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

    double start = MPI_Wtime();
    pthread_mutex_init(&mutex, nullptr);
    pthread_create(&receiver_thread, nullptr, start_receiver, nullptr);
    start_executor(process_count, process_rank);
    pthread_join(receiver_thread, nullptr);
    pthread_mutex_destroy(&mutex);

    if (process_rank == 0) {
        std::cout << "Summary disbalance:" << summary_disbalance / (NUM_OF_TASK_LISTS) * 100 << "%" << std::endl;
        std::cout << "Time taken: " << MPI_Wtime() - start << std::endl;
    }

    MPI_Finalize();
    return 0;
}