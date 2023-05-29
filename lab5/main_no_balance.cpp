#include <iostream>
#include <cmath>
#include <mpi.h>
#include <cstdlib>

constexpr int L = 1000;
constexpr int LISTS_COUNT = 10;
constexpr int NUMBER_OF_TASKS = 20000;
constexpr int MIN_SHARING_TASKS = 2;

constexpr int FINISH_SIGNAL = -1;
constexpr int NO_TASKS = -2;
constexpr int TASKS_TAG = 2;
constexpr int TASK_COUNT_TAG = 3;

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

void print_tasks(int *task_set, int process_rank) {
    std::cout << "Process: " << process_rank << std::endl;
    for (int i = 0; i < NUMBER_OF_TASKS; i++) {
        std::cout << task_set[i] << " ";
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
        if (0 == task_info.remaining_tasks) {
            break;
        }
        int weight = tasks[i];
        i++;
        task_info.executed_tasks++;
        task_info.remaining_tasks--;

        for (int j = 0; j < weight; j++) {
            local_result += sqrt(0.00005);
        }
    }
    task_info.remaining_tasks = 0;
    return local_result;
}

void start_executor(int process_count, int process_rank) {
    task_info.tasks = new int[NUMBER_OF_TASKS];
    double start_time, finish_time, iteration_duration, shortest_iteration, longest_iteration, current_disbalance;
    int total_executed_tasks = 0;
    double local_result = 0;
    double global_result_on_cur_iteration = 0;
    for (int i = 0; i < LISTS_COUNT; i++) {
        start_time = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "Iteration " << i << ". Initializing tasks. " << std::endl;

        init_task_set(task_info.tasks, NUMBER_OF_TASKS, i, process_count, process_rank);
        print_tasks(task_info.tasks, process_rank);

        task_info.remaining_tasks = NUMBER_OF_TASKS;
        task_info.executed_tasks = 0;
        task_info.additional_tasks = 0;

        local_result += execute_tasks(task_info.tasks);

        finish_time = MPI_Wtime();
        iteration_duration = finish_time - start_time;

        MPI_Allreduce(&iteration_duration, &longest_iteration, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&iteration_duration, &shortest_iteration, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&task_info.executed_tasks, &total_executed_tasks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_result, &global_result_on_cur_iteration, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        global_result += global_result_on_cur_iteration;

        std::cout << "[EXECUTOR] Process " << process_rank << " executed " << task_info.executed_tasks << " tasks."<< std::endl;
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
    if (process_rank == 0) {
        std::cout << "Global result: " << global_result << std::endl;
    }

    finished_execution = true;
    delete[] task_info.tasks;

}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int process_count;
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    double start = MPI_Wtime();
    start_executor(process_count, process_rank);

    if (process_rank == 0) {
        std::cout << "Summary disbalance:" << summary_disbalance / (LISTS_COUNT) * 100 << "%" << std::endl;
        std::cout << "Time taken: " << MPI_Wtime() - start << std::endl;
    }

    MPI_Finalize();
    return 0;
}