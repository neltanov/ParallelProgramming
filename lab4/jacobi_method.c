#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>


typedef struct {
    float * Values;
    int N;
    float X0;
    float Y0;
    float D;

    int LowerEdge;
    int UpperEdge;
} Grid2D;

typedef float (* Function)(
        float x,
        float y);

typedef float (* GridFunction)(
        Grid2D grid,
        int i,
        int j);


Grid2D grid_new(
        int size,
        float x0,
        float y0,
        float real_size) {
    assert(size >= 1 && "Size must be >= 1");

    float * const data = (float *) calloc(size * size, sizeof(*data));

    return (Grid2D) {
            .Values = data,
            .N = size,
            .X0 = x0,
            .Y0 = y0,
            .D = real_size,
            .LowerEdge = 0,
            .UpperEdge = size - 1,
    };
}


Grid2D grid_copy(Grid2D grid) {
    float * const data = (float *) calloc(grid.N * grid.N, sizeof(*data));
    memcpy(data, grid.Values, sizeof(*data) * grid.N * grid.N);

    Grid2D copy = grid;
    copy.Values = data;

    return copy;
}


void grid_free(Grid2D grid) {
    free(grid.Values);
}


float * grid_at(
        Grid2D grid,
        int i,
        int j) {
    return &grid.Values[j * grid.N + i];
}


float grid_index_to_x(
        Grid2D grid,
        int i) {
    return grid.X0 + grid.D * (float) i / (float) (grid.N - 1);
}


float grid_index_to_y(
        Grid2D grid,
        int j) {
    return grid.Y0 + grid.D * (float) j / (float) (grid.N - 1);
}


void grid_print(Grid2D grid) {
    for (int j = 0; j < grid.N; j += 1) {
        for (int i = 0; i < grid.N; i += 1) {
            printf("%.2f ", *grid_at(grid, i, j));
        }

        printf("\n");
    }
}


float grid_max_diff(
        Grid2D grid,
        Function target) {
    float delta = 0.0f;

    for (int j = 0; j < grid.N; j += 1) {
        for (int i = 0; i < grid.N; i += 1) {
            const float x = grid_index_to_x(grid, i);
            const float y = grid_index_to_y(grid, j);
            const float current_delta = fabsf(target(x, y) - *grid_at(grid, i, j));

            delta = fmaxf(delta, current_delta);
        }
    }

    return delta;
}


void grid_swap(
        Grid2D * g1,
        Grid2D * g2) {
    Grid2D tmp = *g1;
    *g1 = *g2;
    *g2 = tmp;
}


float phi(
        float x,
        float y) {
    return x * x + y * y;
}


void fill_edges(
        Grid2D grid,
        Function target_fn) {
    float x, y;
    for (int i = 0; i < grid.N; i += 1) {
        x = grid_index_to_x(grid, i);

        *grid_at(grid, i, grid.LowerEdge) = target_fn(x, grid_index_to_y(grid, grid.LowerEdge));
        *grid_at(grid, i, grid.UpperEdge) = target_fn(x, grid_index_to_y(grid, grid.UpperEdge));

        y = grid_index_to_y(grid, i);

        *grid_at(grid, grid.LowerEdge, i) = target_fn(grid_index_to_x(grid, grid.LowerEdge), y);
        *grid_at(grid, grid.UpperEdge, i) = target_fn(grid_index_to_x(grid, grid.UpperEdge), y);
    }
}


void solve(
        Grid2D grid,
        GridFunction grid_approximation,
        float target_delta) {
    float iter_delta = INFINITY;
    int n_iter = 0;

    Grid2D next = grid_copy(grid);

    while (iter_delta > target_delta && n_iter < 20) {
#ifdef PRINT_ITER
        printf("n_iter = %d\niter_delta = %.8f\n", n_iter, iter_delta);
        grid_print(grid);
        printf("\n");
#endif // PRINT_ITER
        iter_delta = 0;

        for (int j = grid.LowerEdge + 1; j <= grid.UpperEdge - 1; j += 1) {
            for (int i = grid.LowerEdge + 1; i <= grid.UpperEdge - 1; i += 1) {
                const float old_value = *grid_at(grid, i, j);
                const float new_value = grid_approximation(grid, i, j);
                // printf("[%d, %d] %.2f -> %.2f\n",i, j, old_value, new_value);
                *grid_at(next, i, j) = new_value;

                const float current_delta = fabsf(new_value - old_value);
                iter_delta = fmaxf(iter_delta, current_delta);
            }
        }

        grid_swap(&next, &grid);

        n_iter += 1;
        printf("n_iter = %d\n", n_iter);
    }

    if (n_iter % 2 != 0) {
        grid_swap(&grid, &next);
    }
    grid_free(next);

#ifdef PRINT_ITER
    printf("DONE\nn_iter = %d\niter_delta = %.8f\n", n_iter, iter_delta);
    grid_print(grid);
    printf("\n");
#endif // PRINT_ITER
}


float approximate_value(
        Grid2D grid,
        int i,
        int j) {
    const float x = grid_index_to_x(grid, i);
    const float y = grid_index_to_y(grid, j);
    const float a = 1e5f;

    const float h = grid.D / (float) (grid.N - 1);
    const float hx_sq = h * h;
    const float hy_sq = h * h;

    const float C = 1.0f / (a + 2.0f / hx_sq + 2.0f / hy_sq);
    const float x_part = (*grid_at(grid, i - 1, j) + *grid_at(grid, i + 1, j)) / hx_sq;
    const float y_part = (*grid_at(grid, i, j - 1) + *grid_at(grid, i, j + 1)) / hy_sq;
    const float rho = 6 - a * phi(x, y);

    return C * (x_part + y_part - rho);
}


int main(void) {
    const float eps = 1e-8f;

    const int grid_size = 51;
    const float grid_min = -1.0f;
    const float grid_real_size = 2.0f;

    Grid2D grid = grid_new(grid_size, grid_min, grid_min, grid_real_size);

    Function target_fn = phi;
    fill_edges(grid, target_fn);
    solve(grid, approximate_value, eps);

    printf("max_delta = %.8f\n", grid_max_diff(grid, target_fn));

    grid_free(grid);

    return EXIT_SUCCESS;
}