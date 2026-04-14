#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <cuda_runtime.h>

extern "C" {
#include "data.h"
#include "vtk.h"
#include "setup.h"
#include "boundary.h"
#include "args.h"
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define IDX(i, j, width) ((i) * (width) + (j))

double *d_u, *d_v, *d_p, *d_f, *d_g, *d_rhs;
char *d_flag;
double *d_residual_array;  // For reduction
double *d_umax_array, *d_vmax_array;  // For CFL reduction

double *h_u_flat, *h_v_flat, *h_p_flat, *h_f_flat, *h_g_flat, *h_rhs_flat;
char *h_flag_flat;

int grid_width, grid_height;

/**
 * @brief Linearize 2D pointer arrays into flat 1D arrays for GPU transfer
 */
void linearize_2d_arrays() {
    grid_width = jmax + 2;   // Width (j-direction)
    grid_height = imax + 2;  // Height (i-direction)

    size_t array_size = grid_height * grid_width;

    h_u_flat = (double*)malloc(array_size * sizeof(double));
    h_v_flat = (double*)malloc(array_size * sizeof(double));
    h_p_flat = (double*)malloc(array_size * sizeof(double));
    h_f_flat = (double*)malloc(array_size * sizeof(double));
    h_g_flat = (double*)malloc(array_size * sizeof(double));
    h_rhs_flat = (double*)malloc(array_size * sizeof(double));
    h_flag_flat = (char*)malloc(array_size * sizeof(char));

    for (int i = 0; i < grid_height; i++) {
        for (int j = 0; j < grid_width; j++) {
            int idx = IDX(i, j, grid_width);
            h_u_flat[idx] = u[i][j];
            h_v_flat[idx] = v[i][j];
            h_p_flat[idx] = p[i][j];
            h_f_flat[idx] = f[i][j];
            h_g_flat[idx] = g[i][j];
            h_rhs_flat[idx] = rhs[i][j];
            h_flag_flat[idx] = flag[i][j];
        }
    }
}

/**
 * @brief Copy linearized arrays back to 2D pointer arrays
 */
void delinearize_2d_arrays() {
    for (int i = 0; i < grid_height; i++) {
        for (int j = 0; j < grid_width; j++) {
            int idx = IDX(i, j, grid_width);
            u[i][j] = h_u_flat[idx];
            v[i][j] = h_v_flat[idx];
            p[i][j] = h_p_flat[idx];
            f[i][j] = h_f_flat[idx];
            g[i][j] = h_g_flat[idx];
            rhs[i][j] = h_rhs_flat[idx];
        }
    }
}

/**
 * @brief Allocate GPU memory for all arrays
 */
void allocate_device_memory() {
    size_t array_size = grid_height * grid_width * sizeof(double);
    size_t flag_size = grid_height * grid_width * sizeof(char);

    CUDA_CHECK(cudaMalloc(&d_u, array_size));
    CUDA_CHECK(cudaMalloc(&d_v, array_size));
    CUDA_CHECK(cudaMalloc(&d_p, array_size));
    CUDA_CHECK(cudaMalloc(&d_f, array_size));
    CUDA_CHECK(cudaMalloc(&d_g, array_size));
    CUDA_CHECK(cudaMalloc(&d_rhs, array_size));
    CUDA_CHECK(cudaMalloc(&d_flag, flag_size));

    int max_blocks = ((grid_height + 15) / 16) * ((grid_width + 15) / 16);
    CUDA_CHECK(cudaMalloc(&d_residual_array, max_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_umax_array, max_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vmax_array, max_blocks * sizeof(double)));
}

/**
 * @brief Transfer data from host to device
 */
void copy_to_device() {
    size_t array_size = grid_height * grid_width * sizeof(double);
    size_t flag_size = grid_height * grid_width * sizeof(char);

    CUDA_CHECK(cudaMemcpy(d_u, h_u_flat, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v_flat, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p, h_p_flat, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f, h_f_flat, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, h_g_flat, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs, h_rhs_flat, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flag, h_flag_flat, flag_size, cudaMemcpyHostToDevice));
}

/**
 * @brief Transfer data from device to host
 */
void copy_from_device() {
    size_t array_size = grid_height * grid_width * sizeof(double);

    CUDA_CHECK(cudaMemcpy(h_u_flat, d_u, array_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v_flat, d_v, array_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_p_flat, d_p, array_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_f_flat, d_f, array_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_g_flat, d_g, array_size, cudaMemcpyDeviceToHost));
}

/**
 * @brief Free device memory
 */
void free_device_memory() {
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_f));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_rhs));
    CUDA_CHECK(cudaFree(d_flag));
    CUDA_CHECK(cudaFree(d_residual_array));
    CUDA_CHECK(cudaFree(d_umax_array));
    CUDA_CHECK(cudaFree(d_vmax_array));
}

/**
 * @brief Free host flat arrays
 */
void free_host_flat_arrays() {
    free(h_u_flat);
    free(h_v_flat);
    free(h_p_flat);
    free(h_f_flat);
    free(h_g_flat);
    free(h_rhs_flat);
    free(h_flag_flat);
}


/**
 * @brief CUDA kernel to compute tentative velocity F
 */
__global__ void compute_f_kernel(double *u, double *v, double *f, char *flag,
                                  double del_t, double delx, double dely,
                                  double Re, double gamma,
                                  int imax, int jmax, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;  // Start from 1
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;  // Start from 1

    if (i < imax && j <= jmax) {
        int idx = IDX(i, j, width);
        int idx_ip1 = IDX(i+1, j, width);

        if ((flag[idx] & 0x10) && (flag[idx_ip1] & 0x10)) {  // C_F = 0x10
            int idx_im1 = IDX(i-1, j, width);
            int idx_jp1 = IDX(i, j+1, width);
            int idx_jm1 = IDX(i, j-1, width);
            int idx_ip1_jp1 = IDX(i+1, j+1, width);
            int idx_ip1_jm1 = IDX(i+1, j-1, width);

            double du2dx = ((u[idx] + u[idx_ip1]) * (u[idx] + u[idx_ip1]) +
                            gamma * fabs(u[idx] + u[idx_ip1]) * (u[idx] - u[idx_ip1]) -
                            (u[idx_im1] + u[idx]) * (u[idx_im1] + u[idx]) -
                            gamma * fabs(u[idx_im1] + u[idx]) * (u[idx_im1] - u[idx]))
                            / (4.0 * delx);

            double duvdy = ((v[idx] + v[idx_ip1]) * (u[idx] + u[idx_jp1]) +
                            gamma * fabs(v[idx] + v[idx_ip1]) * (u[idx] - u[idx_jp1]) -
                            (v[idx_jm1] + v[idx_ip1_jm1]) * (u[idx_jm1] + u[idx]) -
                            gamma * fabs(v[idx_jm1] + v[idx_ip1_jm1]) * (u[idx_jm1] - u[idx]))
                            / (4.0 * dely);

            double laplu = (u[idx_ip1] - 2.0 * u[idx] + u[idx_im1]) / (delx * delx) +
                           (u[idx_jp1] - 2.0 * u[idx] + u[idx_jm1]) / (dely * dely);

            f[idx] = u[idx] + del_t * (laplu / Re - du2dx - duvdy);
        } else {
            f[idx] = u[idx];
        }
    }
}

/**
 * @brief CUDA kernel to compute tentative velocity G
 */
__global__ void compute_g_kernel(double *u, double *v, double *g, char *flag,
                                  double del_t, double delx, double dely,
                                  double Re, double gamma,
                                  int imax, int jmax, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;  // Start from 1
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;  // Start from 1

    if (i <= imax && j < jmax) {
        int idx = IDX(i, j, width);
        int idx_jp1 = IDX(i, j+1, width);

        if ((flag[idx] & 0x10) && (flag[idx_jp1] & 0x10)) {  // C_F = 0x10
            int idx_im1 = IDX(i-1, j, width);
            int idx_ip1 = IDX(i+1, j, width);
            int idx_jm1 = IDX(i, j-1, width);
            int idx_im1_jp1 = IDX(i-1, j+1, width);
            int idx_ip1_jp1 = IDX(i+1, j+1, width);
            int idx_im1_jm1 = IDX(i-1, j-1, width);

            double duvdx = ((u[idx] + u[idx_jp1]) * (v[idx] + v[idx_ip1]) +
                            gamma * fabs(u[idx] + u[idx_jp1]) * (v[idx] - v[idx_ip1]) -
                            (u[idx_im1] + u[idx_im1_jp1]) * (v[idx_im1] + v[idx]) -
                            gamma * fabs(u[idx_im1] + u[idx_im1_jp1]) * (v[idx_im1] - v[idx]))
                            / (4.0 * delx);

            double dv2dy = ((v[idx] + v[idx_jp1]) * (v[idx] + v[idx_jp1]) +
                            gamma * fabs(v[idx] + v[idx_jp1]) * (v[idx] - v[idx_jp1]) -
                            (v[idx_jm1] + v[idx]) * (v[idx_jm1] + v[idx]) -
                            gamma * fabs(v[idx_jm1] + v[idx]) * (v[idx_jm1] - v[idx]))
                            / (4.0 * dely);

            double laplv = (v[idx_ip1] - 2.0 * v[idx] + v[idx_im1]) / (delx * delx) +
                           (v[idx_jp1] - 2.0 * v[idx] + v[idx_jm1]) / (dely * dely);

            g[idx] = v[idx] + del_t * (laplv / Re - duvdx - dv2dy);
        } else {
            g[idx] = v[idx];
        }
    }
}

/**
 * @brief CUDA kernel to set f and g at external boundaries
 */
__global__ void set_fg_boundaries_kernel(double *u, double *v, double *f, double *g,
                                          int imax, int jmax, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 1 && idx <= jmax) {
        f[IDX(0, idx, width)] = u[IDX(0, idx, width)];
        f[IDX(imax, idx, width)] = u[IDX(imax, idx, width)];
    }

    if (idx >= 1 && idx <= imax) {
        g[IDX(idx, 0, width)] = v[IDX(idx, 0, width)];
        g[IDX(idx, jmax, width)] = v[IDX(idx, jmax, width)];
    }
}

/**
 * @brief CUDA kernel to compute RHS of pressure Poisson equation
 */
__global__ void compute_rhs_kernel(double *f, double *g, double *rhs, char *flag,
                                    double del_t, double delx, double dely,
                                    int imax, int jmax, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;  // Start from 1
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;  // Start from 1

    if (i <= imax && j <= jmax) {
        int idx = IDX(i, j, width);

        if (flag[idx] & 0x10) {  // C_F = 0x10
            int idx_im1 = IDX(i-1, j, width);
            int idx_jm1 = IDX(i, j-1, width);

            rhs[idx] = ((f[idx] - f[idx_im1]) / delx +
                        (g[idx] - g[idx_jm1]) / dely) / del_t;
        }
    }
}

/**
 * @brief CUDA kernel for Red cells in Red/Black SOR iteration
 */
__global__ void sor_red_kernel(double *p, double *rhs, char *flag,
                                double rdx2, double rdy2, double omega,
                                int imax, int jmax, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= imax && j <= jmax) {
        if ((i + j) % 2 == 0) {
            int idx = IDX(i, j, width);
            char f = flag[idx];

            if (f & 0x10) {  // C_F = 0x10
                int idx_ip1 = IDX(i+1, j, width);
                int idx_im1 = IDX(i-1, j, width);
                int idx_jp1 = IDX(i, j+1, width);
                int idx_jm1 = IDX(i, j-1, width);

                if (f == (0x10 | 0x0F)) {  // C_F | B_NSEW
                    double beta_2 = -omega / (2.0 * (rdx2 + rdy2));
                    p[idx] = (1.0 - omega) * p[idx] -
                             beta_2 * ((p[idx_ip1] + p[idx_im1]) * rdx2 +
                                       (p[idx_jp1] + p[idx_jm1]) * rdy2 -
                                       rhs[idx]);
                } else {
                    double eps_E = ((flag[idx_ip1] & 0x10) ? 1.0 : 0.0);
                    double eps_W = ((flag[idx_im1] & 0x10) ? 1.0 : 0.0);
                    double eps_N = ((flag[idx_jp1] & 0x10) ? 1.0 : 0.0);
                    double eps_S = ((flag[idx_jm1] & 0x10) ? 1.0 : 0.0);

                    double beta_mod = -omega / ((eps_E + eps_W) * rdx2 + (eps_N + eps_S) * rdy2);
                    p[idx] = (1.0 - omega) * p[idx] -
                             beta_mod * ((eps_E * p[idx_ip1] + eps_W * p[idx_im1]) * rdx2 +
                                         (eps_N * p[idx_jp1] + eps_S * p[idx_jm1]) * rdy2 -
                                         rhs[idx]);
                }
            }
        }
    }
}

/**
 * @brief CUDA kernel for Black cells in Red/Black SOR iteration
 */
__global__ void sor_black_kernel(double *p, double *rhs, char *flag,
                                  double rdx2, double rdy2, double omega,
                                  int imax, int jmax, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= imax && j <= jmax) {
        if ((i + j) % 2 == 1) {
            int idx = IDX(i, j, width);
            char f = flag[idx];

            if (f & 0x10) {  // C_F = 0x10
                int idx_ip1 = IDX(i+1, j, width);
                int idx_im1 = IDX(i-1, j, width);
                int idx_jp1 = IDX(i, j+1, width);
                int idx_jm1 = IDX(i, j-1, width);

                if (f == (0x10 | 0x0F)) {  // C_F | B_NSEW
                    double beta_2 = -omega / (2.0 * (rdx2 + rdy2));
                    p[idx] = (1.0 - omega) * p[idx] -
                             beta_2 * ((p[idx_ip1] + p[idx_im1]) * rdx2 +
                                       (p[idx_jp1] + p[idx_jm1]) * rdy2 -
                                       rhs[idx]);
                } else {
                    double eps_E = ((flag[idx_ip1] & 0x10) ? 1.0 : 0.0);
                    double eps_W = ((flag[idx_im1] & 0x10) ? 1.0 : 0.0);
                    double eps_N = ((flag[idx_jp1] & 0x10) ? 1.0 : 0.0);
                    double eps_S = ((flag[idx_jm1] & 0x10) ? 1.0 : 0.0);

                    double beta_mod = -omega / ((eps_E + eps_W) * rdx2 + (eps_N + eps_S) * rdy2);
                    p[idx] = (1.0 - omega) * p[idx] -
                             beta_mod * ((eps_E * p[idx_ip1] + eps_W * p[idx_im1]) * rdx2 +
                                         (eps_N * p[idx_jp1] + eps_S * p[idx_jm1]) * rdy2 -
                                         rhs[idx]);
                }
            }
        }
    }
}

/**
 * @brief CUDA kernel to compute residual for convergence check
 */
__global__ void compute_residual_kernel(double *p, double *rhs, char *flag,
                                         double rdx2, double rdy2,
                                         double *residual_array,
                                         int imax, int jmax, int width) {
    __shared__ double sdata[256];  // Shared memory for block reduction

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double local_sum = 0.0;

    if (i <= imax && j <= jmax) {
        int idx = IDX(i, j, width);

        if (flag[idx] & 0x10) {  // C_F
            int idx_ip1 = IDX(i+1, j, width);
            int idx_im1 = IDX(i-1, j, width);
            int idx_jp1 = IDX(i, j+1, width);
            int idx_jm1 = IDX(i, j-1, width);

            double eps_E = ((flag[idx_ip1] & 0x10) ? 1.0 : 0.0);
            double eps_W = ((flag[idx_im1] & 0x10) ? 1.0 : 0.0);
            double eps_N = ((flag[idx_jp1] & 0x10) ? 1.0 : 0.0);
            double eps_S = ((flag[idx_jm1] & 0x10) ? 1.0 : 0.0);

            double add = (eps_E * (p[idx_ip1] - p[idx]) -
                         eps_W * (p[idx] - p[idx_im1])) * rdx2  +
                        (eps_N * (p[idx_jp1] - p[idx]) -
                         eps_S * (p[idx] - p[idx_jm1])) * rdy2  -  rhs[idx];
            local_sum = add * add;
        }
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int block_id = blockIdx.y * gridDim.x + blockIdx.x;
        residual_array[block_id] = sdata[0];
    }
}

/**
 * @brief CUDA kernel to compute p0 normalization factor
 */
__global__ void compute_p0_kernel(double *p, char *flag, double *p0_array,
                                   int imax, int jmax, int width) {
    __shared__ double sdata[256];

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double local_sum = 0.0;

    if (i <= imax && j <= jmax) {
        int idx = IDX(i, j, width);
        if (flag[idx] & 0x10) {  // C_F
            local_sum = p[idx] * p[idx];
        }
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int block_id = blockIdx.y * gridDim.x + blockIdx.x;
        p0_array[block_id] = sdata[0];
    }
}

/**
 * @brief CUDA kernel to update velocity based on pressure gradient
 */
__global__ void update_velocity_kernel(double *u, double *v, double *p,
                                        double *f, double *g, char *flag,
                                        double del_t, double delx, double dely,
                                        int imax, int jmax, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i >= 1 && i < imax-2 && j >= 1 && j < jmax-1) {
        int idx = IDX(i, j, width);
        int idx_ip1 = IDX(i+1, j, width);

        if ((flag[idx] & 0x10) && (flag[idx_ip1] & 0x10)) {
            u[idx] = f[idx] - (p[idx_ip1] - p[idx]) * del_t / delx;
        }
    }

    if (i >= 1 && i < imax-1 && j >= 1 && j < jmax-2) {
        int idx = IDX(i, j, width);
        int idx_jp1 = IDX(i, j+1, width);

        if ((flag[idx] & 0x10) && (flag[idx_jp1] & 0x10)) {
            v[idx] = g[idx] - (p[idx_jp1] - p[idx]) * del_t / dely;
        }
    }
}

/**
 * @brief CUDA kernel to find maximum u velocity (for CFL)
 */
__global__ void find_umax_kernel(double *u, double *umax_array,
                                  int imax, int jmax, int width) {
    __shared__ double sdata[256];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double local_max = 0.0;

    if (i < imax+2 && j < jmax+2) {
        local_max = fabs(u[IDX(i, j, width)]);
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        int block_id = blockIdx.y * gridDim.x + blockIdx.x;
        umax_array[block_id] = sdata[0];
    }
}

/**
 * @brief CUDA kernel to find maximum v velocity (for CFL)
 */
__global__ void find_vmax_kernel(double *v, double *vmax_array,
                                  int imax, int jmax, int width) {
    __shared__ double sdata[256];

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double local_max = 0.0;

    if (i < imax+2 && j < jmax+2) {
        local_max = fabs(v[IDX(i, j, width)]);
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        int block_id = blockIdx.y * gridDim.x + blockIdx.x;
        vmax_array[block_id] = sdata[0];
    }
}

/**
 * @brief CUDA kernel to apply boundary conditions
 */
__global__ void apply_boundary_conditions_kernel(double *u, double *v, char *flag,
                                                  double ui, double vi,
                                                  int imax, int jmax, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < jmax+2) {
        int j = idx;
        u[IDX(0, j, width)] = u[IDX(1, j, width)];
        v[IDX(0, j, width)] = v[IDX(1, j, width)];

        u[IDX(imax, j, width)] = u[IDX(imax-1, j, width)];
        v[IDX(imax+1, j, width)] = v[IDX(imax, j, width)];
    }

    if (idx < imax+2) {
        int i = idx;
        v[IDX(i, jmax, width)] = 0.0;
        u[IDX(i, jmax+1, width)] = u[IDX(i, jmax, width)];

        v[IDX(i, 0, width)] = 0.0;
        u[IDX(i, 0, width)] = u[IDX(i, 1, width)];
    }
}

/**
 * @brief CUDA kernel to apply no-slip boundary conditions around obstacles
 */
__global__ void apply_obstacle_boundary_kernel(double *u, double *v, char *flag,
                                                int imax, int jmax, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= imax && j <= jmax) {
        int idx = IDX(i, j, width);
        char f = flag[idx];

        if (f & 0x0F) {  // B_NSEW bits set
            int idx_im1 = IDX(i-1, j, width);
            int idx_ip1 = IDX(i+1, j, width);
            int idx_jm1 = IDX(i, j-1, width);
            int idx_jp1 = IDX(i, j+1, width);
            int idx_im1_jm1 = IDX(i-1, j-1, width);
            int idx_im1_jp1 = IDX(i-1, j+1, width);
            int idx_ip1_jm1 = IDX(i+1, j-1, width);

            switch(f) {
                case 0x01: // B_N
                    v[idx] = 0.0;
                    u[idx] = -u[idx_jp1];
                    u[idx_im1] = -u[idx_im1_jp1];
                    break;
                case 0x08: // B_E
                    u[idx] = 0.0;
                    v[idx] = -v[idx_ip1];
                    v[idx_jm1] = -v[idx_ip1_jm1];
                    break;
                case 0x02: // B_S
                    v[idx_jm1] = 0.0;
                    u[idx] = -u[idx_jm1];
                    u[idx_im1] = -u[idx_im1_jm1];
                    break;
                case 0x04: // B_W
                    u[idx_im1] = 0.0;
                    v[idx] = -v[idx_im1];
                    v[idx_jm1] = -v[idx_im1_jm1];
                    break;
                case 0x09: // B_NE
                    v[idx] = 0.0;
                    u[idx] = 0.0;
                    v[idx_jm1] = -v[idx_ip1_jm1];
                    u[idx_im1] = -u[idx_im1_jp1];
                    break;
                case 0x0A: // B_SE
                    v[idx_jm1] = 0.0;
                    u[idx] = 0.0;
                    v[idx] = -v[idx_ip1];
                    u[idx_im1] = -u[idx_im1_jm1];
                    break;
                case 0x06: // B_SW
                    v[idx_jm1] = 0.0;
                    u[idx_im1] = 0.0;
                    v[idx] = -v[idx_im1];
                    u[idx] = -u[idx_jm1];
                    break;
                case 0x05: // B_NW
                    v[idx] = 0.0;
                    u[idx_im1] = 0.0;
                    v[idx_jm1] = -v[idx_im1_jm1];
                    u[idx] = -u[idx_jp1];
                    break;
            }
        }
    }
}

/**
 * @brief CUDA kernel to set inflow boundary conditions
 */
__global__ void apply_inflow_boundary_kernel(double *u, double *v,
                                              double ui, double vi,
                                              int jmax, int width) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j == 0) {
        v[IDX(0, 0, width)] = 2.0 * vi - v[IDX(1, 0, width)];
    }

    if (j >= 1 && j <= jmax) {
        u[IDX(0, j, width)] = ui;
        v[IDX(0, j, width)] = 2.0 * vi - v[IDX(1, j, width)];
    }
}


/**
 * @brief Compute tentative velocity field (f, g)
 */
void compute_tentative_velocity() {
    dim3 block(16, 16);  // 256 threads per block
    dim3 grid_f((imax + block.x - 1) / block.x, (jmax + block.y - 1) / block.y);
    dim3 grid_g((imax + block.x - 1) / block.x, (jmax + block.y - 1) / block.y);

    compute_f_kernel<<<grid_f, block>>>(d_u, d_v, d_f, d_flag,
                                         del_t, delx, dely, Re, y,
                                         imax, jmax, grid_width);

    compute_g_kernel<<<grid_g, block>>>(d_u, d_v, d_g, d_flag,
                                         del_t, delx, dely, Re, y,
                                         imax, jmax, grid_width);

    int boundary_threads = fmax(imax, jmax) + 1;
    int boundary_blocks = (boundary_threads + 255) / 256;
    set_fg_boundaries_kernel<<<boundary_blocks, 256>>>(d_u, d_v, d_f, d_g,
                                                        imax, jmax, grid_width);

    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Calculate the right hand side of the pressure equation
 */
void compute_rhs() {
    dim3 block(16, 16);
    dim3 grid((imax + block.x - 1) / block.x, (jmax + block.y - 1) / block.y);

    compute_rhs_kernel<<<grid, block>>>(d_f, d_g, d_rhs, d_flag,
                                         del_t, delx, dely,
                                         imax, jmax, grid_width);

    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Red/Black SOR to solve the Poisson equation
 */
double poisson() {
    double rdx2 = 1.0 / (delx * delx);
    double rdy2 = 1.0 / (dely * dely);

    dim3 block(16, 16);
    dim3 grid((imax + block.x - 1) / block.x, (jmax + block.y - 1) / block.y);
    int num_blocks = grid.x * grid.y;

    compute_p0_kernel<<<grid, block>>>(d_p, d_flag, d_residual_array,
                                        imax, jmax, grid_width);

    double *h_partial = (double*)malloc(num_blocks * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_partial, d_residual_array, num_blocks * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double p0 = 0.0;
    for (int i = 0; i < num_blocks; i++) {
        p0 += h_partial[i];
    }
    p0 = sqrt(p0 / fluid_cells);
    if (p0 < 0.0001) p0 = 1.0;

    double res = 0.0;
    int iter;
    for (iter = 0; iter < itermax; iter++) {
        sor_red_kernel<<<grid, block>>>(d_p, d_rhs, d_flag,
                                         rdx2, rdy2, omega,
                                         imax, jmax, grid_width);

        sor_black_kernel<<<grid, block>>>(d_p, d_rhs, d_flag,
                                           rdx2, rdy2, omega,
                                           imax, jmax, grid_width);

        if (iter % 10 == 0) {
            compute_residual_kernel<<<grid, block>>>(d_p, d_rhs, d_flag,
                                                      rdx2, rdy2, d_residual_array,
                                                      imax, jmax, grid_width);

            CUDA_CHECK(cudaMemcpy(h_partial, d_residual_array,
                                  num_blocks * sizeof(double),
                                  cudaMemcpyDeviceToHost));

            res = 0.0;
            for (int i = 0; i < num_blocks; i++) {
                res += h_partial[i];
            }
            res = sqrt(res / fluid_cells) / p0;

            if (res < eps) break;
        }
    }

    if (iter % 10 != 0) {
        compute_residual_kernel<<<grid, block>>>(d_p, d_rhs, d_flag,
                                                  rdx2, rdy2, d_residual_array,
                                                  imax, jmax, grid_width);

        CUDA_CHECK(cudaMemcpy(h_partial, d_residual_array,
                              num_blocks * sizeof(double),
                              cudaMemcpyDeviceToHost));

        res = 0.0;
        for (int i = 0; i < num_blocks; i++) {
            res += h_partial[i];
        }
        res = sqrt(res / fluid_cells) / p0;
    }

    free(h_partial);
    return res;
}

/**
 * @brief Update the velocity values based on the tentative velocity values and the new pressure matrix
 */
void update_velocity() {
    dim3 block(16, 16);
    dim3 grid((imax + block.x - 1) / block.x, (jmax + block.y - 1) / block.y);

    update_velocity_kernel<<<grid, block>>>(d_u, d_v, d_p, d_f, d_g, d_flag,
                                             del_t, delx, dely,
                                             imax, jmax, grid_width);

    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Set the timestep size so that we satisfy the CFL conditions. Otherwise the simulation becomes unstable.
 */
void set_timestep_interval() {
    if (tau >= 1.0e-10) {
        dim3 block(16, 16);
        dim3 grid_u((imax + 2 + block.x - 1) / block.x,
                    (jmax + 2 + block.y - 1) / block.y);
        dim3 grid_v((imax + 2 + block.x - 1) / block.x,
                    (jmax + 2 + block.y - 1) / block.y);
        int num_blocks = grid_u.x * grid_u.y;

        find_umax_kernel<<<grid_u, block>>>(d_u, d_umax_array, imax, jmax, grid_width);
        find_vmax_kernel<<<grid_v, block>>>(d_v, d_vmax_array, imax, jmax, grid_width);

        double *h_umax = (double*)malloc(num_blocks * sizeof(double));
        double *h_vmax = (double*)malloc(num_blocks * sizeof(double));

        CUDA_CHECK(cudaMemcpy(h_umax, d_umax_array, num_blocks * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vmax, d_vmax_array, num_blocks * sizeof(double),
                              cudaMemcpyDeviceToHost));

        double umax = 1.0e-10;
        double vmax = 1.0e-10;
        for (int i = 0; i < num_blocks; i++) {
            umax = fmax(umax, h_umax[i]);
            vmax = fmax(vmax, h_vmax[i]);
        }

        double deltu = delx / umax;
        double deltv = dely / vmax;
        double deltRe = 1.0 / (1.0 / (delx * delx) + 1.0 / (dely * dely)) * Re / 2.0;

        del_t = fmin(fmin(deltu, deltv), deltRe) * tau;

        free(h_umax);
        free(h_vmax);
    }
}

/**
 * @brief Apply boundary conditions on GPU
 */
void apply_boundary_conditions_gpu() {
    int max_boundary = fmax(imax, jmax) + 2;
    int threads_1d = 256;
    int blocks_1d = (max_boundary + threads_1d - 1) / threads_1d;

    apply_boundary_conditions_kernel<<<blocks_1d, threads_1d>>>(
        d_u, d_v, d_flag, ui, vi, imax, jmax, grid_width);

    dim3 block(16, 16);
    dim3 grid((imax + block.x - 1) / block.x, (jmax + block.y - 1) / block.y);

    apply_obstacle_boundary_kernel<<<grid, block>>>(
        d_u, d_v, d_flag, imax, jmax, grid_width);

    int blocks_inflow = (jmax + 1 + threads_1d - 1) / threads_1d;
    apply_inflow_boundary_kernel<<<blocks_inflow, threads_1d>>>(
        d_u, d_v, ui, vi, jmax, grid_width);

    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief The main routine that sets up the problem and executes the solving routines
 */
int main(int argc, char *argv[]) {
    set_defaults();
    parse_args(argc, argv);
    setup();

    if (verbose) print_opts();

    allocate_arrays();
    problem_set_up();

    linearize_2d_arrays();
    allocate_device_memory();
    copy_to_device();

    double res;

    int iters = 0;
    double t;
    for (t = 0.0; t < t_end; t += del_t, iters++) {
        if (!fixed_dt)
            set_timestep_interval();

        compute_tentative_velocity();
        compute_rhs();
        res = poisson();
        update_velocity();

        apply_boundary_conditions_gpu();

        if ((iters % output_freq == 0)) {
            printf("Step %8d, Time: %14.8e (del_t: %14.8e), Residual: %14.8e\n",
                   iters, t+del_t, del_t, res);

            if ((!no_output) && (enable_checkpoints)) {
                copy_from_device();
                delinearize_2d_arrays();
                write_checkpoint(iters, t+del_t);
            }
        }
    }

    printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", iters, t, res);
    printf("Simulation complete.\n");

    copy_from_device();
    delinearize_2d_arrays();

    if (!no_output)
        write_result(iters, t);

    free_device_memory();
    free_host_flat_arrays();
    free_arrays();

    return 0;
}
