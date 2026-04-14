#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <mpi.h>

#include "data.h"
#include "vtk.h"
#include "setup.h"
#include "boundary.h"
#include "args.h"

int rank = 0;
int nprocs = 1;
int i_start = 1;
int i_end = 0;
int local_imax = 0;

void exchange_halos(double **array, int size_x, int size_y) {
    MPI_Status status;

    if (rank < nprocs - 1) {
        MPI_Sendrecv(array[i_end], size_y, MPI_DOUBLE, rank+1, 0,
                     array[i_end+1], size_y, MPI_DOUBLE, rank+1, 1,
                     MPI_COMM_WORLD, &status);
    }

    if (rank > 0) {
        MPI_Sendrecv(array[i_start], size_y, MPI_DOUBLE, rank-1, 1,
                     array[i_start-1], size_y, MPI_DOUBLE, rank-1, 0,
                     MPI_COMM_WORLD, &status);
    }
}

void exchange_flag_halos(char **array, int size_x, int size_y) {
    MPI_Status status;

    if (rank < nprocs - 1) {
        MPI_Sendrecv(array[i_end], size_y, MPI_CHAR, rank+1, 0,
                     array[i_end+1], size_y, MPI_CHAR, rank+1, 1,
                     MPI_COMM_WORLD, &status);
    }

    if (rank > 0) {
        MPI_Sendrecv(array[i_start], size_y, MPI_CHAR, rank-1, 1,
                     array[i_start-1], size_y, MPI_CHAR, rank-1, 0,
                     MPI_COMM_WORLD, &status);
    }
}

void gather_vtk_data() {

    if (rank == 0) {
        for (int src_rank = 1; src_rank < nprocs; src_rank++) {
            int src_local_imax = imax / nprocs;
            int remainder = imax % nprocs;
            int src_i_start, src_i_end;

            if (src_rank < remainder) {
                src_local_imax++;
                src_i_start = src_rank * src_local_imax + 1;
            } else {
                src_i_start = src_rank * src_local_imax + remainder + 1;
            }
            src_i_end = src_i_start + src_local_imax - 1;

            for (int i = src_i_start; i <= src_i_end; i++) {
                MPI_Recv(u[i], jmax+2, MPI_DOUBLE, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(v[i], jmax+2, MPI_DOUBLE, src_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(p[i], jmax+2, MPI_DOUBLE, src_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (int i = i_start; i <= i_end; i++) {
            MPI_Send(u[i], jmax+2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(v[i], jmax+2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            MPI_Send(p[i], jmax+2, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        }
    }
}

void compute_tentative_velocity() {
    static int first_call = 1;

    exchange_halos(u, u_size_x, u_size_y);
    exchange_halos(v, v_size_x, v_size_y);

    int i_start_f = (i_start > 1) ? i_start : 1;
    int i_end_f = i_end;  // Compute f for all owned cells
    if (rank == nprocs - 1) {
        i_end_f = imax - 1;  // Last rank stops at imax-1
    }

    for (int i = i_start_f; i <= i_end_f; i++) {
        for (int j = 1; j < jmax+1; j++) {
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                double du2dx = ((u[i][j] + u[i+1][j]) * (u[i][j] + u[i+1][j]) +
                                y * fabs(u[i][j] + u[i+1][j]) * (u[i][j] - u[i+1][j]) -
                                (u[i-1][j] + u[i][j]) * (u[i-1][j] + u[i][j]) -
                                y * fabs(u[i-1][j] + u[i][j]) * (u[i-1][j]-u[i][j]))
                                / (4.0 * delx);
                double duvdy = ((v[i][j] + v[i+1][j]) * (u[i][j] + u[i][j+1]) +
                                y * fabs(v[i][j] + v[i+1][j]) * (u[i][j] - u[i][j+1]) -
                                (v[i][j-1] + v[i+1][j-1]) * (u[i][j-1] + u[i][j]) -
                                y * fabs(v[i][j-1] + v[i+1][j-1]) * (u[i][j-1] - u[i][j]))
                                / (4.0 * dely);
                double laplu = (u[i+1][j] - 2.0 * u[i][j] + u[i-1][j]) / delx / delx +
                                (u[i][j+1] - 2.0 * u[i][j] + u[i][j-1]) / dely / dely;

                f[i][j] = u[i][j] + del_t * (laplu / Re - du2dx - duvdy);

                if (rank == 0 && first_call && i == 128 && j == 124) {
                    printf("Rank 0 computing f[128][124] in loop: result=%.6e\n", f[i][j]);
                    printf("  u x-direction: u[i-1][j]=%.6e, u[i][j]=%.6e, u[i+1][j]=%.6e\n", u[i-1][j], u[i][j], u[i+1][j]);
                    printf("  u y-direction: u[i][j-1]=%.6e, u[i][j]=%.6e, u[i][j+1]=%.6e\n", u[i][j-1], u[i][j], u[i][j+1]);
                    printf("  v values: v[i][j]=%.6e, v[i+1][j]=%.6e\n", v[i][j], v[i+1][j]);
                    printf("  Parameters: del_t=%.6e, Re=%.6e, delx=%.6e, dely=%.6e\n", del_t, Re, delx, dely);
                    printf("  du2dx=%.6e, duvdy=%.6e, laplu=%.6e\n", du2dx, duvdy, laplu);
                }
            } else {
                f[i][j] = u[i][j];
            }
        }
    }

    for (int i = i_start; i <= i_end; i++) {
        for (int j = 1; j < jmax; j++) {
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                double duvdx = ((u[i][j] + u[i][j+1]) * (v[i][j] + v[i+1][j]) +
                                y * fabs(u[i][j] + u[i][j+1]) * (v[i][j] - v[i+1][j]) -
                                (u[i-1][j] + u[i-1][j+1]) * (v[i-1][j] + v[i][j]) -
                                y * fabs(u[i-1][j] + u[i-1][j+1]) * (v[i-1][j]-v[i][j]))
                                / (4.0 * delx);
                double dv2dy = ((v[i][j] + v[i][j+1]) * (v[i][j] + v[i][j+1]) +
                                y * fabs(v[i][j] + v[i][j+1]) * (v[i][j] - v[i][j+1]) -
                                (v[i][j-1] + v[i][j]) * (v[i][j-1] + v[i][j]) -
                                y * fabs(v[i][j-1] + v[i][j]) * (v[i][j-1] - v[i][j]))
                                / (4.0 * dely);
                double laplv = (v[i+1][j] - 2.0 * v[i][j] + v[i-1][j]) / delx / delx +
                                (v[i][j+1] - 2.0 * v[i][j] + v[i][j-1]) / dely / dely;

                g[i][j] = v[i][j] + del_t * (laplv / Re - duvdx - dv2dy);
            } else {
                g[i][j] = v[i][j];
            }
        }
    }

    if (rank == 0) {
        for (int j = 1; j < jmax+1; j++) {
            f[0][j] = u[0][j];
        }
    }

    if (rank == nprocs-1) {
        for (int j = 1; j < jmax+1; j++) {
            f[imax][j] = u[imax][j];
        }
    }

    for (int i = i_start; i <= i_end; i++) {
        g[i][0] = v[i][0];
        g[i][jmax] = v[i][jmax];
    }

    exchange_halos(f, f_size_x, f_size_y);
    exchange_halos(g, g_size_x, g_size_y);

    if (first_call && rank == 0) {
        int i = 128, j = 124;
        printf("DEBUG Rank 0 AFTER compute loop at [128][124]:\n");
        printf("  flag[128][124]=%d, flag[129][124]=%d\n", (int)flag[i][j], (int)flag[i+1][j]);
        printf("  u[127][124]=%.6e, u[128][124]=%.6e, u[129][124]=%.6e\n", u[i-1][j], u[i][j], u[i+1][j]);
        printf("  f[128][124]=%.6e (should be ~0.878 per serial)\n", f[i][j]);
        printf("  Loop bounds: i_start_f=%d, i_end_f=%d\n", i_start_f, i_end_f);
        fflush(stdout);
        first_call = 0;
    }
}


void compute_rhs() {
    double rhs_sum_local = 0.0;
    int rhs_count_local = 0;
    double rhs_max_local = 0.0;
    double max_f_diff = 0.0, max_g_diff = 0.0;
    int max_i = i_start, max_j = 1;

    for (int i = i_start; i <= i_end; i++) {
        for (int j = 1; j < jmax+1; j++) {
            if (flag[i][j] & C_F) {
                double f_diff = (f[i][j] - f[i-1][j]) / delx;
                double g_diff = (g[i][j] - g[i][j-1]) / dely;
                rhs[i][j] = (f_diff + g_diff) / del_t;
                double rhs_abs = fabs(rhs[i][j]);
                if (rhs_abs > rhs_max_local) {
                    rhs_max_local = rhs_abs;
                    max_f_diff = f_diff;
                    max_g_diff = g_diff;
                    max_i = i;
                    max_j = j;
                }
                rhs_sum_local += rhs[i][j] * rhs[i][j];
                rhs_count_local++;
            }
        }
    }

    if (rank == 1) {
        static int first = 1;
        if (first) {
            int i = 129, j = 124;
            if (flag[i][j] & C_F) {
                double f_diff = (f[i][j] - f[i-1][j]) / delx;
                double g_diff = (g[i][j] - g[i][j-1]) / dely;
                double rhs_val = (f_diff + g_diff) / del_t;
                printf("Rank 1 RHS at [129][124]: flag=%d, f_diff=%.6e, g_diff=%.6e, rhs=%.6e\n",
                       (int)flag[i][j], f_diff, g_diff, rhs_val);
            } else {
                printf("Rank 1 at [129][124]: flag=%d (not fluid, RHS not computed)\n", (int)flag[i][j]);
            }
            printf("Rank 1 max RHS location: i=%d, j=%d, f_diff=%.6e, g_diff=%.6e, rhs_max=%.6e\n",
                   max_i, max_j, max_f_diff, max_g_diff, rhs_max_local);
            fflush(stdout);
            first = 0;
        }
    }

    static int first_call = 1;
    if (first_call) {
        double rhs_sum_global = 0.0;
        MPI_Allreduce(&rhs_sum_local, &rhs_sum_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for (int r = 0; r < nprocs; r++) {
            if (rank == r) {
                printf("DEBUG Rank %d RHS: sum_local=%.6e, count=%d, max_local=%.6e\n",
                       rank, rhs_sum_local, rhs_count_local, rhs_max_local);
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (rank == 0) {
            printf("DEBUG RHS sum_global=%.6e\n", rhs_sum_global);
            fflush(stdout);
        }
        first_call = 0;
    }
}


double poisson() {
    double rdx2 = 1.0 / (delx * delx);
    double rdy2 = 1.0 / (dely * dely);
    double beta_2 = -omega / (2.0 * (rdx2 + rdy2));

    double p0_local = 0.0;
    for (int i = i_start; i <= i_end; i++) {
        for (int j = 1; j < jmax+1; j++) {
            if (flag[i][j] & C_F) { p0_local += p[i][j] * p[i][j]; }
        }
    }

    double p0 = 0.0;
    MPI_Allreduce(&p0_local, &p0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    p0 = sqrt(p0 / fluid_cells);
    if (p0 < 0.0001) { p0 = 1.0; }

    static int first_call = 1;
    if (first_call && (rank == 0 || rank == 7)) {
        printf("DEBUG Rank %d: p0_local=%.6e, p0=%.6e, p0_after=%.6e\n", rank, p0_local, p0, p0);
        fflush(stdout);
        first_call = 0;
    }

    int iter;
    double res = 0.0;
    for (iter = 0; iter < itermax; iter++) {
        for (int rb = 0; rb < 2; rb++) {
            for (int i = i_start; i <= i_end; i++) {
                for (int j = 1; j < jmax+1; j++) {
                    if ((i + j) % 2 != rb) { continue; }
                    if (flag[i][j] == (C_F | B_NSEW)) {
                        p[i][j] = (1.0 - omega) * p[i][j] -
                              beta_2 * ((p[i+1][j] + p[i-1][j]) * rdx2
                                  + (p[i][j+1] + p[i][j-1]) * rdy2
                                  - rhs[i][j]);
                    } else if (flag[i][j] & C_F) {
                        double eps_E = ((flag[i+1][j] & C_F) ? 1.0 : 0.0);
                        double eps_W = ((flag[i-1][j] & C_F) ? 1.0 : 0.0);
                        double eps_N = ((flag[i][j+1] & C_F) ? 1.0 : 0.0);
                        double eps_S = ((flag[i][j-1] & C_F) ? 1.0 : 0.0);

                        double beta_mod = -omega / ((eps_E + eps_W) * rdx2 + (eps_N + eps_S) * rdy2);
                        p[i][j] = (1.0 - omega) * p[i][j] -
                            beta_mod * ((eps_E * p[i+1][j] + eps_W * p[i-1][j]) * rdx2
                                + (eps_N * p[i][j+1] + eps_S * p[i][j-1]) * rdy2
                                - rhs[i][j]);
                    }
                }
            }
            exchange_halos(p, p_size_x, p_size_y);
        }

        res = 0.0;
        double res_local = 0.0;
        for (int i = i_start; i <= i_end; i++) {
            for (int j = 1; j < jmax+1; j++) {
                if (flag[i][j] & C_F) {
                    double eps_E = ((flag[i+1][j] & C_F) ? 1.0 : 0.0);
                    double eps_W = ((flag[i-1][j] & C_F) ? 1.0 : 0.0);
                    double eps_N = ((flag[i][j+1] & C_F) ? 1.0 : 0.0);
                    double eps_S = ((flag[i][j-1] & C_F) ? 1.0 : 0.0);

                    double add = (eps_E * (p[i+1][j] - p[i][j]) -
                        eps_W * (p[i][j] - p[i-1][j])) * rdx2 +
                        (eps_N * (p[i][j+1] - p[i][j]) -
                        eps_S * (p[i][j] - p[i][j-1])) * rdy2 - rhs[i][j];
                    res_local += add * add;
                }
            }
        }

        MPI_Allreduce(&res_local, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        res = sqrt(res / fluid_cells) / p0;

        if (res < eps) break;
    }

    return res;
}


void update_velocity() {
    int i_start_u = (i_start > 1) ? i_start : 1;
    int i_end_u = i_end;
    if (rank == nprocs - 1 && i_end_u >= imax - 2) {
        i_end_u = imax - 3;
    }

    for (int i = i_start_u; i <= i_end_u; i++) {
        for (int j = 1; j < jmax-1; j++) {
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                u[i][j] = f[i][j] - (p[i+1][j] - p[i][j]) * del_t / delx;
            }
        }
    }

    int i_start_v = (i_start > 1) ? i_start : 1;
    int i_end_v = i_end;
    if (rank == nprocs - 1 && i_end_v >= imax - 1) {
        i_end_v = imax - 2;
    }

    for (int i = i_start_v; i <= i_end_v; i++) {
        for (int j = 1; j < jmax-2; j++) {
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                v[i][j] = g[i][j] - (p[i][j+1] - p[i][j]) * del_t / dely;
            }
        }
    }
}


void set_timestep_interval() {
    if (tau >= 1.0e-10) {
        double umax_local = 1.0e-10;
        double vmax_local = 1.0e-10;

        for (int i = i_start-1; i <= i_end+1; i++) {
            for (int j = 1; j < jmax+2; j++) {
                umax_local = fmax(fabs(u[i][j]), umax_local);
            }
        }

        for (int i = i_start; i <= i_end; i++) {
            for (int j = 0; j < jmax+2; j++) {
                vmax_local = fmax(fabs(v[i][j]), vmax_local);
            }
        }

        double umax, vmax;
        MPI_Allreduce(&umax_local, &umax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&vmax_local, &vmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        double deltu = delx / umax;
        double deltv = dely / vmax;
        double deltRe = 1.0 / (1.0 / (delx * delx) + 1 / (dely * dely)) * Re / 2.0;

        if (deltu < deltv) {
            del_t = fmin(deltu, deltRe);
        } else {
            del_t = fmin(deltv, deltRe);
        }
        del_t = tau * del_t;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    set_defaults();
    parse_args(argc, argv);
    setup();

    local_imax = imax / nprocs;
    int remainder = imax % nprocs;

    if (rank < remainder) {
        local_imax++;
        i_start = rank * local_imax + 1;
    } else {
        i_start = rank * local_imax + remainder + 1;
    }
    i_end = i_start + local_imax - 1;

    if (rank == 0 && verbose) print_opts();

    if (verbose) {
        printf("Rank %d: local_imax=%d, i_start=%d, i_end=%d\n", rank, local_imax, i_start, i_end);
        fflush(stdout);
    }

    allocate_arrays();
    problem_set_up();

    exchange_halos(u, u_size_x, u_size_y);
    exchange_halos(v, v_size_x, v_size_y);
    exchange_halos(p, p_size_x, p_size_y);
    exchange_flag_halos(flag, flag_size_x, flag_size_y);

    if ((rank == 0 || rank == 1) && verbose) {
        if (rank == 0) {
            printf("Rank 0 AFTER flag exchange: flag[128][124]=%d, flag[129][124]=%d, flag[130][124]=%d\n",
                   (int)flag[128][124], (int)flag[129][124], (int)flag[130][124]);
        } else {
            printf("Rank 1 AFTER flag exchange: flag[128][124]=%d, flag[129][124]=%d, flag[130][124]=%d\n",
                   (int)flag[128][124], (int)flag[129][124], (int)flag[130][124]);
        }
        fflush(stdout);
    }

    apply_boundary_conditions();

    exchange_halos(u, u_size_x, u_size_y);
    exchange_halos(v, v_size_x, v_size_y);

    double res;
    double t_start = MPI_Wtime();

    int iters = 0;
    double t;
    for (t = 0.0; t < t_end; t += del_t, iters++) {
        if (!fixed_dt)
            set_timestep_interval();

        compute_tentative_velocity();

        compute_rhs();

        res = poisson();

        update_velocity();

        apply_boundary_conditions();

        exchange_halos(u, u_size_x, u_size_y);
        exchange_halos(v, v_size_x, v_size_y);

        if (iters % output_freq == 0) {
            if (rank == 0) {
                printf("Step %8d, Time: %14.8e (del_t: %14.8e), Residual: %14.8e\n", iters, t+del_t, del_t, res);
            }

            if ((!no_output) && (enable_checkpoints)) {
                gather_vtk_data();
                if (rank == 0) {
                    write_checkpoint(iters, t+del_t);
                }
            }
        }
    }

    double t_end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", iters, t, res);
        printf("Simulation complete.\n");
        printf("Total wall time: %.6f seconds\n", t_end_time - t_start);
    }

    if (!no_output) {
        gather_vtk_data();
        if (rank == 0) {
            write_result(iters, t);
        }
    }

    free_arrays();

    MPI_Finalize();
    return 0;
}

