#include "data.h"
#include "boundary.h"

extern int rank, nprocs, i_start, i_end;

void apply_boundary_conditions() {
    if (rank == 0) {
        for (int j = 0; j < jmax+2; j++) {
            u[0][j] = u[1][j];
            v[0][j] = v[1][j];
        }
    }

    if (rank == nprocs-1) {
        for (int j = 0; j < jmax+2; j++) {
            u[imax][j] = u[imax-1][j];
            v[imax+1][j] = v[imax][j];
        }
    }

    for (int i = i_start; i <= i_end; i++) {
        v[i][jmax] = 0.0;
        u[i][jmax+1] = u[i][jmax];

        v[i][0] = 0.0;
        u[i][0] = u[i][1];
    }

    // Handle obstacle boundary conditions
    // NOTE: In MPI, we must be careful not to modify cells outside our domain
    // Boundary conditions that set u[i-1] can only be applied if i > i_start
    // Otherwise, the left neighbor rank is responsible for that cell
    for (int i = i_start; i <= i_end; i++) {
        for (int j = 1; j < jmax+1; j++) {
            if (flag[i][j] & B_NSEW) {
                switch (flag[i][j]) {
                    case B_N:
                        v[i][j]   = 0.0;
                        u[i][j]   = -u[i][j+1];
                        if (i >= i_start && (rank > 0 || i > 0)) {
                            u[i-1][j] = -u[i-1][j+1];
                        }
                        break;
                    case B_E:
                        u[i][j]   = 0.0;
                        v[i][j]   = -v[i+1][j];
                        v[i][j-1] = -v[i+1][j-1];
                        break;
                    case B_S:
                        v[i][j-1] = 0.0;
                        u[i][j]   = -u[i][j-1];
                        if (i >= i_start && (rank > 0 || i > 0)) {
                            u[i-1][j] = -u[i-1][j-1];
                        }
                        break;
                    case B_W:
                        if (i >= i_start && (rank > 0 || i > 0)) {
                            u[i-1][j] = 0.0;
                        }
                        v[i][j]   = -v[i-1][j];
                        v[i][j-1] = -v[i-1][j-1];
                        break;
                    case B_NE:
                        v[i][j]   = 0.0;
                        u[i][j]   = 0.0;
                        v[i][j-1] = -v[i+1][j-1];
                        if (i >= i_start && (rank > 0 || i > 0)) {
                            u[i-1][j] = -u[i-1][j+1];
                        }
                        break;
                    case B_SE:
                        v[i][j-1] = 0.0;
                        u[i][j]   = 0.0;
                        v[i][j]   = -v[i+1][j];
                        if (i >= i_start && (rank > 0 || i > 0)) {
                            u[i-1][j] = -u[i-1][j-1];
                        }
                        break;
                    case B_SW:
                        v[i][j-1] = 0.0;
                        if (i >= i_start && (rank > 0 || i > 0)) {
                            u[i-1][j] = 0.0;
                        }
                        v[i][j]   = -v[i-1][j];
                        u[i][j]   = -u[i][j-1];
                        break;
                    case B_NW:
                        v[i][j]   = 0.0;
                        if (i >= i_start && (rank > 0 || i > 0)) {
                            u[i-1][j] = 0.0;
                        }
                        v[i][j-1] = -v[i-1][j-1];
                        u[i][j]   = -u[i][j+1];
                        break;
                }
            }
        }
    }

    // CRITICAL FIX: Handle boundary conditions at domain boundaries
    // If flag[i_end+1] has an obstacle, it may affect u[i_end] which is owned by this rank
    // but rank+1 cannot modify it (halos are receive-only)
    if (rank < nprocs - 1) {
        int i = i_end + 1;  // Right halo
        for (int j = 1; j < jmax+1; j++) {
            if (flag[i][j] & B_W) {
                // West boundary of obstacle at i affects u[i-1][j]
                u[i-1][j] = 0.0;
            }
            if (flag[i][j] & B_SW) {
                u[i-1][j] = 0.0;
            }
            if (flag[i][j] & B_NW) {
                u[i-1][j] = 0.0;
            }
            // For B_N, B_S, B_NE, B_SE: check the formula in the main loop
            if (flag[i][j] == B_N) {
                u[i-1][j] = -u[i-1][j+1];
            }
            if (flag[i][j] == B_S) {
                u[i-1][j] = -u[i-1][j-1];
            }
            if (flag[i][j] == B_NE) {
                u[i-1][j] = -u[i-1][j+1];
            }
            if (flag[i][j] == B_SE) {
                u[i-1][j] = -u[i-1][j-1];
            }
        }
    }

    if (rank == 0) {
        v[0][0] = 2 * vi - v[1][0];
        for (int j = 1; j < jmax+1; j++) {
            u[0][j] = ui;
            v[0][j] = 2 * vi - v[1][j];
        }
    }
}