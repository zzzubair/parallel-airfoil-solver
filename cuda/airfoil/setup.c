#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "data.h"
#include "vtk.h"
#include "boundary.h"

/**
 * @brief Set up some default values before arguments are parsed.
 * 
 */
void set_defaults() {
	set_default_base();
}

/**
 * @brief Set up some values after arguments have been parsed.
 * 
 */
void setup() {
	delx = xlength/imax;
    dely = ylength/jmax;
}
 
/**
 * @brief Allocate all of the arrays used by the computation.
 * 
 */
void allocate_arrays() {
	    /* Allocate arrays */
    u_size_x = imax+2; u_size_y = jmax+2;
    u = alloc_2d_array(u_size_x, u_size_y);
    v_size_x = imax+2; v_size_y = jmax+2;
    v = alloc_2d_array(v_size_x, v_size_y);
    f_size_x = imax+2; f_size_y = jmax+2;
    f = alloc_2d_array(f_size_x, f_size_y);
    g_size_x = imax+2; g_size_y = jmax+2;
    g = alloc_2d_array(g_size_x, g_size_y);
    p_size_x = imax+2; p_size_y = jmax+2;
    p = alloc_2d_array(p_size_x, p_size_y);
    rhs_size_x = imax+2; rhs_size_y = jmax+2;
    rhs = alloc_2d_array(rhs_size_x, rhs_size_y);
    flag_size_x = imax+2; flag_size_y = jmax+2;
    flag = alloc_2d_char_array(flag_size_x, flag_size_y);

    if (!u || !v || !f || !g || !p || !rhs || !flag) {
        fprintf(stderr, "Couldn't allocate memory for matrices.\n");
		exit(1);
    }
}

/**
 * @brief Free all of the arrays used for the computation.
 * 
 */
void free_arrays() {
	free_2d_array((void**) u);
    free_2d_array((void**) v);
    free_2d_array((void**) f);
    free_2d_array((void**) g);
    free_2d_array((void**) p);
    free_2d_array((void**) rhs);
    free_2d_array((void**) flag);
}

/**
 * @brief Initialise the velocity arrays and then initialize the flag array, 
 * marking any obstacle cells and the edge cells as boundaries. The cells 
 * adjacent to boundary cells have their relevant flags set too.
 */
void problem_set_up() {
    for (int i = 0; i < imax+2; i++) {
        for (int j = 0; j < jmax+2; j++) {
            u[i][j] = ui;
            v[i][j] = vi;
            p[i][j] = 0.0;
            flag[i][j] = C_F;
        }
    }

    /* Mark an airfoil obstacle as boundary cells, the rest as fluid */ 
    int thickness = airfoil % 100;
    int camber_loc = (airfoil / 100) % 10;
    int camber = (airfoil / 1000);

    double m = camber / 100.0; // maximum camber
    double p = camber_loc / 10.0; // location of maximum camber
    double t = thickness / 100.0; // thickness
    for (int i = 1; i <= imax; i++) {
        double x = ((xlength / imax) * i) - 0.5;
        double y = ((ylength / jmax) * i) - (ylength / 2.0);

        if ((x < 0.0) | (x > 1.0)) continue;

        double y_t = 5.0 * t * ((0.2969 * sqrt(x)) - (0.1260 * x) - (0.3516 * x * x) + (0.2843 * x * x * x) - (0.1015 * x * x * x * x));
        double y_c = (x <= p) ? 
                        ((m / (p * p)) * ((2.0 * p * x) - (x * x))) :
                        ((m / ((1.0 - p) * (1.0 - p))) * ((1.0 - (2.0 * p)) + (2.0 * p * x) - (x * x)));
        double dyc_dx = (x <= p) ?
                        (((2.0 * m) / (p * p)) * (p - x)) :
                        (((2.0 * m) / ((1.0 - p) * (1.0 - p))) * (p - x));
        double theta = atan(dyc_dx);
        double y_u = y_c + (y_t * cos(theta));
        double y_l = y_c - (y_t * cos(theta));
        
        int j_start = (int) floor((y_l+(ylength / 2.0)) * (jmax / ylength));
        int j_end = (int) ceil((y_u+(ylength / 2.0)) * (jmax / ylength));

        for (int jn = j_start; jn < j_end; jn++) {
            flag[i][jn] = C_B;
        }
    }
    
    /* Mark the north & south boundary cells */
    for (int i = 0; i <= imax + 1; i++) {
        flag[i][0]      = C_B;
        flag[i][jmax+1] = C_B;
    }
    /* Mark the east and west boundary cells */
    for (int j = 1; j <= jmax; j++) {
        flag[0][j]      = C_B;
        flag[imax+1][j] = C_B;
    }

    fluid_cells = imax * jmax;

    /* flags for boundary cells */
    for (int i = 1; i <= imax; i++) {
        for (int j = 1; j <= jmax; j++) {
            if (!(flag[i][j] & C_F)) {
                fluid_cells--;
                if (flag[i-1][j] & C_F) flag[i][j] |= B_W;
                if (flag[i+1][j] & C_F) flag[i][j] |= B_E;
                if (flag[i][j-1] & C_F) flag[i][j] |= B_S;
                if (flag[i][j+1] & C_F) flag[i][j] |= B_N;
            }
        }
    }

	apply_boundary_conditions();
}