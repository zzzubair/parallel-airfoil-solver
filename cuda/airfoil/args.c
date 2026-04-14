#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>

#include "args.h"
#include "data.h"
#include "vtk.h"

int verbose = 0;
int no_output = 0;
int output_freq = 100;
int enable_checkpoints = 0;
int fixed_dt = 0;

static struct option long_options[] = {
	{"airfoil",    required_argument, 0, 'a'},
	{"del-t",      required_argument, 0, 'd'},
	{"cellx",      required_argument, 0, 'x'},
	{"celly",      required_argument, 0, 'y'},
	{"freq",       required_argument, 0, 'f'},
	{"endtime",    required_argument, 0, 't'},
	{"noio",       no_argument,       0, 'd'},
	{"output",     required_argument, 0, 'o'},
	{"checkpoint", no_argument,       0, 'c'},	
    {"verbose",    no_argument,       0, 'v'},
    {"help",       no_argument,       0, 'h'},
	{0, 0, 0, 0}
};
#define GETOPTS "a:d:x:y:f:t:no:cvh"

/**
 * @brief Print a help message
 * 
 * @param progname The name of the current application
 */
void print_help(char *progname) {
	fprintf(stderr, "A simple computational fluid dynamics solver for NACA 4-digit Airfoils.\n\n");
	fprintf(stderr, "Usage: %s [options]\n", progname);
	fprintf(stderr, "Options and arguments:\n");
	fprintf(stderr, "  -a N, --airfoil=N       NACA Four-digit Airfoil\n");
	fprintf(stderr, "  -x N, --cellx=N         Cells in X-dimension\n");
	fprintf(stderr, "  -y N, --celly=N         Cells in Y-dimension\n");
	fprintf(stderr, "  -t N, --endtime=N       Set the end time (see -n)\n");
    fprintf(stderr, "  -d, --del-t=DELT        Set the simulation timestep size\n");
	fprintf(stderr, "  -f N, --freq=N          Output frequency (i.e. steps between output)\n");
	fprintf(stderr, "  -n, --noio              Disable file I/O\n");
	fprintf(stderr, "  -o FILE, --output=FILE  Set base filename for output (final output will be in BASENAME.vtk\n");
	fprintf(stderr, "  -c, --checkpoint        Enable checkpointing, checkpoints will be in BASENAME-ITERATION.vtk\n");
	fprintf(stderr, "  -v, --verbose           Set verbose output\n");
	fprintf(stderr, "  -h, --help              Print this message and exit\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Report bugs to <steven.wright@york.ac.uk>\n");
}

/**
 * @brief Parse the argv arguments passed to the application
 * 
 * @param argc The number of arguments present
 * @param argv An array of the arguments presented
 */
void parse_args(int argc, char *argv[]) {
    int option_index = 0;
    char c;

    while ((c = getopt_long(argc, argv, GETOPTS, long_options, &option_index)) != -1) {
        switch (c) {
			case 'a':
				// check the length of the NACA spec is 4 characters exactly
				if (strlen(optarg) == 4) {
					int tmp1, tmp2, tmp3, tmp4;
					// check that all 4 characters are digits
					if (sscanf(optarg, "%1d%1d%1d%1d", &tmp1, &tmp2, &tmp3, &tmp4) == 4) {
						sscanf(optarg, "%d", &airfoil);
						break;
					}
				}
				fprintf(stderr, "Error: NACA Airfoil Specification must be 4 digits.\n");
				print_help(argv[0]);
				exit(1);
			case 'x':
				imax = atoi(optarg);
				break;
			case 'y':
				jmax = atoi(optarg);
				break;
			case 't':
                t_end = atof(optarg);
				break;
            case 'd':
                fixed_dt = 1;
                del_t = atof(optarg);
                break;
			case 'f':
				output_freq = atoi(optarg);
				break;
			case 'n':
				no_output = 1;
				break;
			case 'o':
				set_basename(optarg);
				break;
			case 'c':
				enable_checkpoints = 1;
				break;
			case 'v':
				verbose = 1;
				break;
			case '?':
            case 'h':
				print_help(argv[0]);
				exit(1);
        }
    }
}

/**
 * @brief Print out the current parameters
 * 
 */
void print_opts() {
    printf("=======================================\n");
    printf("Started with the following options\n");
    printf("=======================================\n");
	printf("  airfoil          = %d\n", airfoil);
	printf("  cellx            = %14d\n", imax);
	printf("  celly            = %14d\n", jmax);
	printf("  endtime          = %14.12lf\n", t_end);
    printf("  del-t            = %14lf\n", del_t);
	printf("  freq             = %14d\n", output_freq);
	printf("  noio             = %14d\n", no_output);
	printf("  output           = %s\n", get_basename());
	printf("  checkpoint       = %14d\n", enable_checkpoints);	
    printf("=======================================\n");
}