# Parallel Airfoil Solver

A two-dimensional computational fluid dynamics project for simulating flow around symmetric and cambered four-digit NACA airfoils. The same finite-difference pressure-projection solver is presented in three parallel implementations so its shared-memory, distributed-memory, and GPU approaches can be studied side by side.

## Implementations

- **OpenMP** (`omp/airfoil`) parallelizes the CPU loops and red/black SOR pressure solve.
- **MPI** (`mpi/airfoil`) partitions the grid in the x direction, exchanges halo cells, performs global reductions, and gathers fields for VTK output.
- **CUDA** (`cuda/airfoil`) moves flattened simulation fields to the GPU and uses CUDA kernels for velocity, pressure, residual, CFL, and boundary calculations. Its Makefile also builds a serial CPU reference binary.

The default setup models a NACA 2412 airfoil in a 4.0 m × 1.0 m domain on a 1024 × 256 grid through 2 seconds of simulated time. Results are written as legacy VTK files for tools such as [VisIt](https://visit-dav.github.io/visit-website/) or ParaView.

## Repository layout

```text
.
├── omp/airfoil/   # OpenMP implementation
├── mpi/airfoil/   # MPI implementation
└── cuda/airfoil/  # CUDA implementation plus serial reference
```

Each directory is self-contained and includes the shared C support files, a Makefile, and implementation-specific solver source.

## Requirements

- Linux or another POSIX-like environment
- GNU Make and a C compiler
- OpenMP support for the OpenMP version (GCC uses `-fopenmp`)
- an MPI implementation providing `mpicc` and `mpirun` for the MPI version
- the NVIDIA CUDA toolkit and a compatible GPU for the CUDA version

The CUDA Makefile currently targets compute capability `sm_80`. Change `CUDAFLAGS` if your GPU requires another architecture.

## Build and run

Choose an implementation and build inside its directory.

### OpenMP

```bash
cd omp/airfoil
make
OMP_NUM_THREADS=8 ./airfoil
```

### MPI

```bash
cd mpi/airfoil
make
mpirun -np 4 ./airfoil
```

The MPI implementation contains first-iteration diagnostic output in addition to normal progress messages.

### CUDA

```bash
cd cuda/airfoil
make
./airfoil_cuda
```

`make` also creates `./airfoil`, the serial CPU reference in that directory.

## Runtime options

All binaries accept the same core options. Inspect the full list with:

```bash
./airfoil --help
# or, for the GPU executable
./airfoil_cuda --help
```

A small, output-free smoke run can be requested with:

```bash
./airfoil -x 64 -y 16 -t 0.01 -n
```

Select another four-digit profile with `-a`, set the grid using `-x` and `-y`, and control simulated end time using `-t`. Without `-n`, the final field is written to a VTK file. Checkpoint output can be enabled with `-c` and an output basename supplied with `-o`.

## Numerical method and scope

The solver uses a staggered-grid finite-difference scheme, adaptive CFL-limited timesteps, donor-cell convection terms, and red/black successive over-relaxation for the pressure Poisson equation. It is an educational implementation rather than a validated production aerodynamics package: no benchmark suite, force-coefficient calculation, turbulence model, or quantitative cross-implementation results are included.

## Attribution

The original implementation describes the computational scheme from Michael Griebel, Thomas Dornseifer, and Tilman Neunhoeffer, *Numerical Simulation in Fluid Dynamics* (SIAM, 1998), with a related [NAST2D reference implementation](https://people.math.sc.edu/Burkardt/cpp_src/nast2d/nast2d.html). Existing source help text retains the original bug-report contact for Steven Wright at the University of York.
