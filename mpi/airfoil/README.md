# A Computational Fluid Dynamics Solver for Simulating Symmetric and Cambered 4-Digit NACA Airfoils

This is a simple CFD simulation application, implemented according to the computational scheme by Griebel et al. [1]. It is set up to simulate fluid flow around NACA airfoils, a problem often studied using CFD.

By default, the problem is set up with a 4.0m x 1.0m domain, discretised with a 1024 x 256 grid, simulating 2 seconds of time of fluid flowing passing a NACA 2412 airfoil.

At the end of execution, a VTK file is produced for visualisation purposes. This can be loaded into VisIt for analysis.

## Building

The application can be built with the provided Makefile. e.g.

```
$ make
```

This will build an `airfoil` binary.

## Running

The application can be ran in its default configuration with:

```
$ ./airfoil
```

This will output its status every 100 iterations. At the end of execution a VTK file will be produced. 

There are numerous other options available. These can be queried with:

```
$ ./airfoil --help
```

To write out the simulation state every 100 iterations, you could use:

```
$ mkdir out
$ ./airfoil -c -o out/my_sim
```

## References

[1]: Michael Griebel, Thomas Dornseifer, Tilman Neunhoeffer, “Numerical Simulation in Fluid Dynamics”, SIAM, 1998. [https://people.math.sc.edu/Burkardt/cpp_src/nast2d/nast2d.html](https://people.math.sc.edu/Burkardt/cpp_src/nast2d/nast2d.html)
