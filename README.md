# Continuous Deformation Based Panelization For Design Rationalization App

## Results & Reproducibility
You can find our results at 'results' folder.
We attach additional files for reproducibility, the files are:
   1. `README.md` file - has the non-default parameters
   1. `init_A.off` + `init_R.off` files - has the init values for the auxiliary variables (in case of cylinders)

## Parameters
* Homotopy method (Lambda)
    * Lambda Init 
        1. `lambda_init` - initial value for lambda (default 2^0)
    * Automatic Lambda progression - the following parameters are dependent on the count of the optimization (e.g., ADAM, Newton, Gradient-Descent, etc.) iterations.
        1. `start_from_iter` - Skip first `start_from_iter` optimization iterations, then start halving lambda (default 100).
        1. `halve_per_iters` - Halve lambda every `halve_per_iters` optimization iterations (default 70). 
        1. `stop_halving_at` - Stop halving lambda when lambda value reaches the `stop_halving_at` value (default 2^-40).
    

* Objectives
    1. AuxCylinder1
        1. `w` - objective function weight which is used at calculating the total objective function (default 0)
        1. `w1`
        1. `w2`
        1. `w3`
    1. AuxCylinder2
        1. `w` - objective function weight which is used at calculating the total objective function (default 0)
        1. `w1`
        1. `w2`
        1. `w3`
    1. AuxCylinder3
        1. `w` - objective function weight which is used at calculating the total objective function (default 0)
        1. `w1`
        1. `w2`
        1. `w3`
        1. `w_a`
        1. `w_r`
        1. `w_c`
    1. AuxSphere
        1. `w` - objective function weight which is used at calculating the total objective function (default 0)
        1. `w1`
        1. `w2`
    1. AuxPlanar
        1. `w` - objective function weight which is used at calculating the total objective function (default 0)
        1. `w1`
        1. `w2` 
        1. `w3`
    1. Planar
        1. `w` - objective function weight which is used at calculating the total objective function (default 0)
    1. STVK
        1. `w` - objective function weight which is used at calculating the total objective function (default 0)
    1. Symmetric Dirichlet
        1. `w` - objective function weight which is used at calculating the total objective function (default 0.5)
    1. Pin Vertices
        1. `w` - objective function weight which is used at calculating the total objective function (default 0.3)
    1. Pin Chosen Vertices
        1. `w` - objective function weight which is used at calculating the total objective function (default 500)
    1. Round Radius
        1. `w` - objective function weight which is used at calculating the total objective function (default 0)
        1. `min`    - Min available radius (default 2)
        1. `max`    - Max available radius (default 10)
        1. `alpha`  - Scale Radius (default 23)
    1. Uniform Smoothness
        1. `w` - objective function weight which is used at calculating the total objective function (default 0.05)
        
## Additional parameters
Those parameters don't effect the output mesh, their porpuse is to make the GUI user friendly.

For example, we can control via GUI the following parameters:
* objects-colors
* hide/unhide objects
* screen options
* Mesh coloring methods
* UI coloring/tools options
* etc


## See the tutorial first

Then build, run and understand the [libigl
tutorial](http://libigl.github.io/libigl/tutorial/).

## Dependencies

The only dependencies are STL, Eigen, [libigl](http://libigl.github.io/libigl/) and the dependencies
of the `igl::opengl::glfw::Viewer` (OpenGL, glad and GLFW).
The CMake build system will automatically download libigl and its dependencies using
[CMake FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html),
thus requiring no setup on your part.

To use a local copy of libigl rather than downloading the repository via FetchContent, you can use
the CMake cache variable `FETCHCONTENT_SOURCE_DIR_LIBIGL` when configuring your CMake project for
the first time:
```
cmake -DFETCHCONTENT_SOURCE_DIR_LIBIGL=<path-to-libigl> ..
```
When changing this value, do not forget to clear your `CMakeCache.txt`, or to update the cache variable
via `cmake-gui` or `ccmake`.

## Compile

Compile this project using the standard cmake routine:

    mkdir build
    cd build
    cmake ..
    make

This should find and build the dependencies and create a `example_bin` binary.

## Run

From within the `build` directory just issue:

    ./example

A glfw app should launch displaying a 3D cube.
