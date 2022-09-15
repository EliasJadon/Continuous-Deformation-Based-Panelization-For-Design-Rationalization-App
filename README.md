# Continuous Deformation Based Panelization For Design Rationalization App


## Parameters
1. Lambda
    1. `Start_from`
    1. `Stop_at`
    1. `update_lambda_frequency`

1. Init
    1. `Type`
    1. `Fixed_radius_value`

1. Objective functions
    1. AuxCylinder1
        1. `w` - total weight (float)
        1. `w1`
        1. `w2`
        1. `w3`
    1. AuxCylinder2
        1. `w` - total weight (float)
        1. `w1`
        1. `w2`
        1. `w3`
    1. AuxCylinder3
        1. `w` - total weight (float)
        1. `w1`
        1. `w2`
        1. `w3`
        1. `w_a`
        1. `w_r`
        1. `w_c`
    1. AuxSphere
        1. `w` - total weight (float)
        1. `w1`
        1. `w2`
    1. AuxPlanar
        1. `w` - total weight (float)
        1. `w1`
        1. `w2` 
        1. `w3`
    1. Planar
        1. `w` - total weight (float)
    1. STVK
        1. `w` - total weight (float)
    1. Symmetric Dirichlet
        1. `w` - total weight (float)
    1. Pin Vertices
        1. `w` - total weight (float)
    1. Pin Chosen Vertices
        1. `w` - total weight (float)
    1. Round Radius
        1. `w`      - total weight (flaot)
        1. `min`    - Min available radius (unsigned int)
        1. `max`    - Max available radius (unsigned int)
        1. `alpha`  - Scale Radius (float)
    1. Uniform Smoothness
        1. `w`      - total weight (float)
        
        

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
