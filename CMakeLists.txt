cmake_minimum_required(VERSION 3.16)
project(Mesh-Simplification-App)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
option(LIBIGL_GLFW "Build target igl::glfw" ON)
include(libigl)

# Add your project files
file(GLOB SRC_FILES 
		src/*.cpp src/*.h
		src/*/*.cpp src/*/*.h
		src/*/*/*.cpp src/*/*/*.h)

add_executable(${PROJECT_NAME} ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME} PUBLIC 
		igl::glfw 
		igl::core
		igl::opengl
		igl::imgui)
