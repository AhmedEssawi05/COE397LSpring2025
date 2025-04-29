# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(CommunityDetection C)

# Find MPI package (Make sure you have OpenMPI or MPICH installed)
find_package(MPI REQUIRED)

# Set the C standard (no need for C++ standard here)
set(CMAKE_C_STANDARD 11)

# Enable OpenMP support
find_package(OpenMP REQUIRED)

# Add the source files
set(SOURCES
    community_detection.c
)

# Add the executable
add_executable(community_detection ${SOURCES})

# Link MPI and OpenMP libraries
target_link_libraries(community_detection MPI::MPI_C OpenMP::OpenMP_C)

# Enable optimization for release mode
set(CMAKE_BUILD_TYPE Release)

# Specify the necessary flags for compiling with OpenMP
if(OpenMP_C_FOUND)
    target_compile_options(community_detection PRIVATE ${OpenMP_C_FLAGS})
endif()

