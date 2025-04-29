cmake_minimum_required(VERSION 3.16)
project(CommunityDetection C)

# Use C11 standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Enable Release build by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Find MPI
find_package(MPI REQUIRED)
# Find OpenMP
find_package(OpenMP REQUIRED)

# Source files
set(SOURCES
  community_detection.c
)

# Define executable
add_executable(community_detection ${SOURCES})

# Include MPI headers
target_include_directories(community_detection PRIVATE ${MPI_C_INCLUDE_PATH})

# Compile options: optimization, warnings, OpenMP and MPI flags
target_compile_options(community_detection PRIVATE
  $<$<CONFIG:Release>:-O3>
  -Wall -Wextra -Wpedantic
  ${OpenMP_C_FLAGS}
  ${MPI_C_COMPILE_FLAGS}
)

# Link libraries: MPI and OpenMP
target_link_libraries(community_detection PRIVATE
  MPI::MPI_C
  OpenMP::OpenMP_C
  ${MPI_C_LINK_FLAGS}
)
