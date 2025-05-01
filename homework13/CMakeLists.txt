cmake_minimum_required(VERSION 3.10)
project(PETSc_MatVec C)

set (CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0072 NEW)

set(CMAKE_C_COMPILER mpicc)

find_package(MPI REQUIRED)

set(PETSC_DIR $ENV{TACC_PETSC_DIR})
include_directories(${PETSC_DIR}/include)

add_executable(matvec matvec.c)

target_link_libraries(matvec
    -L${PETSC_DIR}/lib
    -lpetsc
    ${MPI_C_LIBRARIES}    # ← BAD: the leading “-” gets prepended to the entire path
)

target_include_directories(matvec PRIVATE ${MPI_C_INCLUDE_PATH})