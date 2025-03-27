cmake_minimum_required(VERSION 3.13)
project(sudokusolver VERSION 1.0)

find_package(OpenMP REQUIRED)

add_executable(sudokusolver sudokusolver.c)

if(OpenMP_C_FOUND)
  target_link_libraries(sudokusolver PUBLIC OpenMP::OpenMP_C)
endif()


install(TARGETS sudokusolver DESTINATION .)
