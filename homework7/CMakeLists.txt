cmake_minimum_required(VERSION 3.13)
project(vectorsum VERSION 1.0)


find_package(OpenMP REQUIRED)


add_executable(vectorsum vectorsum.c)


if(OpenMP_C_FOUND)
  target_link_libraries(vectorsum PUBLIC OpenMP::OpenMP_C)
endif()
target_link_libraries(vectorsum PUBLIC m)


install(TARGETS vectorsum DESTINATION .)
