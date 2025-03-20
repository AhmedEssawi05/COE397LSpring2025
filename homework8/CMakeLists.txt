cmake_minimum_required(VERSION 3.13)
project(reductcoord VERSION 1.0)


find_package(OpenMP REQUIRED)

add_executable(reductcoord reductcoord.c)

if(OpenMP_C_FOUND)
  target_link_libraries(reductcoord PUBLIC OpenMP::OpenMP_C)
endif()


target_link_libraries(reductcoord PUBLIC m)



install(TARGETS reductcoord DESTINATION .)
