cmake_minimum_required(VERSION 3.13)
project(vectorsum VERSION 1.0 LANGUAGES CUDA)

add_executable(sharedmem sharedmem.cu)

# You can add additional include directories if needed.
target_include_directories(sharedmem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries if necessary (for example, if your CUDA code uses extra libraries).
# target_link_libraries(sharedmem PUBLIC <your_libraries_here>)

install(TARGETS sharedmem DESTINATION .)
