cmake_minimum_required(VERSION 3.13)
project(threadplot VERSION 1.0 LANGUAGES CUDA)

# Optionally set the CUDA standard (e.g., C++11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Create the executable from the CUDA source file
add_executable(threadplot threadplot.cu)

# Installation rule (install the executable to the current directory)
install(TARGETS threadplot DESTINATION .)
