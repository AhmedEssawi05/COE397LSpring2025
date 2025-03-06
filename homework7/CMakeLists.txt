cmake_minimum_required( VERSION 3.13 )
project( vectorsum VERSION 1.0 )

add_executable( vectorsum vectorsum.c )
find_package( MPI )
target_include_directories(
	vectorsum PUBLIC
	${MPI_C_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries(
	vectorsum PUBLIC 
	${MPI_C_LIBRARIES} )

install( TARGETS vectorsum DESTINATION . )