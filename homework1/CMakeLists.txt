cmake_minimum_required( VERSION 3.13 )
project( commrank.c VERSION 1.0 )

add_executable( commrank.c commrank.c )
find_package( MPI )
target_include_directories(
	commrank.c PUBLIC
	${MPI_C_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries(
	commrank.c PUBLIC 
	${MPI_C_LIBRARIES} )

install( TARGETS commrank.c DESTINATION . )
