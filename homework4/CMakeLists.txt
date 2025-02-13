cmake_minimum_required( VERSION 3.13 )
project( isendirecvarray.c VERSION 1.0 )

add_executable( isendirecvarray commrank.c )
find_package( MPI )
target_include_directories(
	isendirecvarray PUBLIC
	${MPI_C_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries(
	isendirecvarray PUBLIC 
	${MPI_C_LIBRARIES} )

install( TARGETS isendirecvarray DESTINATION . )
