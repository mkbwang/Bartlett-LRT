cmake_minimum_required(VERSION 3.26)
project(Bartlett_LRT)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Armadillo REQUIRED)
SET(NLOPT_INCLUDE_DIR "/usr/local/include" "/usr/include/")
SET(NLOPT_LIBRARY "/usr/local/lib/libnlopt.so")
SET(NLOPT_LIBRARIES "/usr/local/lib/libnlopt.so.0" "/usr/local/lib/libnlopt.so.0.11.1")
SET(ARMADILLO_LIBRARIES "/usr/lib/x86_64-linux-gnu/libarmadillo.so" "/usr/lib/x86_64-linux-gnu/libarmadillo.so.12" "/usr/lib/x86_64-linux-gnu/libarmadillo.so.12.4.2")

add_executable(Bartlett_LRT main.cpp Tobit.cpp Tobit.h)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_include_directories(Bartlett_LRT PUBLIC ${ARMADILLO_INCLUDE_DIRS} ${NLOPT_INCLUDE_DIR})
target_link_libraries(Bartlett_LRT PUBLIC ${NLOPT_LIBRARY} ${NLOPT_LIBRARIES} ${ARMADILLO_LIBRARIES})
