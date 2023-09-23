cmake_minimum_required(VERSION 3.26)
project(Bartlett_LRT)

set(CMAKE_CXX_STANDARD 11)

add_executable(Bartlett_LRT main.cpp Tobit.cpp)
