cmake_minimum_required(VERSION 2.6)
project(reg)

add_definitions(-std=gnu99 -O3
    -D_FILE_OFFSET_BITS=64
    -D_LARGEFILE64_SOURCE)

if (WITH_OPENMP)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

set(WITH_OPENMP True CACHE BOOL "True to enable OpenMP")

if (WITH_OPENMP)
  find_package(OpenMP)
  if (NOT OPENMP_FOUND)
    set(WITH_OPENMP False)
  endif()
endif()

include_directories(${CMAKE_SOURCE_DIR})

add_executable(reg
    reg.c
    semblance.c
    su.c)

target_link_libraries(reg m OpenCL)
