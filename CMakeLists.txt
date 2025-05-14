cmake_minimum_required(VERSION 3.8)
project(pq)

################ CXX FLAGS ##################
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

if(NOT MSVC)
    set(warnings "-Wextra -Werror -pedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread ${warnings}")
    if(OPTIMIZE_FOR_NATIVE)
        include(CheckCXXCompilerFlag)
        CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
        endif()
    endif()
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /O2 /W4")
    if (BUILD_SHARED_LIBS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DBUILD_SHARED_LIBS ")
    endif()
endif()
################ CXX FLAGS ##################

################ LIBJPEGTURBO ##################
option(ENABLE_SHARED ON)
option(WITH_TURBOJPEG ON)
option(WITH_SIMD OFF)
set(CMAKE_INSTALL_PREFIX ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
add_subdirectory(lib/libjpeg-turbo)
set(LIBJPEGTURBO_LIBRARIES jpeg-static)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/libjpeg-turbo)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/lib/libjpeg-turbo)
################ LIBJPEGTURBO ##################

set(SRCS
        main.cc
        src/jpeg_file/JPEGFile.cc
        src/jpeg_processor/JPEGProcessor.cc
        src/perturbed_quantization/PerturbedQuantization.cc
)

set(HDRS
        src/jpeg_file/JPEGFile.h
        src/jpeg_processor/JPEGProcessor.h
        src/perturbed_quantization/PerturbedQuantization.h
)

add_executable(pq
        ${SRCS}
        ${HDRS}
)

target_link_libraries(pq
        PRIVATE
        ${LIBJPEGTURBO_LIBRARIES}
)