##########################################################################################
# Author: Franklin Pereira
# Thinprep Algorithm Group
# Hologic Inc
# 250 Campus Drive Marlborough, MA
##########################################################################################

## set pkg_config_path for vips in shell
##export PKG_CONFIG_PATH=/path/to/pkgconfig/files:$PKG_CONFIG_PATH

cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
include(CMakePrintHelpers)
find_package(PkgConfig REQUIRED)

project(BladderCancer VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(EXECUTABLE_NAME BladderCancer)
set(CMAKE_SOURCE_DIR ${CMAKE_HOME_DIRECTORY}/source)
include_directories(${CMAKE_HOME_DIRECTORY}/include)

set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

######################################################################################
## locate and add yaml-cpp
######################################################################################
#find_package(yaml-cpp REQUIRED)
pkg_check_modules(YamlCpp REQUIRED yaml-cpp)
#pkg_search_module(YamlCpp REQUIRED yaml-cpp)
include_directories(${YamlCpp_INCLUDE_DIRS})

foreach(lib ${YamlCpp_LIBRARIES})
    message(STATUS "yaml-cpp lib: ${lib}")
endforeach()
######################################################################################

set(SOURCES
        ${CMAKE_HOME_DIRECTORY}/main.cpp
        ${CMAKE_SOURCE_DIR}/gridutils.cpp
        ${CMAKE_SOURCE_DIR}/test_utils.cpp
        source/mp_params.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin/${CMAKE_BUILD_TYPE})
message("cmake bin dir")
message(${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

######################################################################################
## locate pkgconfig
######################################################################################
set(ENV{PKG_CONFIG_PATH} "/usr/local/lib/x86_64-linux-gnu/pkgconfig" )
message(STATUS "PKG_CONFIG_PATH: $ENV{PKG_CONFIG_PATH}")

######################################################################################
## locate and add opencv
######################################################################################
find_package(OpenCV PATHS /usr/local)
include_directories(${OpenCV_INCLUDE_DIRS})

message("opencv includes: ")
message("${OpenCV_INCLUDE_DIRS}")
message("opencv libs: ")
foreach(lib ${OpenCV_LIBRARIES})
    message(STATUS "opencv lib: ${lib}")
endforeach()

add_executable(${EXECUTABLE_NAME} ${SOURCES})
target_link_libraries(${EXECUTABLE_NAME} ${OpenCV_LIBRARIES} ${YamlCpp_LIBRARIES})

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
    message(STATUS "dir='${dir}'")
endforeach()
