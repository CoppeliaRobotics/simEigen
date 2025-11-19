cmake_minimum_required(VERSION 3.16.3)
project(simEigen)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_MACOSX_RPATH 1)

if(NOT COPPELIASIM_INCLUDE_DIR)
    if(DEFINED ENV{COPPELIASIM_ROOT_DIR})
        set(COPPELIASIM_INCLUDE_DIR $ENV{COPPELIASIM_ROOT_DIR}/programming/include)
    else()
        set(COPPELIASIM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    endif()
endif()
list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${COPPELIASIM_INCLUDE_DIR}/cmake)
find_package(CoppeliaSim 4.5.1.7 REQUIRED COMPONENTS simLib-1 simPlusPlus-1)
find_package(Eigen3 REQUIRED)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)

coppeliasim_generate_stubs(${CMAKE_CURRENT_BINARY_DIR}/generated
    XML_FILE ${CMAKE_CURRENT_SOURCE_DIR}/sourceCode/callbacks.xml
    LUA_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/lua/simEigen/Matrix.lua
        ${CMAKE_CURRENT_SOURCE_DIR}/lua/simEigen/Vector.lua
        ${CMAKE_CURRENT_SOURCE_DIR}/lua/simEigen/Quaternion.lua
        ${CMAKE_CURRENT_SOURCE_DIR}/lua/simEigen/Pose.lua
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/sourceCode/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

coppeliasim_add_plugin(simEigen SOURCES sourceCode/plugin.cpp)
target_link_libraries(simEigen PRIVATE Eigen3::Eigen)

coppeliasim_add_lua_directory(lua/simEigen)
