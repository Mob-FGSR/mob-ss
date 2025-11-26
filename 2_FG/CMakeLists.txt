cmake_minimum_required(VERSION 3.10.0)

project(Frame-Generation VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "debug/")

include_directories(
    source/inc/
)

add_executable(fg source/src/main.cpp)

target_link_libraries(fg d3d11.lib D3DCompiler.lib dxguid.lib)