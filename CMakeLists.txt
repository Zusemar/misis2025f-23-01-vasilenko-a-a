cmake_minimum_required(VERSION 3.20)

project(misis2025f-micv)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

set(CMAKE_CXX_STANDARD 20)


find_package(OpenCV REQUIRED)

include_directories(prj.lab)

add_subdirectory(prj.lab)