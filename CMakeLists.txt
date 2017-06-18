cmake_minimum_required (VERSION 2.6)
project(densecrf)

find_package(OpenCV 3 REQUIRED core features2d highgui imgproc imgcodecs ml PATHS "$ENV{OPENCVDIR}/build")
find_package(DGM 1.5 REQUIRED PATHS "$ENV{DGMDIR}/build/install")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")	
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS} /DEBUG /INCREMENTAL:NO /OPT:REF /OPT:ICF")					
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS} /DEBUG /INCREMENTAL:NO /OPT:REF /OPT:ICF")

include_directories(include ${OpenCV_INCLUDE_DIRS} ${DGM_INCLUDE_DIRS})

add_subdirectory (src)
add_subdirectory (examples)
