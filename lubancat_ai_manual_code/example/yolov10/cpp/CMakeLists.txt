cmake_minimum_required(VERSION 3.10)

project(rknn_yolov10_demo)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

# 从系统查找opencv
# set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
find_package(OpenCV REQUIRED)

# RGA2 only support under 4G memory
if (ENABLE_DMA32 STREQUAL "TRUE")
    add_definitions(-DDMA_ALLOC_DMA32)
endif()

# support rknn zero copy api
if (ENABLE_ZERO_COPY STREQUAL "TRUE")
    add_definitions(-DENABLE_ZERO_COPY)
endif()

# dma
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/allocator/dma)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../utils/ utils.out)

# rknn_yolov10_image_demo
add_executable(yolov10_image_demo
    src/yolov10.cc
    src/postprocess.cc
    src/yolov10_image.cc
)

target_link_libraries(yolov10_image_demo
 imageutils
 fileutils
 imagedrawing
 ${LIBRKNNRT}
 dl
)

# rknn_yolov10_videocapture_demo
add_executable(yolov10_videocapture_demo
    src/yolov10.cc
    src/postprocess.cc
    src/yolov10_videocapture_demo.cc
)

target_link_libraries(yolov10_videocapture_demo
 imageutils
 fileutils
 ${OpenCV_LIBS}
 ${LIBRKNNRT}
 dl
)

include_directories( 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LIBRKNNRT_INCLUDES}
    ${LIBTIMER_INCLUDES}
)

message(STATUS "!!!!!!!!!!!CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(yolov10_image_demo Threads::Threads)
    target_link_libraries(yolov10_videocapture_demo Threads::Threads)
endif()

install(TARGETS yolov10_image_demo DESTINATION .)
install(TARGETS yolov10_videocapture_demo DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/bus.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/coco_80_labels_list.txt DESTINATION model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)
