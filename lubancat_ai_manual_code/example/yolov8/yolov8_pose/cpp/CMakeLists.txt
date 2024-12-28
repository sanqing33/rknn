cmake_minimum_required(VERSION 3.10)

project(rknn_yolov8_pose_demo)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

# 从系统查找opencv
find_package(OpenCV REQUIRED)

# RGA2 only support under 4G memory
if (ENABLE_DMA32 STREQUAL "TRUE")
    add_definitions(-DDMA_ALLOC_DMA32)
endif()

# dma
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/allocator/dma)

# 3rdparty and utils
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

# yolov8_pose_image_demo
add_executable(yolov8_pose_image_demo
    src/yolov8_pose.cc
    src/postprocess.cc
    src/yolov8_pose_image.cc
)

target_link_libraries(yolov8_pose_image_demo
 imageutils
 fileutils
 imagedrawing
 ${LIBRKNNRT}
 dl
)

# yolov8_pose_videocapture_demo.cc
add_executable(yolov8_pose_videocapture_demo
    src/yolov8_pose.cc
    src/postprocess.cc
    src/yolov8_pose_videocapture_demo.cc
)

target_link_libraries(yolov8_pose_videocapture_demo
 imageutils
 fileutils
 ${LIBRKNNRT}
 ${OpenCV_LIBS}
 dl
)

include_directories( 
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${LIBRKNNRT_INCLUDES}
    ${LIBTIMER_INCLUDES}
)

message(STATUS "!!!!!!!!!!!CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(yolov8_pose_image_demo Threads::Threads)
    target_link_libraries(yolov8_pose_videocapture_demo Threads::Threads)
endif()

install(TARGETS yolov8_pose_image_demo DESTINATION .)
install(TARGETS yolov8_pose_videocapture_demo DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/bus.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/yolov8_pose_labels_list.txt DESTINATION model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)
