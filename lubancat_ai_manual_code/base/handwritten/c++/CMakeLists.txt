cmake_minimum_required(VERSION 3.4.1)

project(handwritten)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 使用鲁班猫板卡系统默认安装rknn
set(RKNN_API_PATH /usr/lib/librknnrt.so)

# 使用鲁班猫板卡系统默认安装opencv
find_package(OpenCV REQUIRED)

# include
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${OpenCV_DIR} ${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(handwritten main.cc)

target_link_libraries(handwritten
  ${RKNN_API_PATH}
  ${OpenCV_LIBS}
)

