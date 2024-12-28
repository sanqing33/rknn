cmake_minimum_required(VERSION 3.10)

project(rknn_ppocr_system_demo)
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

find_package(OpenCV REQUIRED)

# rga
set(RGA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/librga)
set(LIBRGA ${RGA_PATH}/${CMAKE_SYSTEM_NAME}/aarch64/librga.a)
set(RGA_INCLUDES ${RGA_PATH}/include)
install(PROGRAMS ${RGA_PATH}/${CMAKE_SYSTEM_NAME}/aarch64/librga.so DESTINATION lib)

# rknn runtime
if (TARGET_SOC STREQUAL "rk3588" OR TARGET_SOC STREQUAL "rk356x")
    set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/rknpu2)
    set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/aarch64/librknnrt.so)
    set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include)
endif()
install(PROGRAMS ${LIBRKNNRT} DESTINATION lib)
set(LIBRKNNRT ${LIBRKNNRT})

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_executable(${PROJECT_NAME}
    main.cc
    postprocess.cc
    clipper.cc
    image_utils.c
    rknpu2/ppocr_system.cc
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${LIBRGA}
    ${LIBRKNNRT}
)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()


target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${RGA_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/test.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/11.jpg DESTINATION model)

set(file_path ${CMAKE_CURRENT_SOURCE_DIR}/../model/ppocrv4_det_rk3588.rknn)
if (EXISTS ${file_path})
    install(FILES ${file_path} DESTINATION model)
endif()


set(file_path ${CMAKE_CURRENT_SOURCE_DIR}/../model/ch_ppocr_mobile_v2.0_cls_rk3588.rknn)
if (EXISTS ${file_path})
    install(FILES ${file_path} DESTINATION model)
endif()


set(file_path ${CMAKE_CURRENT_SOURCE_DIR}/../model/ppocrv4_rec_rk3588.rknn)
if (EXISTS ${file_path})
    install(FILES ${file_path} DESTINATION model)
endif()

