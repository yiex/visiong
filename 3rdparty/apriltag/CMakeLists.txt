set(APRILTAG_LIB_NAME apriltag)

set(APRILTAG_SOURCES
    apriltag.c
    apriltag_pose.c
    apriltag_quad_thresh.c
    tag16h5.c
    tag25h9.c
    tag36h10.c
    tag36h11.c
    tagCircle21h7.c
    tagStandard41h12.c
    common/g2d.c
    common/homography.c
    common/image_u8.c
    common/image_u8_parallel.c
    common/image_u8x3.c
    common/matd.c
    common/pthreads_cross.c
    common/pnm.c
    common/svd22.c
    common/time_util.c
    common/unionfind.c
    common/workerpool.c
    common/zarray.c
    common/zmaxheap.c
)

add_library(${APRILTAG_LIB_NAME} STATIC ${APRILTAG_SOURCES})
set_target_properties(${APRILTAG_LIB_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON C_STANDARD 99)

target_include_directories(${APRILTAG_LIB_NAME} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(NOT WIN32)
    target_compile_definitions(${APRILTAG_LIB_NAME} PRIVATE _GNU_SOURCE)
    target_link_libraries(${APRILTAG_LIB_NAME} PUBLIC pthread m)
endif()
target_compile_options(${APRILTAG_LIB_NAME} PRIVATE -w)

message(STATUS "AprilTag v3.4.5 minimal library configured with sources: ${APRILTAG_SOURCES}")
