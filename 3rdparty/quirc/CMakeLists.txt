set(QUIRC_LIB_NAME quirc)

set(QUIRC_SOURCES
    lib/quirc.c
    lib/decode.c
    lib/identify.c
    lib/version_db.c
)

add_library(${QUIRC_LIB_NAME} STATIC ${QUIRC_SOURCES})

set_target_properties(${QUIRC_LIB_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${QUIRC_LIB_NAME} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/lib"
)

message(STATUS "quirc (minimal version) library configured with sources: ${QUIRC_SOURCES}")
