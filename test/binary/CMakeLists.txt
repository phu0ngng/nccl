# Binary tests configuration
set(SRC_FILES test_lib.sh test_symbols.sh)

# Copy shell scripts to build directory with executable permissions
foreach(SH_FILE IN LISTS SRC_FILES)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/test/binary/${SH_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/test/binary
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/${SH_FILE}
            ${CMAKE_BINARY_DIR}/test/binary/${SH_FILE}
        COMMAND chmod 755 ${CMAKE_BINARY_DIR}/test/binary/${SH_FILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SH_FILE}
        COMMENT "Grabbing ${SH_FILE}"
    )
endforeach()

# Create a target that depends on the copied files
set(SH_FILES "")
foreach(SH_FILE IN LISTS SRC_FILES)
    list(APPEND SH_FILES ${CMAKE_BINARY_DIR}/test/binary/${SH_FILE})
endforeach()

add_custom_target(binary_tests ALL
    DEPENDS ${SH_FILES}
)
