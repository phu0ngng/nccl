# Add custom target for binary tests
add_custom_target(binary_tests
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_symbols.sh
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_lib.sh
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test/binary
    COMMENT "Running binary compatibility tests"
)
