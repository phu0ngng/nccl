# Documentation package building for NCCL

find_program(MAKE_EXECUTABLE make)
find_program(ZIP_EXECUTABLE zip)

# Set package directories
set(DOCSRC "${CMAKE_SOURCE_DIR}/docs/userguide")
set(DOCDIR "${CMAKE_BINARY_DIR}/doc")
set(PKGDIR "${CMAKE_BINARY_DIR}/pkg/doc")

# Build the documentation and package it
# The Makefile preprocesses source/ -> BUILDDIR/doc/ and runs sphinx -> BUILDDIR/html
add_custom_target(doc_build
  COMMAND ${MAKE_EXECUTABLE} -C ${DOCSRC} BUILDDIR=${CMAKE_BINARY_DIR} html
  COMMAND ${MAKE_EXECUTABLE} -C ${DOCSRC}/../.. BUILDDIR=${CMAKE_BINARY_DIR} pkg.doc.build
  COMMENT "Building and packaging Documentation"
)

# Custom target for cleaning doc build artifacts
add_custom_target(doc_clean
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${DOCDIR}
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/html
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/doctrees
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${PKGDIR}
  COMMENT "Cleaning Documentation build files"
)
