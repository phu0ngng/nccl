# Documentation package building for NCCL

find_program(SED_EXECUTABLE sed)
find_program(MAKE_EXECUTABLE make)
find_program(ZIP_EXECUTABLE zip)

# Set package directories
set(DOCSRC "${CMAKE_SOURCE_DIR}/docs/userguide")
set(DOCPREPDIR "${CMAKE_BINARY_DIR}/doc")
set(PKGDIR "${CMAKE_BINARY_DIR}/pkg/doc")

# Get package architecture
set(PKG_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})

# Find all .in files in doc source
file(GLOB_RECURSE DOCGEN_IN_FILES RELATIVE ${DOCSRC} "${DOCSRC}/*.in")
set(DOCGEN_FILES "")
foreach(DOCGEN_IN_FILE ${DOCGEN_IN_FILES})
  string(REGEX REPLACE "\\.in$" "" DOCGEN_FILE ${DOCGEN_IN_FILE})
  list(APPEND DOCGEN_FILES ${DOCGEN_FILE})
endforeach()

# Find all non-.in files in doc source
file(GLOB_RECURSE DOCCOPY_FILES RELATIVE ${DOCSRC} "${DOCSRC}/*")
list(FILTER DOCCOPY_FILES EXCLUDE REGEX "\\.in$")
# Exclude virtual environment and other build artifacts
list(FILTER DOCCOPY_FILES EXCLUDE REGEX "^\\.venv/")
list(FILTER DOCCOPY_FILES EXCLUDE REGEX "/\\.venv/")

# Custom target for doc package preparation
add_custom_target(doc_prep ALL
  # Create doc prep directory
  COMMAND ${CMAKE_COMMAND} -E make_directory ${DOCPREPDIR}
  COMMENT "Preparing Documentation package files"
)

# Copy all non-.in files
foreach(DOCCOPY_FILE ${DOCCOPY_FILES})
  get_filename_component(DOCCOPY_DIR ${DOCPREPDIR}/${DOCCOPY_FILE} DIRECTORY)
  add_custom_command(
    TARGET doc_prep
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DOCCOPY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${DOCSRC}/${DOCCOPY_FILE} ${DOCPREPDIR}/${DOCCOPY_FILE}
    DEPENDS ${DOCSRC}/${DOCCOPY_FILE}
  )
endforeach()

# Process all .in template files
foreach(DOCGEN_IN_FILE ${DOCGEN_IN_FILES})
  string(REGEX REPLACE "\\.in$" "" DOCGEN_FILE ${DOCGEN_IN_FILE})
  get_filename_component(DOCGEN_DIR ${DOCPREPDIR}/${DOCGEN_FILE} DIRECTORY)
  add_custom_command(
    TARGET doc_prep
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DOCGEN_DIR}
    COMMAND ${SED_EXECUTABLE} -e 's/\\$$\{nccl:Major\}/${NCCL_MAJOR}/g'
                              -e 's/\\$$\{nccl:Minor\}/${NCCL_MINOR}/g'
                              -e 's/\\$$\{nccl:Patch\}/${NCCL_PATCH}/g'
                              -e 's/\\$$\{nccl:Suffix\}/${NCCL_SUFFIX}/g'
                              -e 's/\\$$\{cuda:Major\}/${CUDA_MAJOR}/g'
                              -e 's/\\$$\{cuda:Minor\}/${CUDA_MINOR}/g'
                              -e 's/\\$$\{pkg:Revision\}/${PKG_REVISION}/g'
                              -e 's/\\$$\{pkg:Arch\}/${PKG_ARCH}/g'
              ${DOCSRC}/${DOCGEN_IN_FILE} > ${DOCPREPDIR}/${DOCGEN_FILE}
    DEPENDS ${DOCSRC}/${DOCGEN_IN_FILE}
  )
endforeach()

# Build the package
add_custom_target(doc_build
  COMMAND ${MAKE_EXECUTABLE} -C ${DOCPREPDIR} html
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PKGDIR}
  COMMAND cd ${DOCPREPDIR} && ${ZIP_EXECUTABLE} -r ${PKGDIR}/nccl-doc_${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}${NCCL_SUFFIX}-${PKG_REVISION}.zip html
  WORKING_DIRECTORY ${DOCPREPDIR}
  DEPENDS doc_prep
  COMMENT "Building Documentation package"
)

# Custom target for cleaning doc package
add_custom_target(doc_clean
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${DOCPREPDIR}
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${PKGDIR}
  COMMENT "Cleaning Documentation package build files"
)
